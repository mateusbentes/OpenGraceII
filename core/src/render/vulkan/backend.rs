//! The Vulkan render backend: the engine's own loader, own code.
//!
//! The backend creates the Vulkan instance and device itself, the
//! off-screen render target carries the diagnostic triangle, and the
//! readback path copies the pixels through a staging buffer. The module
//! never links a Vulkan binding: every constant, structure and entry
//! point is declared in `vk_loader.rs` (lesson 3), and the platform's
//! `libvulkan.so.1` is dlopen()ed at runtime.
//!
//! The backend is headless: the frame target is an off-screen render
//! target, no windowing system and no swapchain. The hardware validation
//! on Android hardware runs through the same path (a hardware validation
//! milestone 2 records).

use std::sync::{Arc, Mutex};

use super::vk_loader as vk;
use super::vk_loader::VkLoader;
use super::shader_library;
use crate::math::{Mat4, Vec3};
use crate::render::*;

/// The diagnostic triangle's uniform block (std140: the MVP row-major
/// followed by the RGBA colour).
#[repr(C)]
struct TriUniform {
    mvp: [[f32; 4]; 4],
    color: [f32; 4],
}

/// The generic mesh's uniform block (std140: the view-projection matrix
/// the set_camera path builds and the model matrix the submit() copies).
/// Both shader stages that read the camera declare the same block (the
/// shared-uniform rule the Metal bug taught).
#[repr(C)]
struct MeshUniform {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

/// The lights' uniform block (std140, 256 bytes): the sun's direction
/// (xyz + pad), the sun's colour (xyz) and intensity (w), the four point
/// lights' position (xyz) and radius (w), the four point lights' colour
/// (xyz) and intensity (w), the params (x = the live point-light count,
/// yzw pad). The same byte layout every backend maps (the M4c rule
/// extended to M5's lights).
#[repr(C)]
#[derive(Clone, Copy)]
struct LightsUniform {
    sun_direction: [f32; 4],
    sun_color_intensity: [f32; 4],
    point_position_radius: [[f32; 4]; 4],
    point_color_intensity: [[f32; 4]; 4],
    params: [f32; 4],
    // The tail pad: the block's byte size the uniform buffer carries
    // (256; the declared fields land at 176, the pad completes the
    // buffer the CBV reads — the copies' 256-byte reads stay in bounds).
    _pad: [f32; 20],
}

impl LightsUniform {
    /// The block's byte size (256; the D3D12 path's CBV alignment rule
    /// requires the 256-byte alignment and the std140 layout lands at
    /// exactly 256).
    const SIZE: usize = 256;

    /// The default block: no lights (the intensity zero, the count
    /// zero — the lit draw renders the texture unlit rather than black).
    fn none() -> Self {
        LightsUniform {
            sun_direction: [0.0, -1.0, 0.0, 0.0],
            sun_color_intensity: [0.0, 0.0, 0.0, 0.0],
            point_position_radius: [[0.0; 4]; 4],
            point_color_intensity: [[0.0; 4]; 4],
            params: [0.0, 0.0, 0.0, 0.0],
            _pad: [0.0; 20],
        }
    }

    /// The block from the contract's lights: the sun (the direction
    /// normalised) plus up to four point lights (the rest stay zero).
    fn from_lights(sun: &DirectionalLight, lights: &[DynamicLight]) -> Self {
        let direction = sun.direction.normalised();
        let mut block = LightsUniform {
            sun_direction: [direction.x, direction.y, direction.z, 0.0],
            sun_color_intensity: [
                sun.color.x, sun.color.y, sun.color.z, sun.intensity,
            ],
            point_position_radius: [[0.0; 4]; 4],
            point_color_intensity: [[0.0; 4]; 4],
            params: [0.0, 0.0, 0.0, 0.0],
            _pad: [0.0; 20],
        };
        for (index, dynamic) in lights.iter().take(4).enumerate() {
            // The spot light's cone falloff waits for the next step (the
            // first lit step carries the point lights; the spot enters
            // the block as its position/direction's approximation later).
            if let DynamicLight::Point(light) = dynamic {
                block.point_position_radius[index] = [
                    light.position.x,
                    light.position.y,
                    light.position.z,
                    light.radius,
                ];
                block.point_color_intensity[index] = [
                    light.color.x,
                    light.color.y,
                    light.color.z,
                    light.intensity,
                ];
            }
        }
        block.params[0] = lights.len().min(4) as f32;
        block
    }
}

/// The light volume's uniform block (std140, 48 bytes): the grid's
/// origin (xyz + pad), the grid's dims (x, y, z * 6 the depth layers,
/// pad) and the params (x = the cell size, y = the indirect intensity,
/// z = the toggle). The same three vec4s the POST_GBUFFER fragment's
/// Volume block declares.
#[repr(C)]
#[derive(Clone, Copy)]
struct VolumeUniform {
    origin: [f32; 4],
    dims: [f32; 4],
    params: [f32; 4],
}

/// The Vulkan backend's configuration.
#[derive(Clone, Debug)]
pub struct VulkanBackendConfig {
    pub width: u32,
    pub height: u32,
    /// Draw the diagnostic triangle each frame.
    pub draw_triangle: bool,
    pub triangle_color: ClearColor,
    /// Read back the centre pixel before the frame ends.
    pub readback_center: bool,
    pub clear_buffers: ClearBuffer,
    pub clear_depth: f32,
    pub clear_stencil: i32,
}

impl Default for VulkanBackendConfig {
    fn default() -> Self {
        VulkanBackendConfig {
            width: 1280,
            height: 720,
            draw_triangle: false,
            triangle_color: ClearColor {
                r: 200,
                g: 60,
                b: 40,
                a: 255,
            },
            readback_center: false,
            clear_buffers: ClearBuffer::COLOR_DEPTH_STENCIL,
            clear_depth: 1.0,
            clear_stencil: 0,
        }
    }
}

/// The Vulkan backend's own buffer/image: a device handle plus the memory
/// binding the backend owns.
// The released field is written at creation; the release path (the next
// milestones) reads it.
#[allow(dead_code)]
struct VulkanResource {
    handle: vk::VkBuffer,
    memory: vk::VkDeviceMemory,
    released: bool,
}

/// The backend-owned state of one uploaded generic mesh: the vertex and
/// index buffers (host-visible, the CPU writes them directly), the per-mesh
/// uniform buffer and the descriptor set (the camera block plus the
/// combined image sampler the material binds).
// The stride, the uniform buffer's handle and the vertex count record the
// upload's shape; the draw path reads the buffers and the set (the dead
// code rule keeps the record fields quiet).
#[allow(dead_code)]
struct VulkanMeshState {
    stride: u32,
    vertex_buffer: VulkanResource,
    index_buffer: VulkanResource,
    uniform_buffer: VulkanResource,
    uniform_map: *mut std::os::raw::c_void,
    descriptor_set: vk::VkDescriptorSet,
    vertex_count: u32,
    index_count: u32,
}

/// The Vulkan backend's own texture: a device image plus the memory
/// binding and the view.
// The image/memory/view fields are written at creation; the release path
// (the next milestones) reads them.
#[allow(dead_code)]
struct VulkanTexture2D {
    inner: Arc<Mutex<TextureViewInner>>,
    image: vk::VkImage,
    memory: vk::VkDeviceMemory,
    view: vk::VkImageView,
    released: bool,
}

/// The Vulkan backend's own render target: one colour attachment per
/// colour attachment, one depth attachment.
// The render_pass/framebuffer fields are written at creation and read by
// the off-screen pass wiring (the next milestones wire them).
#[allow(dead_code)]
struct VulkanRenderTarget {
    inner: Arc<Mutex<RenderTargetInner>>,
    views: Vec<VulkanTexture2D>,
    depth_view: Option<VulkanTexture2D>,
    render_pass: vk::VkRenderPass,
    framebuffer: vk::VkFramebuffer,
    released: bool,
}

/// The headless Vulkan render backend.
// The library handle and the diagnostic-triangle bookkeeping fields are
// written at setup; the release path (the next milestones) reads them.
#[allow(dead_code)]
pub struct VulkanBackend {
    config: VulkanBackendConfig,
    loader: VkLoader,
    library: *mut std::os::raw::c_void,
    instance: vk::VkInstance,
    physical_device: vk::VkPhysicalDevice,
    // The picked device's identity (the describe() reports it).
    api_version: u32,
    vendor_id: u32,
    device_name: [std::os::raw::c_char; vk::MAX_EXTENSION_NAME_SIZE],
    device: vk::VkDevice,
    queue: vk::VkQueue,
    queue_family: u32,
    command_pool: vk::VkCommandPool,
    command_buffer: vk::VkCommandBuffer,
    // The readback's own one-off command buffer (the readback path
    // records the copy into it and submits it separately).
    readback_command: vk::VkCommandBuffer,
    // The default framebuffer's depth image (the off-screen target's
    // depth attachment; the contract's clear() clears depth/stencil on it).
    depth_image: vk::VkImage,
    depth_memory: vk::VkDeviceMemory,
    depth_view: vk::VkImageView,
    depth_format: vk::Format,
    render_pass: vk::VkRenderPass,
    framebuffer: vk::VkFramebuffer,
    // The default framebuffer's colour image (the off-screen target's
    // colour attachment; the contract's clear() clears colour on it).
    color_image: vk::VkImage,
    color_memory: vk::VkDeviceMemory,
    color_view: vk::VkImageView,
    // The readback staging buffer (host-visible, mapped for the CPU).
    readback_buffer: VulkanResource,
    readback_map: *mut std::os::raw::c_void,
    // The default texture (1x1 opaque white).
    default_texture: Option<VulkanTexture2D>,
    // The generic mesh path's state: one slot per create_gpu_mesh, the
    // buffers and the descriptor set the draw path binds.
    meshes: Vec<Option<VulkanMeshState>>,
    /// The sampler unit 0's bound texture: the texture's index into the
    /// textures_2d vector (None means the default texture's view).
    bound_texture: Option<usize>,
    mesh_pipeline: vk::VkPipeline,
    mesh_pipeline_ready: bool,
    mesh_pipeline_layout: vk::VkPipelineLayout,
    mesh_set_layout: vk::VkDescriptorSetLayout,
    mesh_pool: vk::VkDescriptorPool,
    mesh_sampler: vk::VkSampler,
    // The lit mesh path's state (M5): the lit pipeline (the same vertex
    // shape, the LIT shaders), its descriptor set layout (the camera
    // block plus the sampler plus the lights' block) and the lights'
    // uniform buffer (one shared host buffer the draws write).
    lit_pipeline: vk::VkPipeline,
    lit_pipeline_ready: bool,
    lit_pipeline_layout: vk::VkPipelineLayout,
    lit_set_layout: vk::VkDescriptorSetLayout,
    lit_lights_buffer: Option<VulkanResource>,
    lit_lights_map: *mut std::os::raw::c_void,
    lit_block: LightsUniform,
    lit_descriptor_set: vk::VkDescriptorSet,
    lit_camera_buffer: Option<VulkanResource>,
    lit_camera_map: *mut std::os::raw::c_void,
    // The deferred lighting pass's resources (M6): the G-buffer's
    // geometry pipeline (three colour attachments) and the lighting
    // pass's fullscreen triangle pipeline (the G-buffer's attachments
    // sample through the lighting set).
    gbuffer_pipeline: vk::VkPipeline,
    gbuffer_pipeline_ready: bool,
    gbuffer_pipeline_layout: vk::VkPipelineLayout,
    gbuffer_set_layout: vk::VkDescriptorSetLayout,
    // The G-buffer's render pass: three colour attachments plus depth
    // (the one pass the deferred path's geometry step opens; the
    // per-target framebuffers build against it).
    gbuffer_render_pass: vk::VkRenderPass,
    gbuffer_camera_buffer: Option<VulkanResource>,
    gbuffer_camera_map: *mut std::os::raw::c_void,
    gbuffer_descriptor_set: vk::VkDescriptorSet,
    gbuffer_descriptor_pool: vk::VkDescriptorPool,
    lighting_pipeline: vk::VkPipeline,
    lighting_pipeline_ready: bool,
    lighting_pipeline_layout: vk::VkPipelineLayout,
    lighting_set_layout: vk::VkDescriptorSetLayout,
    lighting_lights_buffer: Option<VulkanResource>,
    lighting_lights_map: *mut std::os::raw::c_void,
    lighting_descriptor_set: vk::VkDescriptorSet,
    lighting_descriptor_pool: vk::VkDescriptorPool,
    // The post-processing chain's resources (M7): the HDR geometry
    // pipeline (two colour attachments, the radiance and the world
    // position) and the composite pass's fullscreen triangle pipeline.
    post_gbuffer_render_pass: vk::VkRenderPass,
    post_pipeline: vk::VkPipeline,
    post_composite_pipeline: vk::VkPipeline,
    post_pipeline_ready: bool,
    post_gbuffer_pipeline_layout: vk::VkPipelineLayout,
    post_gbuffer_set_layout: vk::VkDescriptorSetLayout,
    post_composite_pipeline_layout: vk::VkPipelineLayout,
    post_composite_set_layout: vk::VkDescriptorSetLayout,
    post_camera_buffer: Option<VulkanResource>,
    post_camera_map: *mut std::os::raw::c_void,
    post_lights_buffer: Option<VulkanResource>,
    post_lights_map: *mut std::os::raw::c_void,
    post_volume_buffer: Option<VulkanResource>,
    post_volume_map: *mut std::os::raw::c_void,
    post_config_buffer: Option<VulkanResource>,
    post_config_map: *mut std::os::raw::c_void,
    post_gbuffer_descriptor_set: vk::VkDescriptorSet,
    post_gbuffer_descriptor_pool: vk::VkDescriptorPool,
    post_composite_descriptor_set: vk::VkDescriptorSet,
    post_composite_descriptor_pool: vk::VkDescriptorPool,
    // The diagnostic triangle resources.
    tri_mesh: Option<GpuMesh>,
    tri_shader: Option<ShaderProgram>,
    tri_pipeline: vk::VkPipeline,
    tri_pipeline_layout: vk::VkPipelineLayout,
    tri_pipeline_ready: bool,
    tri_mesh_uploaded: bool,
    tri_mesh_buffers: Option<VulkanResource>,
    tri_mesh_indices: Option<VulkanResource>,
    tri_uniform_buffer: Option<VulkanResource>,
    tri_uniform_map: *mut std::os::raw::c_void,
    tri_descriptor_set: vk::VkDescriptorSet,
    tri_descriptor_pool: vk::VkDescriptorPool,
    tri_descriptor_set_layout: vk::VkDescriptorSetLayout,
    // The clear colour and the clear state.
    clear_color: ClearColor,
    clear_depth: f32,
    clear_stencil: i32,
    depth_test_enabled: bool,
    camera: CameraParams,
    bound_target: Option<Arc<Mutex<RenderTargetInner>>>,
    pass_open: bool,
    frame_open: bool,
    shutdown: bool,
    violations: u64,
    targets: Vec<VulkanRenderTarget>,
    textures_2d: Vec<VulkanTexture2D>,
    textures_3d: Vec<VulkanTexture2D>,
    // The light volume's 3D texture (M8): the backend-owned image the
    // set_light_volume call (re)creates and fills, and the volume
    // uniform block's values the gbuffer descriptor writes carry.
    light_volume: Option<VulkanTexture2D>,
    volume_uniform: VolumeUniform,
    readback_queue: Vec<(u32, u32)>,
    last_readback: Vec<ReadbackSample>,
    has_center: bool,
    center: ClearColor,
    center_depth: f32,
    center_stencil: i32,
}

impl VulkanBackend {
    /// DEBUG: the created image's device memory requirements (the
    /// depth's memory proof — 3072 bytes for 4x4x24 Rgba16F, 128 for a
    /// dropped depth). The debug example's helper only.
    pub fn debug_texture_3d_memory_size(&self, view: &TextureView) -> u64 {
        let index = match self
            .textures_3d
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &view.inner))
        {
            Some(index) => index,
            None => return 0,
        };
        let mut requirements = vk::VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        unsafe {
            (self.loader.vk_get_image_memory_requirements.unwrap())(
                self.device,
                self.textures_3d[index].image,
                &mut requirements,
            );
        }
        requirements.size
    }

    /// The device's name: the NUL-terminated C string the properties
    /// report.
    fn device_name_string(&self) -> String {
        let end = self
            .device_name
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(self.device_name.len());
        self.device_name[..end]
            .iter()
            .map(|&c| c as u8 as char)
            .collect()
    }

    /// Creates a backend instance; the loader resolves the platform's
    /// `vkGetInstanceProcAddr` through the runtime's library. Throws
    /// `RenderError` when the loader cannot be found, the instance cannot
    /// be created or a required entry point cannot be resolved.
    pub fn new(config: VulkanBackendConfig) -> Result<Self, RenderError> {
        let clear_depth = config.clear_depth;
        let clear_stencil = config.clear_stencil;
        if config.width == 0 || config.height == 0 {
            return Err(RenderError::from(
                "VulkanBackend: width and height must be positive",
            ));
        }
        let library = unsafe { load_library() }?;
        let mut loader = VkLoader::new();
        let mut missing = String::new();
        let get_proc = unsafe {
            get_library_proc(library, "vkGetInstanceProcAddr")
        };
        if get_proc.is_null() {
            return Err(RenderError::from(
                "VulkanBackend: the platform's libvulkan.so.1 does not export \
                 vkGetInstanceProcAddr",
            ));
        }
        // The loader-global entry points resolve against a null instance
        // handle through vkGetInstanceProcAddr itself.
        let resolve: vk::ResolveFn = unsafe {
            std::mem::transmute::<
                *mut std::os::raw::c_void,
                vk::ResolveFn,
            >(get_proc)
        };
        if !loader.load(resolve, &mut missing) {
            return Err(RenderError::from(format!(
                "VulkanBackend: the loader could not resolve {}",
                missing
            )));
        }
        let mut backend = VulkanBackend {
            config,
            loader,
            library,
            instance: std::ptr::null_mut(),
            physical_device: std::ptr::null_mut(),
            api_version: 0,
            vendor_id: 0,
            device_name: [0; vk::MAX_EXTENSION_NAME_SIZE],
            device: std::ptr::null_mut(),
            queue: std::ptr::null_mut(),
            queue_family: 0,
            command_pool: std::ptr::null_mut(),
            command_buffer: std::ptr::null_mut(),
            readback_command: std::ptr::null_mut(),
            depth_image: std::ptr::null_mut(),
            depth_memory: std::ptr::null_mut(),
            depth_view: std::ptr::null_mut(),
            depth_format: vk::Format::D24UnormS8Uint,
            render_pass: std::ptr::null_mut(),
            framebuffer: std::ptr::null_mut(),
            color_image: std::ptr::null_mut(),
            color_memory: std::ptr::null_mut(),
            color_view: std::ptr::null_mut(),
            readback_buffer: VulkanResource {
                handle: std::ptr::null_mut(),
                memory: std::ptr::null_mut(),
                released: false,
            },
            readback_map: std::ptr::null_mut(),
            default_texture: None,
            meshes: Vec::new(),
            bound_texture: None,
            mesh_pipeline: std::ptr::null_mut(),
            mesh_pipeline_ready: false,
            mesh_pipeline_layout: std::ptr::null_mut(),
            mesh_set_layout: std::ptr::null_mut(),
            mesh_pool: std::ptr::null_mut(),
            mesh_sampler: std::ptr::null_mut(),
            lit_pipeline: std::ptr::null_mut(),
            lit_pipeline_ready: false,
            lit_pipeline_layout: std::ptr::null_mut(),
            lit_set_layout: std::ptr::null_mut(),
            lit_lights_buffer: None,
            lit_lights_map: std::ptr::null_mut(),
            lit_block: LightsUniform::none(),
            lit_descriptor_set: std::ptr::null_mut(),
            lit_camera_buffer: None,
            lit_camera_map: std::ptr::null_mut(),
            gbuffer_pipeline: std::ptr::null_mut(),
            gbuffer_pipeline_ready: false,
            gbuffer_pipeline_layout: std::ptr::null_mut(),
            gbuffer_set_layout: std::ptr::null_mut(),
            gbuffer_render_pass: std::ptr::null_mut(),
            gbuffer_camera_buffer: None,
            gbuffer_camera_map: std::ptr::null_mut(),
            gbuffer_descriptor_set: std::ptr::null_mut(),
            gbuffer_descriptor_pool: std::ptr::null_mut(),
            lighting_pipeline: std::ptr::null_mut(),
            lighting_pipeline_ready: false,
            post_gbuffer_render_pass: std::ptr::null_mut(),
            post_pipeline: std::ptr::null_mut(),
            post_composite_pipeline: std::ptr::null_mut(),
            post_pipeline_ready: false,
            post_gbuffer_pipeline_layout: std::ptr::null_mut(),
            post_gbuffer_set_layout: std::ptr::null_mut(),
            post_composite_pipeline_layout: std::ptr::null_mut(),
            post_composite_set_layout: std::ptr::null_mut(),
            post_camera_buffer: None,
            post_camera_map: std::ptr::null_mut(),
            post_lights_buffer: None,
            post_lights_map: std::ptr::null_mut(),
            post_volume_buffer: None,
            post_volume_map: std::ptr::null_mut(),
            post_config_buffer: None,
            post_config_map: std::ptr::null_mut(),
            post_gbuffer_descriptor_set: std::ptr::null_mut(),
            post_gbuffer_descriptor_pool: std::ptr::null_mut(),
            post_composite_descriptor_set: std::ptr::null_mut(),
            post_composite_descriptor_pool: std::ptr::null_mut(),
            lighting_pipeline_layout: std::ptr::null_mut(),
            lighting_set_layout: std::ptr::null_mut(),
            lighting_lights_buffer: None,
            lighting_lights_map: std::ptr::null_mut(),
            lighting_descriptor_set: std::ptr::null_mut(),
            lighting_descriptor_pool: std::ptr::null_mut(),
            tri_mesh: None,
            tri_shader: None,
            tri_pipeline: std::ptr::null_mut(),
            tri_pipeline_layout: std::ptr::null_mut(),
            tri_pipeline_ready: false,
            tri_mesh_uploaded: false,
            tri_mesh_buffers: None,
            tri_mesh_indices: None,
            tri_uniform_buffer: None,
            tri_uniform_map: std::ptr::null_mut(),
            tri_descriptor_set: std::ptr::null_mut(),
            tri_descriptor_pool: std::ptr::null_mut(),
            tri_descriptor_set_layout: std::ptr::null_mut(),
            clear_color: ClearColor {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            clear_depth,
            clear_stencil,
            depth_test_enabled: true,
            camera: CameraParams::default(),
            bound_target: None,
            pass_open: false,
            frame_open: false,
            shutdown: false,
            violations: 0,
            targets: Vec::new(),
            textures_2d: Vec::new(),
            textures_3d: Vec::new(),
            light_volume: None,
            volume_uniform: VolumeUniform {
                origin: [0.0; 4],
                dims: [0.0; 4],
                params: [0.0; 4],
            },
            readback_queue: Vec::new(),
            last_readback: Vec::new(),
            has_center: false,
            center: ClearColor {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            center_depth: 1.0,
            center_stencil: 0,
        };
        backend.initialise()?;
        Ok(backend)
    }
    /// The instance and device creation. The device pick is the first
    /// physical device with a graphics queue family; the device pick is a
    /// headless one (no surface required).
    fn initialise(&mut self) -> Result<(), RenderError> {
        if self.shutdown {
            return Err(RenderError::from(
                "VulkanBackend::initialise: the backend is shut down",
            ));
        }
        // The application info: the engine's identity and the API version
        // the engine requires (the platform's Vulkan 1.1 loader minimum).
        let app_name = b"OpenGraceII\0";
        let engine_name = b"OpenGraceII\0";
        let app_info = vk::VkApplicationInfo {
            s_type: vk::StructureType::ApplicationInfo,
            p_next: std::ptr::null(),
            p_application_name: app_name.as_ptr() as *const _,
            application_version: 1,
            p_engine_name: engine_name.as_ptr() as *const _,
            engine_version: 1,
            api_version: (1 << 22) | (0 << 12) | 0,
        };
        let create_info = vk::VkInstanceCreateInfo {
            s_type: vk::StructureType::InstanceCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            p_application_info: &app_info as *const _ as *const _,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
        };
        let mut instance = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_instance.unwrap())(
                &create_info,
                std::ptr::null(),
                &mut instance,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(format!(
                "VulkanBackend::initialise: vkCreateInstance failed \
                 (result {})",
                result as i32
            )));
        }
        self.instance = instance;
        let mut missing = String::new();
        if !self.loader.load_instance(self.instance, &mut missing) {
            return Err(RenderError::from(format!(
                "VulkanBackend: the loader could not resolve {}",
                missing
            )));
        }
        self.pick_physical_device()?;
        self.create_device()?;
        self.create_command_pool()?;
        self.create_frame_resources()?;
        Ok(())
    }

    /// The device pick: the first physical device with a graphics queue
    /// family. Headless, no surface required.
    fn pick_physical_device(&mut self) -> Result<(), RenderError> {
        let mut count = 0u32;
        let result = unsafe {
            (self.loader.vk_enumerate_physical_devices.unwrap())(
                self.instance,
                &mut count,
                std::ptr::null_mut(),
            )
        };
        if result != vk::Result::Success || count == 0 {
            return Err(RenderError::from(
                "VulkanBackend::initialise: no Vulkan physical device exists",
            ));
        }
        let mut devices = vec![std::ptr::null_mut(); count as usize];
        let result = unsafe {
            (self.loader.vk_enumerate_physical_devices.unwrap())(
                self.instance,
                &mut count,
                devices.as_mut_ptr(),
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend::initialise: vkEnumeratePhysicalDevices failed",
            ));
        }
        for device in devices {
            let mut count = 0u32;
            unsafe {
                (self.loader
                    .vk_get_physical_device_queue_family_properties
                    .unwrap())(device, &mut count, std::ptr::null_mut());
            }
            let mut families = vec![vk::VkQueueFamilyProperties {
                queue_flags: 0,
                queue_count: 0,
                timestamp_valid_bits: 0,
                min_image_transfer_granularity: vk::VkExtent2D {
                    width: 0,
                    height: 0,
                },
            }; count as usize];
            unsafe {
                (self.loader
                    .vk_get_physical_device_queue_family_properties
                    .unwrap())(device, &mut count, families.as_mut_ptr());
            }
            for (index, family) in families.iter().enumerate() {
                if family.queue_flags & (vk::QueueFlags::Graphics as u32)
                    != 0
                {
                    self.physical_device = device;
                    self.queue_family = index as u32;
                    // The device's identity and the API version the
                    // describe() reports (the same fields the C++ backend
                    // records).
                    let mut properties = vk::VkPhysicalDeviceProperties {
                        api_version: 0,
                        driver_version: 0,
                        vendor_id: 0,
                        device_id: 0,
                        device_type: 0,
                        device_name: [0; vk::MAX_EXTENSION_NAME_SIZE],
                        pipeline_cache_uuid: [0; 16],
                        limits: vk::VkPhysicalDeviceLimits {
                            max_image_dimension1_d: 0,
                            max_image_dimension2_d: 0,
                            max_image_dimension3_d: 0,
                            max_image_dimension_cube: 0,
                            max_framebuffer_width: 0,
                            max_framebuffer_height: 0,
                            max_uniform_buffer_range: 0,
                            max_color_attachments: 0,
                            max_push_constants_size: 0,
                            max_memory_allocation_count: 0,
                            max_descriptor_set_samplers: 0,
                            max_descriptor_set_uniform_buffers: 0,
                            min_uniform_buffer_offset_alignment: 0,
                            max_per_stage_descriptor_samplers: 0,
                            max_per_stage_descriptor_uniform_buffers: 0,
                        },
                    };
                    unsafe {
                        (self.loader
                            .vk_get_physical_device_properties
                            .unwrap())(device, &mut properties);
                    }
                    self.api_version = properties.api_version;
                    self.vendor_id = properties.vendor_id;
                    self.device_name = properties.device_name;
                    return Ok(());
                }
            }
        }
        Err(RenderError::from(
            "VulkanBackend::initialise: no physical device has a graphics \
             queue family",
        ))
    }

    /// The device creation: one queue family, no features, no extensions.
    fn create_device(&mut self) -> Result<(), RenderError> {
        let priority = 1.0f32;
        let queue_info = vk::VkDeviceQueueCreateInfo {
            s_type: vk::StructureType::DeviceQueueCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: self.queue_family,
            queue_count: 1,
            p_queue_priorities: &priority,
        };
        let device_info = vk::VkDeviceCreateInfo {
            s_type: vk::StructureType::DeviceCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
            p_enabled_features: std::ptr::null(),
        };
        let mut device = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_device.unwrap())(
                self.physical_device,
                &device_info,
                std::ptr::null(),
                &mut device,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(format!(
                "VulkanBackend::initialise: vkCreateDevice failed (result {})",
                result as i32
            )));
        }
        self.device = device;
        // Resolves every device-level entry point through the previously
        // loaded vkGetDeviceProcAddr on this device.
        let mut missing = String::new();
        if !self.loader.load_device(device, &mut missing) {
            return Err(RenderError::from(format!(
                "VulkanBackend::initialise: the loader could not resolve {}",
                missing
            )));
        }
        let mut queue = std::ptr::null_mut();
        unsafe {
            (self.loader.vk_get_device_queue.unwrap())(
                self.device,
                self.queue_family,
                0,
                &mut queue,
            );
        }
        self.queue = queue;
        Ok(())
    }

    /// The command pool and the primary command buffer.
    fn create_command_pool(&mut self) -> Result<(), RenderError> {
        let pool_info = vk::VkCommandPoolCreateInfo {
            s_type: vk::StructureType::CommandPoolCreateInfo,
            p_next: std::ptr::null(),
            flags: vk::CommandPoolCreate::ResetCommandBuffer as u32,
            queue_family_index: self.queue_family,
        };
        let mut pool = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_command_pool.unwrap())(
                self.device,
                &pool_info,
                std::ptr::null(),
                &mut pool,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend::initialise: vkCreateCommandPool failed",
            ));
        }
        self.command_pool = pool;
        let alloc_info = vk::VkCommandBufferAllocateInfo {
            s_type: vk::StructureType::CommandBufferAllocateInfo,
            p_next: std::ptr::null(),
            command_pool: self.command_pool,
            level: vk::CommandBufferLevel::Primary,
            command_buffer_count: 1,
        };
        let mut command_buffer = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_command_buffers.unwrap())(
                self.device,
                &alloc_info,
                &mut command_buffer,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend::initialise: vkAllocateCommandBuffers failed",
            ));
        }
        self.command_buffer = command_buffer;
        let mut readback_command = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_command_buffers.unwrap())(
                self.device,
                &alloc_info,
                &mut readback_command,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend::initialise: vkAllocateCommandBuffers failed",
            ));
        }
        self.readback_command = readback_command;
        Ok(())
    }

    /// The frame resources: the off-screen target's depth attachment, the
    /// render pass, the framebuffer and the readback staging buffer.
    fn create_frame_resources(&mut self) -> Result<(), RenderError> {
        self.depth_format = self.pick_depth_format();
        let (image, memory, view) = self.create_offscreen_image(
            self.config.width,
            self.config.height,
            vk::Format::R8g8b8a8Unorm,
            vk::ImageUsage::ColorAttachment | vk::ImageUsage::TransferSrc,
        )?;
        self.color_image = image;
        self.color_memory = memory;
        self.color_view = view;
        let (image, memory, view) = self.create_offscreen_image(
            self.config.width,
            self.config.height,
            self.depth_format,
            vk::ImageUsage::DepthStencilAttachment | vk::ImageUsage::TransferSrc,
        )?;
        self.depth_image = image;
        self.depth_memory = memory;
        self.depth_view = view;
        self.render_pass = self.create_render_pass_for()?;
        self.framebuffer = self.create_framebuffer_for()?;
        // The readback staging buffer (one RGBA8 pixel + the depth word +
        // the stencil byte: the six 32-bit words the readback path maps,
        // the stencil byte landing in the fifth word's low byte).
        let (buffer, memory) = self.create_host_buffer(
            6 * std::mem::size_of::<u32>(),
            vk::BufferUsage::TransferDst,
        )?;
        self.readback_buffer = VulkanResource {
            handle: buffer,
            memory,
            released: false,
        };
        self.readback_map = self.map_memory(self.readback_buffer.memory)?;
        // The default texture: one 1x1 opaque white texel the mesh path's
        // descriptor set samples when no texture is bound (the shader's
        // sampler needs a live view from the first draw on).
        let desc = Texture2DDesc {
            width: 1,
            height: 1,
            format: PixelFormat::Rgba8,
        };
        let view = self.create_texture_2d(&desc)?;
        let white = [255u8, 255, 255, 255];
        self.fill_texture_2d(&view, &white)?;
        let index = self
            .textures_2d
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &view.inner))
            .expect("the default texture's slot");
        self.default_texture = Some(VulkanTexture2D {
            inner: self.textures_2d[index].inner.clone(),
            image: self.textures_2d[index].image,
            memory: self.textures_2d[index].memory,
            view: self.textures_2d[index].view,
            released: false,
        });
        Ok(())
    }
    /// The image layout transition: the generic pipeline barrier the
    /// backend records for the transfer copies (the readback path and the
    /// staging uploads). The queue family stays the same (the backend owns
    /// one queue family).
    fn transition_image_layout(
        &mut self,
        command_buffer: vk::VkCommandBuffer,
        image: vk::VkImage,
        aspect_mask: u32,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        // The barrier's access masks and the pipeline stages: the table
        // the layout transition follows (the readback path and the
        // staging uploads use it).
        let (src_access, dst_access) = match (old_layout, new_layout) {
            (vk::ImageLayout::Undefined,
             vk::ImageLayout::ColorAttachmentOptimal) => (0, vk::Access::ColorAttachmentWrite as u32),
            (vk::ImageLayout::Undefined,
             vk::ImageLayout::DepthStencilAttachmentOptimal) => (0, vk::Access::DepthStencilAttachmentWrite as u32),
            (vk::ImageLayout::ColorAttachmentOptimal,
             vk::ImageLayout::TransferSrcOptimal) => (vk::Access::ColorAttachmentWrite as u32, vk::Access::TransferRead as u32),
            (vk::ImageLayout::DepthStencilAttachmentOptimal,
             vk::ImageLayout::TransferSrcOptimal) => (vk::Access::DepthStencilAttachmentWrite as u32, vk::Access::TransferRead as u32),
            (vk::ImageLayout::TransferSrcOptimal,
             vk::ImageLayout::ColorAttachmentOptimal) => (vk::Access::TransferRead as u32, vk::Access::ColorAttachmentWrite as u32),
            (vk::ImageLayout::TransferSrcOptimal,
             vk::ImageLayout::DepthStencilAttachmentOptimal) => (vk::Access::TransferRead as u32, vk::Access::DepthStencilAttachmentWrite as u32),
            (vk::ImageLayout::Undefined,
             vk::ImageLayout::TransferDstOptimal) => (0, vk::Access::TransferWrite as u32),
            (vk::ImageLayout::TransferDstOptimal,
             vk::ImageLayout::ShaderReadOnlyOptimal) => (vk::Access::TransferWrite as u32, vk::Access::ShaderRead as u32),
            // The deferred lighting pass's G-buffer transitions (M6): the
            // geometry pass's colour attachments hand off to the lighting
            // pass's shader reads, and hand back before the next frame's
            // geometry pass opens the pass again.
            (vk::ImageLayout::ColorAttachmentOptimal,
             vk::ImageLayout::ShaderReadOnlyOptimal) => (vk::Access::ColorAttachmentWrite as u32, vk::Access::ShaderRead as u32),
            (vk::ImageLayout::ShaderReadOnlyOptimal,
             vk::ImageLayout::ColorAttachmentOptimal) => (vk::Access::ShaderRead as u32, vk::Access::ColorAttachmentWrite as u32),
            // The 3D texture's readback handoff (M8): the shader-read
            // state the fill's last transition left, into the copy's
            // source state, and back to Undefined (the next fill's
            // Undefined -> TransferDst entry expects it).
            (vk::ImageLayout::ShaderReadOnlyOptimal,
             vk::ImageLayout::TransferSrcOptimal) => (vk::Access::ShaderRead as u32, vk::Access::TransferRead as u32),
            (vk::ImageLayout::TransferSrcOptimal,
             vk::ImageLayout::Undefined) => (vk::Access::TransferRead as u32, 0),
            _ => (0, 0),
        };
        let (src_stage, dst_stage) = match (old_layout, new_layout) {
            (vk::ImageLayout::Undefined,
             vk::ImageLayout::ColorAttachmentOptimal) => {
                (vk::PipelineStage::TopOfPipe,
                 vk::PipelineStage::ColorAttachmentOutput)
            }
            (vk::ImageLayout::ColorAttachmentOptimal,
             vk::ImageLayout::ShaderReadOnlyOptimal) => {
                (vk::PipelineStage::ColorAttachmentOutput,
                 vk::PipelineStage::FragmentShader)
            }
            (vk::ImageLayout::ShaderReadOnlyOptimal,
             vk::ImageLayout::ColorAttachmentOptimal) => {
                (vk::PipelineStage::FragmentShader,
                 vk::PipelineStage::ColorAttachmentOutput)
            }
            (vk::ImageLayout::ShaderReadOnlyOptimal,
             vk::ImageLayout::TransferSrcOptimal) => {
                (vk::PipelineStage::FragmentShader,
                 vk::PipelineStage::Transfer)
            }
            (vk::ImageLayout::TransferSrcOptimal,
             vk::ImageLayout::Undefined) => {
                (vk::PipelineStage::Transfer,
                 vk::PipelineStage::TopOfPipe)
            }
            (vk::ImageLayout::Undefined,
             vk::ImageLayout::DepthStencilAttachmentOptimal) => {
                (vk::PipelineStage::TopOfPipe,
                 vk::PipelineStage::EarlyFragmentTests)
            }
            (vk::ImageLayout::ColorAttachmentOptimal,
             vk::ImageLayout::TransferSrcOptimal) => {
                (vk::PipelineStage::ColorAttachmentOutput,
                 vk::PipelineStage::Transfer)
            }
            (vk::ImageLayout::DepthStencilAttachmentOptimal,
             vk::ImageLayout::TransferSrcOptimal) => {
                (vk::PipelineStage::EarlyFragmentTests,
                 vk::PipelineStage::Transfer)
            }
            (vk::ImageLayout::TransferSrcOptimal,
             vk::ImageLayout::ColorAttachmentOptimal) => {
                (vk::PipelineStage::Transfer,
                 vk::PipelineStage::ColorAttachmentOutput)
            }
            (vk::ImageLayout::TransferSrcOptimal,
             vk::ImageLayout::DepthStencilAttachmentOptimal) => {
                (vk::PipelineStage::Transfer,
                 vk::PipelineStage::EarlyFragmentTests)
            }
            (vk::ImageLayout::Undefined,
             vk::ImageLayout::TransferDstOptimal) => {
                (vk::PipelineStage::TopOfPipe, vk::PipelineStage::Transfer)
            }
            (vk::ImageLayout::TransferDstOptimal,
             vk::ImageLayout::ShaderReadOnlyOptimal) => {
                (vk::PipelineStage::Transfer,
                 vk::PipelineStage::FragmentShader)
            }
            _ => {
                (vk::PipelineStage::TopOfPipe,
                 vk::PipelineStage::BottomOfPipe)
            }
        };
        let barrier = vk::VkImageMemoryBarrier {
            s_type: vk::StructureType::ImageMemoryBarrier,
            p_next: std::ptr::null(),
            src_access_mask: src_access,
            dst_access_mask: dst_access,
            old_layout,
            new_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: vk::VkImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        };
        unsafe {
            (self.loader.vk_cmd_pipeline_barrier.unwrap())(
                command_buffer,
                src_stage,
                dst_stage,
                0,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                1,
                &barrier,
            );
        }
    }
    /// The frame's command buffer: the reset, the begin and the render
    /// pass's open with the clear values. The clear happens inside the
    /// render pass: the attachments load the clear colour (the same rule
    /// the OpenGL backend's clear follows).
    fn begin_frame(&mut self) {
        if self.command_buffer.is_null() || self.frame_open {
            return;
        }
        let begin_info = vk::VkCommandBufferBeginInfo {
            s_type: vk::StructureType::CommandBufferBeginInfo,
            p_next: std::ptr::null(),
            flags: 0,
            p_inheritance_info: std::ptr::null(),
        };
        let result = unsafe {
            (self.loader.vk_begin_command_buffer.unwrap())(
                self.command_buffer,
                &begin_info,
            )
        };
        if result != vk::Result::Success {
            return;
        }
        let mut clear_values = [
            vk::VkClearValue {
                color: [0.0, 0.0, 0.0, 0.0],
            },
            vk::VkClearValue {
                depth_stencil: vk::VkClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
        ];
        let colour = self.clear_color;
        clear_values[0].color = [
            colour.r as f32 / 255.0,
            colour.g as f32 / 255.0,
            colour.b as f32 / 255.0,
            colour.a as f32 / 255.0,
        ];
        clear_values[1].depth_stencil = vk::VkClearDepthStencilValue {
            depth: self.clear_depth,
            stencil: self.clear_stencil as u32,
        };
        let area = vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        };
        let pass_begin = vk::VkRenderPassBeginInfo {
            s_type: vk::StructureType::RenderPassBeginInfo,
            p_next: std::ptr::null(),
            render_pass: self.render_pass,
            framebuffer: self.framebuffer,
            render_area: area,
            clear_value_count: 2,
            p_clear_values: clear_values.as_ptr(),
        };
        unsafe {
            (self.loader.vk_cmd_begin_render_pass.unwrap())(
                self.command_buffer,
                &pass_begin,
                0,
            );
        }
        self.pass_open = true;
        self.frame_open = true;
    }
    /// The frame's end: the render pass closes, the command buffer ends
    /// and the queue submits it. The off-screen frame is not presented
    /// (the readback copies the colour attachment instead).
    fn submit_frame(&mut self) {
        if self.command_buffer.is_null() || !self.frame_open {
            return;
        }
        if self.pass_open {
            unsafe {
                (self.loader.vk_cmd_end_render_pass.unwrap())(
                    self.command_buffer,
                );
            }
            self.pass_open = false;
        }
        let result = unsafe {
            (self.loader.vk_end_command_buffer.unwrap())(
                self.command_buffer,
            )
        };
        self.frame_open = false;
        if result != vk::Result::Success {
            return;
        }
        let submit = vk::VkSubmitInfo {
            s_type: vk::StructureType::SubmitInfo,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };
        unsafe {
            (self.loader.vk_queue_submit.unwrap())(
                self.queue,
                1,
                &submit,
                std::ptr::null_mut(),
            );
            (self.loader.vk_queue_wait_idle.unwrap())(self.queue);
        }
    }

    /// Copies one pixel of the off-screen colour attachment and the
    /// depth/stencil attachment into the readback staging buffer (the
    /// same one-off copy the contract's readPixelInto follows). The
    /// colour word lands at offset 0 (the RGBA8 bytes) and the
    /// depth/stencil word at offset 16 bytes.
    fn readback_pixel(&mut self, x: u32, y: u32) {
        if self.readback_command.is_null() {
            return;
        }
        // The one-off command buffer: the reset, the begin and the copy
        // (the pool's ResetCommandBuffer flag lets the buffer reset).
        unsafe {
            (self.loader.vk_reset_command_buffer.unwrap())(
                self.readback_command,
                0,
            );
        }
        let begin_info = vk::VkCommandBufferBeginInfo {
            s_type: vk::StructureType::CommandBufferBeginInfo,
            p_next: std::ptr::null(),
            flags: vk::CommandBufferUsage::OneTimeSubmit as u32,
            p_inheritance_info: std::ptr::null(),
        };
        let result = unsafe {
            (self.loader.vk_begin_command_buffer.unwrap())(
                self.readback_command,
                &begin_info,
            )
        };
        if result != vk::Result::Success {
            return;
        }
        let colour = self.color_image;
        let depth = self.depth_image;
        let depth_format = self.depth_format;
        let buffer = self.readback_buffer.handle;
        // The depth/stencil copy follows the spec's one-aspect-per-region
        // rule (VUID 09103): the depth aspect copies into the fourth word
        // and the stencil aspect copies into the fifth word's first byte
        // (only a D24S8 attachment carries a stencil aspect). A strict
        // driver such as the Intel one rejects a Depth|Stencil copy the
        // software rasterizer tolerates.
        let depth_aspect = vk::ImageAspect::Depth as u32;
        let word = std::mem::size_of::<u32>() as u64;
        // The layout transition's barrier covers both aspects (a D24S8
        // attachment carries both; the other formats carry depth alone).
        let depth_transition_aspect = match depth_format {
            vk::Format::D24UnormS8Uint => {
                depth_aspect | (vk::ImageAspect::Stencil as u32)
            }
            _ => depth_aspect,
        };
        let regions = [
            vk::VkBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::VkImageSubresourceLayers {
                    aspect_mask: vk::ImageAspect::Color as u32,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::VkOffset3D {
                    x: x as i32,
                    y: y as i32,
                    z: 0,
                },
                image_extent: vk::VkExtent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            },
            vk::VkBufferImageCopy {
                buffer_offset: 4 * word,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::VkImageSubresourceLayers {
                    aspect_mask: depth_aspect,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::VkOffset3D {
                    x: x as i32,
                    y: y as i32,
                    z: 0,
                },
                image_extent: vk::VkExtent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            },
            vk::VkBufferImageCopy {
                buffer_offset: 5 * word,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::VkImageSubresourceLayers {
                    aspect_mask: vk::ImageAspect::Stencil as u32,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::VkOffset3D {
                    x: x as i32,
                    y: y as i32,
                    z: 0,
                },
                image_extent: vk::VkExtent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            },
        ];
        // The colour attachment: the copy runs in TransferSrcOptimal
        // (the same transition the readback follows). The depth/stencil
        // attachment copies as the D24S8 word the unpack reads.
        self.transition_image_layout(
            self.readback_command,
            colour,
            vk::ImageAspect::Color as u32,
            vk::ImageLayout::ColorAttachmentOptimal,
            vk::ImageLayout::TransferSrcOptimal,
        );
        unsafe {
            (self.loader.vk_cmd_copy_image_to_buffer.unwrap())(
                self.readback_command,
                colour,
                vk::ImageLayout::TransferSrcOptimal,
                buffer,
                1,
                &regions[0],
            );
        }
        self.transition_image_layout(
            self.readback_command,
            colour,
            vk::ImageAspect::Color as u32,
            vk::ImageLayout::TransferSrcOptimal,
            vk::ImageLayout::ColorAttachmentOptimal,
        );
        self.transition_image_layout(
            self.readback_command,
            depth,
            depth_transition_aspect,
            vk::ImageLayout::DepthStencilAttachmentOptimal,
            vk::ImageLayout::TransferSrcOptimal,
        );
        unsafe {
            (self.loader.vk_cmd_copy_image_to_buffer.unwrap())(
                self.readback_command,
                depth,
                vk::ImageLayout::TransferSrcOptimal,
                buffer,
                1,
                &regions[1],
            );
        }
        unsafe {
            (self.loader.vk_cmd_copy_image_to_buffer.unwrap())(
                self.readback_command,
                depth,
                vk::ImageLayout::TransferSrcOptimal,
                buffer,
                1,
                &regions[2],
            );
        }
        self.transition_image_layout(
            self.readback_command,
            depth,
            depth_transition_aspect,
            vk::ImageLayout::TransferSrcOptimal,
            vk::ImageLayout::DepthStencilAttachmentOptimal,
        );
        let result = unsafe {
            (self.loader.vk_end_command_buffer.unwrap())(
                self.readback_command,
            )
        };
        if result != vk::Result::Success {
            return;
        }
        let submit = vk::VkSubmitInfo {
            s_type: vk::StructureType::SubmitInfo,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &self.readback_command,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };
        unsafe {
            (self.loader.vk_queue_submit.unwrap())(
                self.queue,
                1,
                &submit,
                std::ptr::null_mut(),
            );
            (self.loader.vk_queue_wait_idle.unwrap())(self.queue);
        }
        // The depth word and the stencil byte: the spec's one-aspect
        // copies land the depth word at offset 16 (the D24S8 word's low
        // 24 bits) and the stencil byte at offset 20. A D32Sfloat or
        // D16Unorm depth attachment carries a different layout: the
        // depth/stencil readback only follows the D24S8 case the device
        // supports.
        let map = self.readback_map;
        if !map.is_null() {
            let words = unsafe {
                std::slice::from_raw_parts(map as *const u32, 6)
            };
            let colour_word = words[0];
            let depth_word = words[4];
            let stencil_word = words[5];
            let colour = ClearColor {
                r: (colour_word & 0xFFu32) as u8,
                g: ((colour_word >> 8) & 0xFFu32) as u8,
                b: ((colour_word >> 16) & 0xFFu32) as u8,
                a: ((colour_word >> 24) & 0xFFu32) as u8,
            };
            // The depth/stencil word's layout follows the depth
            // attachment's format: D24S8 packs the 24 depth bits in the
            // word's low end and the 8 stencil bits at the top, D16Unorm
            // packs the 16 depth bits in the word's low end, and D32Sfloat
            // carries the depth as a float.
            let (depth, stencil) = match depth_format {
                vk::Format::D24UnormS8Uint => (
                    (depth_word & 0xFFFFFFu32) as f32 / 16777215.0,
                    (stencil_word & 0xFFu32) as i32,
                ),
                vk::Format::D16Unorm => (
                    (depth_word & 0xFFFFu32) as f32 / 65535.0,
                    0,
                ),
                _ => (
                    f32::from_bits(depth_word),
                    0,
                ),
            };
            self.has_center = true;
            self.center = colour;
            self.center_depth = depth;
            self.center_stencil = stencil;
            self.last_readback.push(ReadbackSample {
                x,
                y,
                color: colour,
                depth,
                stencil,
            });
        }
    }

    /// The device's depth/stencil attachment format: D24UnormS8Uint when
    /// the device supports it, falling back to D32Sfloat and then D16Unorm
    /// through vkGetPhysicalDeviceFormatProperties (the depth attachment's
    /// format feature support is a device property, not an assumption).
    fn pick_depth_format(&self) -> vk::Format {
        let candidates = [
            vk::Format::D24UnormS8Uint,
            vk::Format::D32Sfloat,
            vk::Format::D16Unorm,
        ];
        for candidate in candidates {
            let mut properties = vk::VkFormatProperties {
                linear_tiling_features: 0,
                optimal_tiling_features: 0,
                buffer_features: 0,
            };
            unsafe {
                (self.loader
                    .vk_get_physical_device_format_properties
                    .unwrap())(
                    self.physical_device,
                    candidate,
                    &mut properties,
                );
            }
            if properties.optimal_tiling_features
                & (vk::FormatFeature::DepthStencilAttachment as u32)
                != 0
            {
                return candidate;
            }
        }
        // The device reports no depth attachment format: D24S8 stays the
        // contract's declared format and the device's failure surfaces at
        // image creation (the same explicit error the OpenGL backend gives).
        vk::Format::D24UnormS8Uint
    }

    /// Creates an off-screen 2D image (color or depth) with the memory
    /// binding and the view. The backend owns all of them.
    fn create_offscreen_image(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: u32,
    ) -> Result<(vk::VkImage, vk::VkDeviceMemory, vk::VkImageView),
                RenderError> {
        let image_info = vk::VkImageCreateInfo {
            s_type: vk::StructureType::ImageCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            image_type: vk::ImageType::Type2D,
            format: format as u32,
            width,
            height,
            depth: 1,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCount::One,
            tiling: vk::ImageTiling::Optimal,
            usage: usage as u32,
            sharing_mode: vk::SharingMode::Exclusive,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
            initial_layout: vk::ImageLayout::Undefined,
        };
        let mut image = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_image.unwrap())(
                self.device,
                &image_info,
                std::ptr::null(),
                &mut image,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreateImage failed",
            ));
        }
        let mut requirements = vk::VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        unsafe {
            (self.loader.vk_get_image_memory_requirements.unwrap())(
                self.device,
                image,
                &mut requirements,
            );
        }
        let memory_type_index = self.find_memory_type(
            requirements.memory_type_bits,
            vk::MemoryProperty::DeviceLocal as u32,
        )?;
        let alloc_info = vk::VkMemoryAllocateInfo {
            s_type: vk::StructureType::MemoryAllocateInfo,
            p_next: std::ptr::null(),
            allocation_size: requirements.size,
            memory_type_index,
        };
        let mut memory = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_memory.unwrap())(
                self.device,
                &alloc_info,
                std::ptr::null(),
                &mut memory,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkAllocateMemory failed",
            ));
        }
        let result = unsafe {
            (self.loader.vk_bind_image_memory.unwrap())(
                self.device,
                image,
                memory,
                0,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkBindImageMemory failed",
            ));
        }
        let mut aspect = vk::VkImageSubresourceRange {
            aspect_mask: vk::ImageAspect::Color as u32,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        if usage & vk::ImageUsage::DepthStencilAttachment as u32 != 0 {
            // A depth/stencil attachment's view carries both aspects
            // (the same rule the C++ backend's depth view follows).
            aspect.aspect_mask = (vk::ImageAspect::Depth as u32)
                | (vk::ImageAspect::Stencil as u32);
        }
        let view_info = vk::VkImageViewCreateInfo {
            s_type: vk::StructureType::ImageViewCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            image,
            view_type: vk::ImageType::Type2D,
            format: format as u32,
            components: vk::VkComponentMapping {
                r: vk::ComponentSwizzle::Identity,
                g: vk::ComponentSwizzle::Identity,
                b: vk::ComponentSwizzle::Identity,
                a: vk::ComponentSwizzle::Identity,
            },
            subresource_range: aspect,
        };
        let mut view = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_image_view.unwrap())(
                self.device,
                &view_info,
                std::ptr::null(),
                &mut view,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreateImageView failed",
            ));
        }
        Ok((image, memory, view))
    }

    /// The 3D texture's off-screen image: the same creation path the
    /// 2D image follows, with the Type3D image type and the 3D view.
    /// The depth layers the volume's cells fill; the usage pair
    /// (TransferDst | Sampled) the fill and the shader read need.
    fn create_offscreen_image_3d(
        &mut self,
        width: u32,
        height: u32,
        depth: u32,
        format: vk::Format,
        usage: u32,
    ) -> Result<(vk::VkImage, vk::VkDeviceMemory, vk::VkImageView),
                RenderError> {
        let image_info = vk::VkImageCreateInfo {
            s_type: vk::StructureType::ImageCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            image_type: vk::ImageType::Type3D,
            format: format as u32,
            width,
            height,
            depth,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCount::One,
            tiling: vk::ImageTiling::Optimal,
            usage: usage as u32,
            sharing_mode: vk::SharingMode::Exclusive,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
            initial_layout: vk::ImageLayout::Undefined,
        };
        let mut image = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_image.unwrap())(
                self.device,
                &image_info,
                std::ptr::null(),
                &mut image,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreateImage failed",
            ));
        }
        let mut requirements = vk::VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        unsafe {
            (self.loader.vk_get_image_memory_requirements.unwrap())(
                self.device,
                image,
                &mut requirements,
            );
        }
        let memory_type_index = self.find_memory_type(
            requirements.memory_type_bits,
            vk::MemoryProperty::DeviceLocal as u32,
        )?;
        let alloc_info = vk::VkMemoryAllocateInfo {
            s_type: vk::StructureType::MemoryAllocateInfo,
            p_next: std::ptr::null(),
            allocation_size: requirements.size,
            memory_type_index,
        };
        let mut memory = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_memory.unwrap())(
                self.device,
                &alloc_info,
                std::ptr::null(),
                &mut memory,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkAllocateMemory failed",
            ));
        }
        let result = unsafe {
            (self.loader.vk_bind_image_memory.unwrap())(
                self.device,
                image,
                memory,
                0,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkBindImageMemory failed",
            ));
        }
        let aspect = vk::VkImageSubresourceRange {
            aspect_mask: vk::ImageAspect::Color as u32,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let view_info = vk::VkImageViewCreateInfo {
            s_type: vk::StructureType::ImageViewCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            image,
            view_type: vk::ImageType::Type3D,
            format: format as u32,
            components: vk::VkComponentMapping {
                r: vk::ComponentSwizzle::Identity,
                g: vk::ComponentSwizzle::Identity,
                b: vk::ComponentSwizzle::Identity,
                a: vk::ComponentSwizzle::Identity,
            },
            subresource_range: aspect,
        };
        let mut view = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_image_view.unwrap())(
                self.device,
                &view_info,
                std::ptr::null(),
                &mut view,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreateImageView failed",
            ));
        }
        Ok((image, memory, view))
    }

    /// The memory type index for the given requirements and the requested
    /// properties. The device's memory types are the source of truth.
    fn find_memory_type(
        &self,
        type_bits: u32,
        properties: u32,
    ) -> Result<u32, RenderError> {
        let mut memory_properties = vk::VkPhysicalDeviceMemoryProperties {
            memory_type_count: 0,
            memory_types: [vk::VkMemoryType {
                property_flags: 0,
                heap_index: 0,
            }; 32],
            memory_heap_count: 0,
            memory_heaps: [vk::VkMemoryHeap {
                size: 0,
                flags: 0,
            }; 16],
        };
        unsafe {
            (self.loader
                .vk_get_physical_device_memory_properties
                .unwrap())(
                self.physical_device,
                &mut memory_properties,
            );
        }
        for i in 0..memory_properties.memory_type_count as usize {
            let memory_type = &memory_properties.memory_types[i];
            // The memory type must carry every requested property: a
            // subset (a HostVisible type without HostCoherent) makes the
            // host writes invisible to the device's copies on real
            // hardware (the lavapipe tolerates it, a strict driver does
            // not).
            if type_bits & (1 << i) != 0
                && memory_type.property_flags & properties == properties
            {
                return Ok(i as u32);
            }
        }
        Err(RenderError::from(
            "VulkanBackend: no memory type matches the requested properties",
        ))
    }

    /// Creates a host-visible buffer with the memory binding.
    fn create_host_buffer(
        &mut self,
        size: usize,
        usage: vk::BufferUsage,
    ) -> Result<(vk::VkBuffer, vk::VkDeviceMemory), RenderError> {
        let buffer_info = vk::VkBufferCreateInfo {
            s_type: vk::StructureType::BufferCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            size: size as u64,
            usage: usage as u32,
            sharing_mode: vk::SharingMode::Exclusive,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
        };
        let mut buffer = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_buffer.unwrap())(
                self.device,
                &buffer_info,
                std::ptr::null(),
                &mut buffer,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreateBuffer failed",
            ));
        }
        let mut requirements = vk::VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        unsafe {
            (self.loader.vk_get_buffer_memory_requirements.unwrap())(
                self.device,
                buffer,
                &mut requirements,
            );
        }
        let memory_type_index = self.find_memory_type(
            requirements.memory_type_bits,
            vk::MemoryProperty::HostVisible | vk::MemoryProperty::HostCoherent,
        )?;
        let alloc_info = vk::VkMemoryAllocateInfo {
            s_type: vk::StructureType::MemoryAllocateInfo,
            p_next: std::ptr::null(),
            allocation_size: requirements.size,
            memory_type_index,
        };
        let mut memory = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_memory.unwrap())(
                self.device,
                &alloc_info,
                std::ptr::null(),
                &mut memory,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkAllocateMemory failed",
            ));
        }
        let result = unsafe {
            (self.loader.vk_bind_buffer_memory.unwrap())(
                self.device,
                buffer,
                memory,
                0,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkBindBufferMemory failed",
            ));
        }
        Ok((buffer, memory))
    }

    /// Maps the memory range for the CPU. The backend never unmaps the
    /// permanently mapped host-visible memory.
    fn map_memory(
        &self,
        memory: vk::VkDeviceMemory,
    ) -> Result<*mut std::os::raw::c_void, RenderError> {
        let mut map = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_map_memory.unwrap())(
                self.device,
                memory,
                0,
                vk::WHOLE_SIZE,
                0,
                &mut map,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkMapMemory failed",
            ));
        }
        Ok(map)
    }

    /// The render pass for the off-screen target: one colour attachment
    /// (Rgba8) and one depth attachment.
    fn create_render_pass_for(&mut self) -> Result<vk::VkRenderPass,
                                                    RenderError> {
        let color = vk::VkAttachmentDescription {
            flags: 0,
            format: vk::Format::R8g8b8a8Unorm as u32,
            samples: vk::SampleCount::One,
            load_op: vk::AttachmentLoadOp::Clear,
            store_op: vk::AttachmentStoreOp::Store,
            stencil_load_op: vk::AttachmentLoadOp::DontCare,
            stencil_store_op: vk::AttachmentStoreOp::DontCare,
            initial_layout: vk::ImageLayout::Undefined,
            final_layout: vk::ImageLayout::ColorAttachmentOptimal,
        };
        let depth = vk::VkAttachmentDescription {
            flags: 0,
            format: self.depth_format as u32,
            samples: vk::SampleCount::One,
            load_op: vk::AttachmentLoadOp::Clear,
            store_op: vk::AttachmentStoreOp::Store,
            stencil_load_op: vk::AttachmentLoadOp::Clear,
            stencil_store_op: vk::AttachmentStoreOp::Store,
            initial_layout: vk::ImageLayout::Undefined,
            final_layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
        };
        let color_ref = vk::VkAttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::ColorAttachmentOptimal,
        };
        let depth_ref = vk::VkAttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
        };
        let subpass = vk::VkSubpassDescription {
            flags: 0,
            pipeline_bind_point: vk::PipelineBindPoint::Graphics,
            input_attachment_count: 0,
            p_input_attachments: std::ptr::null(),
            color_attachment_count: 1,
            p_color_attachments: &color_ref,
            p_resolve_attachments: std::ptr::null(),
            p_depth_stencil_attachment: &depth_ref,
            preserve_attachment_count: 0,
            p_preserve_attachments: std::ptr::null(),
        };
        let attachments = [color, depth];
        let pass_info = vk::VkRenderPassCreateInfo {
            s_type: vk::StructureType::RenderPassCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            attachment_count: 2,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 0,
            p_dependencies: std::ptr::null(),
        };
        let mut pass = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_render_pass.unwrap())(
                self.device,
                &pass_info,
                std::ptr::null(),
                &mut pass,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreateRenderPass failed",
            ));
        }
        Ok(pass)
    }

    /// The framebuffer for the off-screen target (the colour + depth
    /// views).
    fn create_framebuffer_for(&mut self) -> Result<vk::VkFramebuffer,
                                                    RenderError> {
        let attachments = [
            self.color_view,
            self.depth_view,
        ];
        let framebuffer_info = vk::VkFramebufferCreateInfo {
            s_type: vk::StructureType::FramebufferCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            render_pass: self.render_pass,
            attachment_count: 2,
            p_attachments: attachments.as_ptr(),
            width: self.config.width,
            height: self.config.height,
            layers: 1,
        };
        let mut framebuffer = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_framebuffer.unwrap())(
                self.device,
                &framebuffer_info,
                std::ptr::null(),
                &mut framebuffer,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreateFramebuffer failed",
            ));
        }
        Ok(framebuffer)
    }

    /// Maps the project's PixelFormat onto the Vulkan formats (inside the
    /// Vulkan module only, the same scoping rule the loader's own types
    /// follow).
    fn to_vk_format(&self, format: PixelFormat) -> vk::Format {
        match format {
            PixelFormat::Rgba8 => vk::Format::R8g8b8a8Unorm,
            PixelFormat::Rgba16F => vk::Format::R16g16b16a16Sfloat,
            PixelFormat::R8 => vk::Format::R8Unorm,
            PixelFormat::Depth24Stencil8 => vk::Format::D24UnormS8Uint,
        }
    }

    /// The diagnostic triangle draw: the same flat-unlit path the OpenGL
    /// backend's diagnostic draw records, on the default framebuffer only.
    fn draw_triangle(&mut self) {
        if self.shutdown {
            return;
        }
        self.ensure_triangle_pipeline();
        if !self.tri_pipeline_ready {
            return;
        }
        // The clear path's diagnostic draw happens on the default
        // framebuffer; the triangle never writes depth.
        let command_buffer = self.command_buffer;
        if command_buffer.is_null() {
            return;
        }
        // The uniform block: the MVP and the colour (std140, the layout
        // the shader library's block declares). The engine's Mat4 stores
        // the memory the graphics APIs' uniform blocks read as-is.
        let aspect = self.config.width as f32 / self.config.height as f32;
        let model = Mat4::IDENTITY;
        let view = Mat4::look_at(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let projection = Mat4::perspective(
            std::f32::consts::FRAC_PI_2,
            aspect,
            0.1,
            100.0,
        );
        let mvp = projection.multiply(view).multiply(model);
        let colour = self.config.triangle_color;
        let uniform = TriUniform {
            mvp: mvp.m,
            color: [
                colour.r as f32 / 255.0,
                colour.g as f32 / 255.0,
                colour.b as f32 / 255.0,
                colour.a as f32 / 255.0,
            ],
        };
        if !self.tri_uniform_map.is_null() {
            let map = self.tri_uniform_map;
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    map as *mut u8,
                    std::mem::size_of::<TriUniform>(),
                )
            };
            let uniform_bytes = unsafe {
                std::slice::from_raw_parts(
                    &uniform as *const TriUniform as *const u8,
                    std::mem::size_of::<TriUniform>(),
                )
            };
            bytes.copy_from_slice(uniform_bytes);
        }
        let viewports = [vk::VkViewport {
            x: 0.0,
            y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        }];
        unsafe {
            (self.loader.vk_cmd_bind_pipeline.unwrap())(
                command_buffer,
                vk::PipelineBindPoint::Graphics,
                self.tri_pipeline,
            );
            (self.loader.vk_cmd_set_viewport.unwrap())(
                command_buffer,
                0,
                1,
                viewports.as_ptr(),
            );
            (self.loader.vk_cmd_set_scissor.unwrap())(
                command_buffer,
                0,
                1,
                scissors.as_ptr(),
            );
        }
        let descriptor_sets = [self.tri_descriptor_set];
        let vertex_buffer = self
            .tri_mesh_buffers
            .as_ref()
            .map(|b| b.handle)
            .unwrap_or(std::ptr::null_mut());
        let offsets = [0u64];
        unsafe {
            (self.loader.vk_cmd_bind_descriptor_sets.unwrap())(
                command_buffer,
                vk::PipelineBindPoint::Graphics,
                self.tri_pipeline_layout,
                0,
                1,
                descriptor_sets.as_ptr(),
                0,
                std::ptr::null(),
            );
            (self.loader.vk_cmd_bind_vertex_buffers.unwrap())(
                command_buffer,
                0,
                1,
                &vertex_buffer,
                offsets.as_ptr(),
            );
            (self.loader.vk_cmd_draw.unwrap())(
                command_buffer,
                3,
                1,
                0,
                0,
            );
        }
   }

    /// The diagnostic triangle's uniform block (std140: the MVP row-major
    /// followed by the RGBA colour).
    /// Creates the diagnostic triangle's pipeline, mesh buffers and
    /// descriptor set on first use. The SPIR-V comes from the shader
    /// library's build output; the backend resolves it through the engine's
    /// own loader (lesson 3).
    fn ensure_triangle_pipeline(&mut self) {
        if self.tri_pipeline_ready {
            return;
        }
        if self.create_triangle_pipeline().is_err() {
            return;
        }
        self.tri_pipeline_ready = true;
    }

    /// The pipeline creation path: the two shader modules, the pipeline
    /// layout, the descriptor set layout, the mesh buffers and the uniform
    /// buffer. Any failure leaves the pipeline unready and the diagnostic
    /// draw is skipped (the triangle is a diagnostic, never a dependency).
    fn create_triangle_pipeline(&mut self) -> Result<(), RenderError> {
        // The shader modules come from the build's SPIR-V arrays.
        let vertex_module =
            self.create_shader_module(&shader_library::spirv::TRIANGLE_VERTEX_SHADER_SPIRV)?;
        let fragment_module =
            self.create_shader_module(&shader_library::spirv::TRIANGLE_FRAGMENT_SHADER_SPIRV)?;
        // The descriptor set layout: one uniform buffer at binding 0.
        let binding = vk::VkDescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UniformBuffer,
            descriptor_count: 1,
            stage_flags: vk::ShaderStage::Vertex as u32
                | vk::ShaderStage::Fragment as u32,
            p_immutable_samplers: std::ptr::null(),
        };
        let layout_info = vk::VkDescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DescriptorSetLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: 1,
            p_bindings: &binding,
        };
        let mut set_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_set_layout.unwrap())(
                self.device,
                &layout_info,
                std::ptr::null(),
                &mut set_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreateDescriptorSetLayout failed",
            ));
        }
        self.tri_descriptor_set_layout = set_layout;
        let set_layouts = [set_layout];
        let pipeline_layout_info = vk::VkPipelineLayoutCreateInfo {
            s_type: vk::StructureType::PipelineLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
        };
        let mut pipeline_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_pipeline_layout.unwrap())(
                self.device,
                &pipeline_layout_info,
                std::ptr::null(),
                &mut pipeline_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreatePipelineLayout failed",
            ));
        }
        self.tri_pipeline_layout = pipeline_layout;
        // The graphics pipeline: the two stages, one binding at stride 12
        // (vec3 position), the triangle list, the full-frame dynamic
        // viewport, no depth test and no culling.
        let vertex_entry = b"main\0";
        let stages = [
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Vertex,
                module: vertex_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Fragment,
                module: fragment_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
        ];
        let bindings = [vk::VkVertexInputBindingDescription {
            binding: 0,
            stride: 12,
            input_rate: vk::VertexInputRate::Vertex,
        }];
        let attributes = [vk::VkVertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32Sfloat as u32,
            offset: 0,
        }];
        let vertex_input = vk::VkPipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PipelineVertexInputStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: bindings.as_ptr(),
            vertex_attribute_description_count: 1,
            p_vertex_attribute_descriptions: attributes.as_ptr(),
        };
        let input_assembly = vk::VkPipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PipelineInputAssemblyStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            topology: vk::PrimitiveTopology::TriangleList,
            primitive_restart_enable: 0,
        };
        let viewport = vk::VkViewport {
            x: 0.0,
            y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        };
        let viewport_state = vk::VkPipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PipelineViewportStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
        };
        let rasterization = vk::VkPipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PipelineRasterizationStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_clamp_enable: 0,
            rasterizer_discard_enable: 0,
            polygon_mode: vk::PolygonMode::Fill,
            cull_mode: 0,
            front_face: vk::FrontFace::CounterClockwise,
            depth_bias_enable: 0,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        };
        let multisample = vk::VkPipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PipelineMultisampleStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            rasterization_samples: vk::SampleCount::One,
            sample_shading_enable: 0,
            min_sample_shading: 1.0,
            p_sample_mask: std::ptr::null(),
            alpha_to_coverage_enable: 0,
            alpha_to_one_enable: 0,
        };
        let depth_stencil = vk::VkPipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PipelineDepthStencilStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_test_enable: 0,
            depth_write_enable: 0,
            depth_compare_op: vk::CompareOp::Always,
            depth_bounds_test_enable: 0,
            stencil_test_enable: 0,
            front: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            back: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        };
        let blend_attachment = vk::VkPipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::Zero,
            dst_color_blend_factor: vk::BlendFactor::Zero,
            color_blend_op: vk::BlendOp::Add,
            src_alpha_blend_factor: vk::BlendFactor::Zero,
            dst_alpha_blend_factor: vk::BlendFactor::Zero,
            alpha_blend_op: vk::BlendOp::Add,
            color_write_mask: 0xF,
        };
        let color_blend = vk::VkPipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PipelineColorBlendStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            logic_op_enable: 0,
            logic_op: 0,
            attachment_count: 1,
            p_attachments: &blend_attachment,
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };
        let dynamic_states = [
            vk::DynamicState::Viewport,
            vk::DynamicState::Scissor,
        ];
        let dynamic_state = vk::VkPipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PipelineDynamicStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            dynamic_state_count: 2,
            p_dynamic_states: dynamic_states.as_ptr(),
        };
        let pipeline_info = vk::VkGraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GraphicsPipelineCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_tessellation_state: std::ptr::null(),
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization,
            p_multisample_state: &multisample,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass: self.render_pass,
            subpass: 0,
            base_pipeline_handle: std::ptr::null_mut(),
            base_pipeline_index: -1,
        };
        let mut pipeline = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_graphics_pipelines.unwrap())(
                self.device,
                std::ptr::null_mut(),
                1,
                &pipeline_info,
                std::ptr::null(),
                &mut pipeline,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreateGraphicsPipelines failed",
            ));
        }
        self.tri_pipeline = pipeline;
        self.create_triangle_resources()?;
        Ok(())
    }

    /// One SPIR-V shader module from the build's word array.
    fn create_shader_module(
        &self,
        code: &[u32],
    ) -> Result<vk::VkShaderModule, RenderError> {
        let module_info = vk::VkShaderModuleCreateInfo {
            s_type: vk::StructureType::ShaderModuleCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            code_size: code.len() * 4,
            p_code: code.as_ptr(),
        };
        let mut module = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_shader_module.unwrap())(
                self.device,
                &module_info,
                std::ptr::null(),
                &mut module,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreateShaderModule failed",
            ));
        }
        Ok(module)
    }

    /// The generic mesh pipeline's lazy creation: the two shader modules
    /// (the cube shaders from the library), the descriptor set layout (the
    /// camera block at binding 0, the combined image sampler at binding 1),
    /// the pipeline layout, the sampler and the graphics pipeline (the PNU
    /// vertex input, the depth test on, the triangle list).
    fn ensure_mesh_pipeline(&mut self) {
        if self.mesh_pipeline_ready {
            return;
        }
        if self.create_mesh_pipeline().is_err() {
            return;
        }
        self.mesh_pipeline_ready = true;
    }

    /// The lit path's ensure: the lit pipeline plus the lights' shared
    /// uniform buffer (the one host buffer every lit draw writes).
    fn ensure_lit_pipeline(&mut self) -> bool {
        if self.lit_pipeline_ready {
            return true;
        }
        if self.create_lit_pipeline().is_err() {
            return false;
        }
        // The lights' uniform buffer: host-visible, 256 bytes (the block
        // the D3D12 path's CBV alignment rule requires holds).
        let (buffer, memory) = match self
            .create_host_buffer(
                LightsUniform::SIZE,
                vk::BufferUsage::UniformBuffer,
            ) {
            Ok(pair) => pair,
            Err(_) => return false,
        };
        let map = match self.map_memory(memory) {
            Ok(map) => map,
            Err(_) => return false,
        };
        // The default block (no lights): the draws before the first lit
        // submission render unlit rather than reading garbage.
        let default = LightsUniform::none();
        unsafe {
            std::ptr::copy_nonoverlapping(
                &default as *const LightsUniform as *const u8,
                map as *mut u8,
                LightsUniform::SIZE,
            );
        }
        self.lit_lights_buffer = Some(VulkanResource {
            handle: buffer,
            memory,
            released: false,
        });
        self.lit_lights_map = map;
        // The lit path's camera buffer (a global one: the camera is the
        // same for every mesh the frame draws, the mesh path's per-mesh
        // buffers stay the unlit path's shape).
        let (camera_buffer, camera_memory) = match self.create_host_buffer(
            std::mem::size_of::<MeshUniform>(),
            vk::BufferUsage::UniformBuffer,
        ) {
            Ok(pair) => pair,
            Err(_) => return false,
        };
        let camera_map = match self.map_memory(camera_memory) {
            Ok(map) => map,
            Err(_) => return false,
        };
        self.lit_camera_buffer = Some(VulkanResource {
            handle: camera_buffer,
            memory: camera_memory,
            released: false,
        });
        self.lit_camera_map = camera_map;
        // The lit descriptor set: the camera block at binding 0, the
        // texture's combined sampler at binding 1, the lights' block at
        // binding 2.
        let pool_sizes = [
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 2,
            },
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
            },
        ];
        let pool_info = vk::VkDescriptorPoolCreateInfo {
            s_type: vk::StructureType::DescriptorPoolCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 2,
            p_pool_sizes: pool_sizes.as_ptr(),
        };
        let mut pool = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_pool.unwrap())(
                self.device,
                &pool_info,
                std::ptr::null(),
                &mut pool,
            )
        };
        if result != vk::Result::Success {
            return false;
        }
        let set_layouts = [self.lit_set_layout];
        let alloc_info = vk::VkDescriptorSetAllocateInfo {
            s_type: vk::StructureType::DescriptorSetAllocateInfo,
            p_next: std::ptr::null(),
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
        };
        let mut set = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_descriptor_sets.unwrap())(
                self.device,
                &alloc_info,
                &mut set,
            )
        };
        if result != vk::Result::Success {
            return false;
        }
        self.lit_descriptor_set = set;
        let default_view = self.default_texture.as_ref().map(|t| t.view);
        self.write_lit_descriptors(set, camera_buffer, default_view, buffer);
        true
    }

    /// The lit set's three writes: the camera block (binding 0), the
    /// texture's combined sampler (binding 1) and the lights' block
    /// (binding 2).
    fn write_lit_descriptors(
        &self,
        set: vk::VkDescriptorSet,
        camera_buffer: vk::VkBuffer,
        image_view: Option<vk::VkImageView>,
        lights_buffer: vk::VkBuffer,
    ) {
        let camera_info = vk::VkDescriptorBufferInfo {
            buffer: camera_buffer,
            offset: 0,
            range: std::mem::size_of::<MeshUniform>() as u64,
        };
        let lights_info = vk::VkDescriptorBufferInfo {
            buffer: lights_buffer,
            offset: 0,
            range: LightsUniform::SIZE as u64,
        };
        let image_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view: image_view.unwrap_or(std::ptr::null_mut()),
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let writes = [
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                p_image_info: std::ptr::null(),
                p_buffer_info: &camera_info,
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                p_image_info: &image_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 2,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                p_image_info: std::ptr::null(),
                p_buffer_info: &lights_info,
                p_texel_buffer_view: std::ptr::null(),
            },
        ];
        unsafe {
            (self.loader.vk_update_descriptor_sets.unwrap())(
                self.device,
                3,
                writes.as_ptr(),
                0,
                std::ptr::null(),
            );
        }
    }

    fn create_mesh_pipeline(&mut self) -> Result<(), RenderError> {
        self.mesh_pipeline = self.create_mesh_pipeline_inner(
            &shader_library::spirv::CUBE_VERTEX_SHADER_SPIRV,
            &shader_library::spirv::CUBE_FRAGMENT_SHADER_SPIRV,
            false,
        )?;
        self.mesh_pipeline_ready = true;
        Ok(())
    }

    /// The lit mesh pipeline (M5): the same vertex shape, the LIT
    /// shaders, the lights' uniform block at binding 2. The lit and the
    /// unlit pipelines share the sampler.
    fn create_lit_pipeline(&mut self) -> Result<(), RenderError> {
        self.lit_pipeline = self.create_mesh_pipeline_inner(
            &shader_library::spirv::LIT_VERTEX_SHADER_SPIRV,
            &shader_library::spirv::LIT_FRAGMENT_SHADER_SPIRV,
            true,
        )?;
        self.lit_pipeline_ready = true;
        Ok(())
    }

    /// The mesh pipeline's shared builder: the PNU vertex input, the
    /// depth test (LESS, the write on), no culling, the given shader
    /// modules, and — when `lit` — the lights' uniform block at binding
    /// 2. Returns the pipeline; the caller stores it (the layout and the
    /// set layout land in the lit fields, the unlit path keeps its own).
    fn create_mesh_pipeline_inner(
        &mut self,
        vertex_spirv: &[u32],
        fragment_spirv: &[u32],
        lit: bool,
    ) -> Result<vk::VkPipeline, RenderError> {
        let vertex_module = self.create_shader_module(vertex_spirv)?;
        let fragment_module = self.create_shader_module(fragment_spirv)?;
        // The descriptor set layout: binding 0 is the std140 camera block
        // (the vertex stage reads it), binding 1 is the combined image
        // sampler (the fragment stage reads it).
        let mut bindings = vec![
            vk::VkDescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Vertex as u32,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::VkDescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
        ];
        if lit {
            bindings.push(vk::VkDescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            });
        }
        let layout_info = vk::VkDescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DescriptorSetLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut set_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_set_layout.unwrap())(
                self.device,
                &layout_info,
                std::ptr::null(),
                &mut set_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the mesh's descriptor set layout failed",
            ));
        }
        if lit {
            self.lit_set_layout = set_layout;
        } else {
            self.mesh_set_layout = set_layout;
        }
        let set_layouts = [set_layout];
        let pipeline_layout_info = vk::VkPipelineLayoutCreateInfo {
            s_type: vk::StructureType::PipelineLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
        };
        let mut pipeline_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_pipeline_layout.unwrap())(
                self.device,
                &pipeline_layout_info,
                std::ptr::null(),
                &mut pipeline_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the mesh's pipeline layout failed",
            ));
        }
        if lit {
            self.lit_pipeline_layout = pipeline_layout;
        } else {
            self.mesh_pipeline_layout = pipeline_layout;
        }
        // The sampler: linear filtering, clamp to edge (the mip-map
        // machinery stays out of the first generic path).
        let sampler_info = vk::VkSamplerCreateInfo {
            s_type: vk::StructureType::SamplerCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            mag_filter: vk::Filter::Nearest,
            min_filter: vk::Filter::Nearest,
            mip_map_mode: 0,
            address_mode_u: vk::SamplerAddressMode::ClampToEdge,
            address_mode_v: vk::SamplerAddressMode::ClampToEdge,
            address_mode_w: vk::SamplerAddressMode::ClampToEdge,
            mip_lod_bias: 0.0,
            anisotropy_enable: 0,
            max_anisotropy: 1.0,
            compare_enable: 0,
            compare_op: vk::CompareOp::Always,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: vk::BorderColor::FloatTransparentBlack,
            unnormalized_coordinates: 0,
        };
        let mut sampler = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_sampler.unwrap())(
                self.device,
                &sampler_info,
                std::ptr::null(),
                &mut sampler,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the mesh's sampler failed",
            ));
        }
        if !lit {
            self.mesh_sampler = sampler;
        }
        // The graphics pipeline: the PNU vertex (one binding at stride 32,
        // three attributes: position at 0, normal at 12, uv at 24), the
        // depth test on (LESS, the depth write on), no culling (the cube's
        // faces are CCW but the path's first step stays forgiving).
        let vertex_entry = b"main\0";
        let stages = [
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Vertex,
                module: vertex_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Fragment,
                module: fragment_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
        ];
        let bindings = [vk::VkVertexInputBindingDescription {
            binding: 0,
            stride: 32,
            input_rate: vk::VertexInputRate::Vertex,
        }];
        let attributes = [
            vk::VkVertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32Sfloat as u32,
                offset: 0,
            },
            vk::VkVertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32Sfloat as u32,
                offset: 12,
            },
            vk::VkVertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::Rg32Sfloat as u32,
                offset: 24,
            },
        ];
        let vertex_input = vk::VkPipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PipelineVertexInputStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: bindings.as_ptr(),
            vertex_attribute_description_count: 3,
            p_vertex_attribute_descriptions: attributes.as_ptr(),
        };
        let input_assembly = vk::VkPipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PipelineInputAssemblyStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            topology: vk::PrimitiveTopology::TriangleList,
            primitive_restart_enable: 0,
        };
        let viewport = vk::VkViewport {
            x: 0.0,
            y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        };
        let viewport_state = vk::VkPipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PipelineViewportStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
        };
        let rasterization = vk::VkPipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PipelineRasterizationStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_clamp_enable: 0,
            rasterizer_discard_enable: 0,
            polygon_mode: vk::PolygonMode::Fill,
            cull_mode: 0,
            front_face: vk::FrontFace::CounterClockwise,
            depth_bias_enable: 0,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        };
        let multisample = vk::VkPipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PipelineMultisampleStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            rasterization_samples: vk::SampleCount::One,
            sample_shading_enable: 0,
            min_sample_shading: 1.0,
            p_sample_mask: std::ptr::null(),
            alpha_to_coverage_enable: 0,
            alpha_to_one_enable: 0,
        };
        let depth_stencil = vk::VkPipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PipelineDepthStencilStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::Less,
            depth_bounds_test_enable: 0,
            stencil_test_enable: 0,
            front: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            back: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        };
        let blend_attachment = vk::VkPipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::Zero,
            dst_color_blend_factor: vk::BlendFactor::Zero,
            color_blend_op: vk::BlendOp::Add,
            src_alpha_blend_factor: vk::BlendFactor::Zero,
            dst_alpha_blend_factor: vk::BlendFactor::Zero,
            alpha_blend_op: vk::BlendOp::Add,
            color_write_mask: 0xF,
        };
        let color_blend = vk::VkPipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PipelineColorBlendStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            logic_op_enable: 0,
            logic_op: 0,
            attachment_count: 1,
            p_attachments: &blend_attachment,
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };
        let dynamic_states = [
            vk::DynamicState::Viewport,
            vk::DynamicState::Scissor,
        ];
        let dynamic_state = vk::VkPipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PipelineDynamicStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            dynamic_state_count: 2,
            p_dynamic_states: dynamic_states.as_ptr(),
        };
        let pipeline_info = vk::VkGraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GraphicsPipelineCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_tessellation_state: std::ptr::null(),
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization,
            p_multisample_state: &multisample,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass: self.render_pass,
            subpass: 0,
            base_pipeline_handle: std::ptr::null_mut(),
            base_pipeline_index: -1,
        };
        let mut pipeline = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_graphics_pipelines.unwrap())(
                self.device,
                std::ptr::null_mut(),
                1,
                &pipeline_info,
                std::ptr::null(),
                &mut pipeline,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the mesh's graphics pipeline failed",
            ));
        }
        Ok(pipeline)
    }

    /// The G-buffer's geometry pipeline (M6): the same PNU vertex input
    /// the mesh pipeline carries, the GBUFFER shaders, the three colour
    /// attachments (the albedo, the normal, the world position) and the
    /// depth test on. The render pass is the gbuffer_render_pass the
    /// first three-attachment target's creation registers.
    fn create_gbuffer_pipeline(&mut self) -> Result<(), RenderError> {
        if self.gbuffer_render_pass.is_null() {
            self.gbuffer_render_pass = self.create_gbuffer_render_pass()?;
        }
        let vertex_module = self.create_shader_module(
            &shader_library::spirv::GBUFFER_VERTEX_SHADER_SPIRV,
        )?;
        let fragment_module = self.create_shader_module(
            &shader_library::spirv::GBUFFER_FRAGMENT_SHADER_SPIRV,
        )?;
        // The descriptor set layout: the camera block at binding 0 (the
        // vertex stage reads it), the surface texture at binding 1 (the
        // fragment stage reads it). The same shape the mesh pipeline's
        // set carries.
        let bindings = [
            vk::VkDescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Vertex as u32,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::VkDescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
        ];
        let layout_info = vk::VkDescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DescriptorSetLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut set_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_set_layout.unwrap())(
                self.device,
                &layout_info,
                std::ptr::null(),
                &mut set_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the G-buffer's set layout failed",
            ));
        }
        self.gbuffer_set_layout = set_layout;
        let set_layouts = [set_layout];
        let pipeline_layout_info = vk::VkPipelineLayoutCreateInfo {
            s_type: vk::StructureType::PipelineLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
        };
        let mut pipeline_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_pipeline_layout.unwrap())(
                self.device,
                &pipeline_layout_info,
                std::ptr::null(),
                &mut pipeline_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the G-buffer's pipeline layout failed",
            ));
        }
        self.gbuffer_pipeline_layout = pipeline_layout;
        // The graphics pipeline: the PNU vertex input, the depth test on
        // (LESS, the write on), no culling, three colour attachments.
        let vertex_entry = b"main\0";
        let stages = [
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Vertex,
                module: vertex_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Fragment,
                module: fragment_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
        ];
        let bindings = [vk::VkVertexInputBindingDescription {
            binding: 0,
            stride: 32,
            input_rate: vk::VertexInputRate::Vertex,
        }];
        let attributes = [
            vk::VkVertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32Sfloat as u32,
                offset: 0,
            },
            vk::VkVertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32Sfloat as u32,
                offset: 12,
            },
            vk::VkVertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::Rg32Sfloat as u32,
                offset: 24,
            },
        ];
        let vertex_input = vk::VkPipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PipelineVertexInputStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: bindings.as_ptr(),
            vertex_attribute_description_count: 3,
            p_vertex_attribute_descriptions: attributes.as_ptr(),
        };
        let input_assembly = vk::VkPipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PipelineInputAssemblyStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            topology: vk::PrimitiveTopology::TriangleList,
            primitive_restart_enable: 0,
        };
        let viewport = vk::VkViewport {
            x: 0.0,
            y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        };
        let viewport_state = vk::VkPipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PipelineViewportStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
        };
        let rasterization = vk::VkPipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PipelineRasterizationStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_clamp_enable: 0,
            rasterizer_discard_enable: 0,
            polygon_mode: vk::PolygonMode::Fill,
            cull_mode: 0,
            front_face: vk::FrontFace::CounterClockwise,
            depth_bias_enable: 0,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        };
        let multisample = vk::VkPipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PipelineMultisampleStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            rasterization_samples: vk::SampleCount::One,
            sample_shading_enable: 0,
            min_sample_shading: 1.0,
            p_sample_mask: std::ptr::null(),
            alpha_to_coverage_enable: 0,
            alpha_to_one_enable: 0,
        };
        let depth_stencil = vk::VkPipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PipelineDepthStencilStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::Less,
            depth_bounds_test_enable: 0,
            stencil_test_enable: 0,
            front: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            back: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        };
        let blend_attachment = vk::VkPipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::Zero,
            dst_color_blend_factor: vk::BlendFactor::Zero,
            color_blend_op: vk::BlendOp::Add,
            src_alpha_blend_factor: vk::BlendFactor::Zero,
            dst_alpha_blend_factor: vk::BlendFactor::Zero,
            alpha_blend_op: vk::BlendOp::Add,
            color_write_mask: 0xF,
        };
        let blend_attachments = [blend_attachment; 3];
        let color_blend = vk::VkPipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PipelineColorBlendStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            logic_op_enable: 0,
            logic_op: 0,
            attachment_count: 3,
            p_attachments: blend_attachments.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };
        let dynamic_states = [
            vk::DynamicState::Viewport,
            vk::DynamicState::Scissor,
        ];
        let dynamic_state = vk::VkPipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PipelineDynamicStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            dynamic_state_count: 2,
            p_dynamic_states: dynamic_states.as_ptr(),
        };
        let pipeline_info = vk::VkGraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GraphicsPipelineCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_tessellation_state: std::ptr::null(),
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization,
            p_multisample_state: &multisample,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass: self.gbuffer_render_pass,
            subpass: 0,
            base_pipeline_handle: std::ptr::null_mut(),
            base_pipeline_index: -1,
        };
        let mut pipeline = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_graphics_pipelines.unwrap())(
                self.device,
                std::ptr::null_mut(),
                1,
                &pipeline_info,
                std::ptr::null(),
                &mut pipeline,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the G-buffer's graphics pipeline failed",
            ));
        }
        self.gbuffer_pipeline = pipeline;
        Ok(())
    }

    /// The lighting pass's fullscreen triangle pipeline (M6): no vertex
    /// input (the three vertices the vertex index generates), the
    /// LIGHTING shaders, the depth test off, one colour attachment. The
    /// four G-buffer samplers (the albedo, the normal, the world) and
    /// the lights' block bind through the lighting set.
    fn create_lighting_pipeline(&mut self) -> Result<(), RenderError> {
        let vertex_module = self.create_shader_module(
            &shader_library::spirv::LIGHTING_VERTEX_SHADER_SPIRV,
        )?;
        let fragment_module = self.create_shader_module(
            &shader_library::spirv::LIGHTING_FRAGMENT_SHADER_SPIRV,
        )?;
        // The descriptor set layout: the lights' block at binding 0, the
        // albedo at binding 1, the normal at binding 2, the world at
        // binding 3 (the same set the lighting shaders declare).
        let bindings = [
            vk::VkDescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::VkDescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::VkDescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::VkDescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
        ];
        let layout_info = vk::VkDescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DescriptorSetLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut set_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_set_layout.unwrap())(
                self.device,
                &layout_info,
                std::ptr::null(),
                &mut set_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the lighting set layout failed",
            ));
        }
        self.lighting_set_layout = set_layout;
        let set_layouts = [set_layout];
        let pipeline_layout_info = vk::VkPipelineLayoutCreateInfo {
            s_type: vk::StructureType::PipelineLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
        };
        let mut pipeline_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_pipeline_layout.unwrap())(
                self.device,
                &pipeline_layout_info,
                std::ptr::null(),
                &mut pipeline_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the lighting pipeline layout failed",
            ));
        }
        self.lighting_pipeline_layout = pipeline_layout;
        // The graphics pipeline: no vertex input, the depth test off,
        // one colour attachment.
        let vertex_entry = b"main\0";
        let stages = [
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Vertex,
                module: vertex_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Fragment,
                module: fragment_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
        ];
        let vertex_input = vk::VkPipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PipelineVertexInputStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: std::ptr::null(),
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: std::ptr::null(),
        };
        let input_assembly = vk::VkPipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PipelineInputAssemblyStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            topology: vk::PrimitiveTopology::TriangleList,
            primitive_restart_enable: 0,
        };
        let viewport = vk::VkViewport {
            x: 0.0,
            y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        };
        let viewport_state = vk::VkPipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PipelineViewportStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
        };
        let rasterization = vk::VkPipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PipelineRasterizationStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_clamp_enable: 0,
            rasterizer_discard_enable: 0,
            polygon_mode: vk::PolygonMode::Fill,
            cull_mode: 0,
            front_face: vk::FrontFace::CounterClockwise,
            depth_bias_enable: 0,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        };
        let multisample = vk::VkPipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PipelineMultisampleStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            rasterization_samples: vk::SampleCount::One,
            sample_shading_enable: 0,
            min_sample_shading: 1.0,
            p_sample_mask: std::ptr::null(),
            alpha_to_coverage_enable: 0,
            alpha_to_one_enable: 0,
        };
        // The depth test off: the lighting pass's fullscreen triangle
        // covers the whole frame and the depth attachment stays out of
        // the pass's subpass.
        let depth_stencil = vk::VkPipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PipelineDepthStencilStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_test_enable: 0,
            depth_write_enable: 0,
            depth_compare_op: vk::CompareOp::Always,
            depth_bounds_test_enable: 0,
            stencil_test_enable: 0,
            front: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            back: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        };
        let blend_attachment = vk::VkPipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::Zero,
            dst_color_blend_factor: vk::BlendFactor::Zero,
            color_blend_op: vk::BlendOp::Add,
            src_alpha_blend_factor: vk::BlendFactor::Zero,
            dst_alpha_blend_factor: vk::BlendFactor::Zero,
            alpha_blend_op: vk::BlendOp::Add,
            color_write_mask: 0xF,
        };
        let color_blend = vk::VkPipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PipelineColorBlendStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            logic_op_enable: 0,
            logic_op: 0,
            attachment_count: 1,
            p_attachments: &blend_attachment,
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };
        let dynamic_states = [
            vk::DynamicState::Viewport,
            vk::DynamicState::Scissor,
        ];
        let dynamic_state = vk::VkPipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PipelineDynamicStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            dynamic_state_count: 2,
            p_dynamic_states: dynamic_states.as_ptr(),
        };
        let pipeline_info = vk::VkGraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GraphicsPipelineCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_tessellation_state: std::ptr::null(),
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization,
            p_multisample_state: &multisample,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass: self.render_pass,
            subpass: 0,
            base_pipeline_handle: std::ptr::null_mut(),
            base_pipeline_index: -1,
        };
        let mut pipeline = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_graphics_pipelines.unwrap())(
                self.device,
                std::ptr::null_mut(),
                1,
                &pipeline_info,
                std::ptr::null(),
                &mut pipeline,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the lighting pipeline failed",
            ));
        }
        self.lighting_pipeline = pipeline;
        Ok(())
    }

    /// The post chain's HDR geometry pipeline (M7): the PNU vertex
    /// input, the depth test on, two Rgba16F colour attachments, the
    /// POST_GBUFFER shaders (the radiance plus the world position).
    fn create_post_pipeline(&mut self) -> Result<(), RenderError> {
        if self.post_gbuffer_render_pass.is_null() {
            self.post_gbuffer_render_pass = self.create_post_render_pass()?;
        }
        let vertex_module = self.create_shader_module(
            &shader_library::spirv::GBUFFER_VERTEX_SHADER_SPIRV,
        )?;
        let fragment_module = self.create_shader_module(
            &shader_library::spirv::POST_GBUFFER_FRAGMENT_SHADER_SPIRV,
        )?;
        // The descriptor set layout: the camera block at binding 0 (the
        // vertex stage reads it — the same binding the GBUFFER_VERTEX
        // declares), the lights' block at binding 1 and the surface
        // texture at binding 2 (the fragment stage's reads — the same
        // bindings the POST_GBUFFER fragment declares).
        let bindings = [
            vk::VkDescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Vertex as u32,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::VkDescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::VkDescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::VkDescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::VkDescriptorSetLayoutBinding {
                binding: 4,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
        ];
        let layout_info = vk::VkDescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DescriptorSetLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut set_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_set_layout.unwrap())(
                self.device,
                &layout_info,
                std::ptr::null(),
                &mut set_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the post chain's set layout failed",
            ));
        }
        self.post_gbuffer_set_layout = set_layout;
        let set_layouts = [set_layout];
        let pipeline_layout_info = vk::VkPipelineLayoutCreateInfo {
            s_type: vk::StructureType::PipelineLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
        };
        let mut pipeline_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_pipeline_layout.unwrap())(
                self.device,
                &pipeline_layout_info,
                std::ptr::null(),
                &mut pipeline_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the post chain's pipeline layout failed",
            ));
        }
        self.post_gbuffer_pipeline_layout = pipeline_layout;
        let vertex_entry = b"main\0";
        let stages = [
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Vertex,
                module: vertex_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Fragment,
                module: fragment_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
        ];
        let bindings = [vk::VkVertexInputBindingDescription {
            binding: 0,
            stride: 32,
            input_rate: vk::VertexInputRate::Vertex,
        }];
        let attributes = [
            vk::VkVertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32Sfloat as u32,
                offset: 0,
            },
            vk::VkVertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32Sfloat as u32,
                offset: 12,
            },
            vk::VkVertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::Rg32Sfloat as u32,
                offset: 24,
            },
        ];
        let vertex_input = vk::VkPipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PipelineVertexInputStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: bindings.as_ptr(),
            vertex_attribute_description_count: 3,
            p_vertex_attribute_descriptions: attributes.as_ptr(),
        };
        let input_assembly = vk::VkPipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PipelineInputAssemblyStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            topology: vk::PrimitiveTopology::TriangleList,
            primitive_restart_enable: 0,
        };
        let viewport = vk::VkViewport {
            x: 0.0,
            y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        };
        let viewport_state = vk::VkPipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PipelineViewportStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
        };
        let rasterization = vk::VkPipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PipelineRasterizationStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_clamp_enable: 0,
            rasterizer_discard_enable: 0,
            polygon_mode: vk::PolygonMode::Fill,
            cull_mode: 0,
            front_face: vk::FrontFace::CounterClockwise,
            depth_bias_enable: 0,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        };
        let multisample = vk::VkPipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PipelineMultisampleStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            rasterization_samples: vk::SampleCount::One,
            sample_shading_enable: 0,
            min_sample_shading: 1.0,
            p_sample_mask: std::ptr::null(),
            alpha_to_coverage_enable: 0,
            alpha_to_one_enable: 0,
        };
        let depth_stencil = vk::VkPipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PipelineDepthStencilStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::Less,
            depth_bounds_test_enable: 0,
            stencil_test_enable: 0,
            front: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            back: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        };
        let blend_attachment = vk::VkPipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::Zero,
            dst_color_blend_factor: vk::BlendFactor::Zero,
            color_blend_op: vk::BlendOp::Add,
            src_alpha_blend_factor: vk::BlendFactor::Zero,
            dst_alpha_blend_factor: vk::BlendFactor::Zero,
            alpha_blend_op: vk::BlendOp::Add,
            color_write_mask: 0xF,
        };
        let blend_attachments = [blend_attachment; 2];
        let color_blend = vk::VkPipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PipelineColorBlendStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            logic_op_enable: 0,
            logic_op: 0,
            attachment_count: 2,
            p_attachments: blend_attachments.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };
        let dynamic_states = [
            vk::DynamicState::Viewport,
            vk::DynamicState::Scissor,
        ];
        let dynamic_state = vk::VkPipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PipelineDynamicStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            dynamic_state_count: 2,
            p_dynamic_states: dynamic_states.as_ptr(),
        };
        let pipeline_info = vk::VkGraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GraphicsPipelineCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_tessellation_state: std::ptr::null(),
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization,
            p_multisample_state: &multisample,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass: self.post_gbuffer_render_pass,
            subpass: 0,
            base_pipeline_handle: std::ptr::null_mut(),
            base_pipeline_index: -1,
        };
        let mut pipeline = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_graphics_pipelines.unwrap())(
                self.device,
                std::ptr::null_mut(),
                1,
                &pipeline_info,
                std::ptr::null(),
                &mut pipeline,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the post chain's graphics pipeline failed",
            ));
        }
        self.post_pipeline = pipeline;
        Ok(())
    }

    /// The post chain's composite pipeline (M7): the fullscreen
    /// triangle, the depth test off, one colour attachment (the default
    /// framebuffer's), the POST_COMPOSITE shaders.
    fn create_post_composite_pipeline(&mut self) -> Result<(), RenderError> {
        let vertex_module = self.create_shader_module(
            &shader_library::spirv::LIGHTING_VERTEX_SHADER_SPIRV,
        )?;
        let fragment_module = self.create_shader_module(
            &shader_library::spirv::POST_COMPOSITE_FRAGMENT_SHADER_SPIRV,
        )?;
        // The descriptor set layout: the config block at binding 0, the
        // radiance at binding 1, the world at binding 2.
        let bindings = [
            vk::VkDescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::VkDescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::VkDescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
                stage_flags: vk::ShaderStage::Fragment as u32,
                p_immutable_samplers: std::ptr::null(),
            },
        ];
        let layout_info = vk::VkDescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DescriptorSetLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut set_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_set_layout.unwrap())(
                self.device,
                &layout_info,
                std::ptr::null(),
                &mut set_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the composite set layout failed",
            ));
        }
        self.post_composite_set_layout = set_layout;
        let set_layouts = [set_layout];
        let pipeline_layout_info = vk::VkPipelineLayoutCreateInfo {
            s_type: vk::StructureType::PipelineLayoutCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
        };
        let mut pipeline_layout = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_pipeline_layout.unwrap())(
                self.device,
                &pipeline_layout_info,
                std::ptr::null(),
                &mut pipeline_layout,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the composite pipeline layout failed",
            ));
        }
        self.post_composite_pipeline_layout = pipeline_layout;
        let vertex_entry = b"main\0";
        let stages = [
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Vertex,
                module: vertex_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
            vk::VkPipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: std::ptr::null(),
                flags: 0,
                stage: vk::ShaderStage::Fragment,
                module: fragment_module,
                p_name: vertex_entry.as_ptr() as *const _,
                p_specialization_info: std::ptr::null(),
            },
        ];
        let vertex_input = vk::VkPipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PipelineVertexInputStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: std::ptr::null(),
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: std::ptr::null(),
        };
        let input_assembly = vk::VkPipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PipelineInputAssemblyStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            topology: vk::PrimitiveTopology::TriangleList,
            primitive_restart_enable: 0,
        };
        let viewport = vk::VkViewport {
            x: 0.0,
            y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        };
        let viewport_state = vk::VkPipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PipelineViewportStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
        };
        let rasterization = vk::VkPipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PipelineRasterizationStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_clamp_enable: 0,
            rasterizer_discard_enable: 0,
            polygon_mode: vk::PolygonMode::Fill,
            cull_mode: 0,
            front_face: vk::FrontFace::CounterClockwise,
            depth_bias_enable: 0,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        };
        let multisample = vk::VkPipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PipelineMultisampleStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            rasterization_samples: vk::SampleCount::One,
            sample_shading_enable: 0,
            min_sample_shading: 1.0,
            p_sample_mask: std::ptr::null(),
            alpha_to_coverage_enable: 0,
            alpha_to_one_enable: 0,
        };
        let depth_stencil = vk::VkPipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PipelineDepthStencilStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            depth_test_enable: 0,
            depth_write_enable: 0,
            depth_compare_op: vk::CompareOp::Always,
            depth_bounds_test_enable: 0,
            stencil_test_enable: 0,
            front: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            back: vk::VkStencilOpState {
                fail_op: vk::StencilOp::Keep,
                pass_op: vk::StencilOp::Keep,
                depth_fail_op: vk::StencilOp::Keep,
                compare_op: vk::CompareOp::Always,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            },
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        };
        let blend_attachment = vk::VkPipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::Zero,
            dst_color_blend_factor: vk::BlendFactor::Zero,
            color_blend_op: vk::BlendOp::Add,
            src_alpha_blend_factor: vk::BlendFactor::Zero,
            dst_alpha_blend_factor: vk::BlendFactor::Zero,
            alpha_blend_op: vk::BlendOp::Add,
            color_write_mask: 0xF,
        };
        let color_blend = vk::VkPipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PipelineColorBlendStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            logic_op_enable: 0,
            logic_op: 0,
            attachment_count: 1,
            p_attachments: &blend_attachment,
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };
        let dynamic_states = [
            vk::DynamicState::Viewport,
            vk::DynamicState::Scissor,
        ];
        let dynamic_state = vk::VkPipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PipelineDynamicStateCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            dynamic_state_count: 2,
            p_dynamic_states: dynamic_states.as_ptr(),
        };
        let pipeline_info = vk::VkGraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GraphicsPipelineCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_tessellation_state: std::ptr::null(),
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization,
            p_multisample_state: &multisample,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass: self.render_pass,
            subpass: 0,
            base_pipeline_handle: std::ptr::null_mut(),
            base_pipeline_index: -1,
        };
        let mut pipeline = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_graphics_pipelines.unwrap())(
                self.device,
                std::ptr::null_mut(),
                1,
                &pipeline_info,
                std::ptr::null(),
                &mut pipeline,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the composite pipeline failed",
            ));
        }
        self.post_composite_pipeline = pipeline;
        Ok(())
    }

    /// The deferred lighting path's ensure: the two pipelines (the
    /// G-buffer's geometry pipeline and the lighting pass's fullscreen
    /// triangle), the camera buffer and the lights' buffer, and the two
    /// descriptor sets. The same lazy rule the lit path's ensure follows
    /// (the creation happens before the frame's first deferred draw; the
    /// creation inside an open frame's recording is the software
    /// rasterizer's crash the M5 run taught).
    fn ensure_deferred_pipeline(&mut self) -> bool {
        if self.gbuffer_pipeline_ready && self.lighting_pipeline_ready {
            return true;
        }
        if self.create_gbuffer_pipeline().is_err() {
            return false;
        }
        if self.create_lighting_pipeline().is_err() {
            return false;
        }
        // The G-buffer's camera buffer (the same MeshUniform block the
        // mesh path's per-mesh uniform carries).
        let (camera_buffer, camera_memory) = match self.create_host_buffer(
            std::mem::size_of::<MeshUniform>(),
            vk::BufferUsage::UniformBuffer,
        ) {
            Ok(pair) => pair,
            Err(_) => return false,
        };
        let camera_map = match self.map_memory(camera_memory) {
            Ok(map) => map,
            Err(_) => return false,
        };
        self.gbuffer_camera_buffer = Some(VulkanResource {
            handle: camera_buffer,
            memory: camera_memory,
            released: false,
        });
        self.gbuffer_camera_map = camera_map;
        // The lighting pass's lights buffer (the same 256-byte block the
        // lit path's lights buffer carries; the default block carries no
        // lights so the draws before the first submission render
        // unlit rather than reading garbage).
        let (lights_buffer, lights_memory) = match self
            .create_host_buffer(LightsUniform::SIZE, vk::BufferUsage::UniformBuffer)
        {
            Ok(pair) => pair,
            Err(_) => return false,
        };
        let lights_map = match self.map_memory(lights_memory) {
            Ok(map) => map,
            Err(_) => return false,
        };
        let default = LightsUniform::none();
        unsafe {
            std::ptr::copy_nonoverlapping(
                &default as *const LightsUniform as *const u8,
                lights_map as *mut u8,
                LightsUniform::SIZE,
            );
        }
        self.lighting_lights_buffer = Some(VulkanResource {
            handle: lights_buffer,
            memory: lights_memory,
            released: false,
        });
        self.lighting_lights_map = lights_map;
        // The G-buffer's descriptor set: the camera block at binding 0,
        // the surface texture's combined sampler at binding 1.
        let pool_sizes = [
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
            },
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
            },
        ];
        let pool_info = vk::VkDescriptorPoolCreateInfo {
            s_type: vk::StructureType::DescriptorPoolCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 2,
            p_pool_sizes: pool_sizes.as_ptr(),
        };
        let mut pool = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_pool.unwrap())(
                self.device,
                &pool_info,
                std::ptr::null(),
                &mut pool,
            )
        };
        if result != vk::Result::Success {
            return false;
        }
        self.gbuffer_descriptor_pool = pool;
        let set_layouts = [self.gbuffer_set_layout];
        let alloc_info = vk::VkDescriptorSetAllocateInfo {
            s_type: vk::StructureType::DescriptorSetAllocateInfo,
            p_next: std::ptr::null(),
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
        };
        let mut set = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_descriptor_sets.unwrap())(
                self.device,
                &alloc_info,
                &mut set,
            )
        };
        if result != vk::Result::Success {
            return false;
        }
        self.gbuffer_descriptor_set = set;
        let default_view = self.default_texture.as_ref().map(|t| t.view);
        self.write_gbuffer_descriptors(set, camera_buffer, default_view);
        // The lighting pass's descriptor set: the lights' block at
        // binding 0, the three G-buffer attachments' combined samplers at
        // bindings 1-3 (the views land on the pass's draw — the target's
        // identity the submission carries).
        let pool_sizes = [
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
            },
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 3,
            },
        ];
        let pool_info = vk::VkDescriptorPoolCreateInfo {
            s_type: vk::StructureType::DescriptorPoolCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 2,
            p_pool_sizes: pool_sizes.as_ptr(),
        };
        let mut pool = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_pool.unwrap())(
                self.device,
                &pool_info,
                std::ptr::null(),
                &mut pool,
            )
        };
        if result != vk::Result::Success {
            return false;
        }
        self.lighting_descriptor_pool = pool;
        let set_layouts = [self.lighting_set_layout];
        let alloc_info = vk::VkDescriptorSetAllocateInfo {
            s_type: vk::StructureType::DescriptorSetAllocateInfo,
            p_next: std::ptr::null(),
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
        };
        let mut set = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_descriptor_sets.unwrap())(
                self.device,
                &alloc_info,
                &mut set,
            )
        };
        if result != vk::Result::Success {
            return false;
        }
        self.lighting_descriptor_set = set;
        true
    }

    /// The G-buffer's descriptor set writes: the camera block at binding 0
    /// and the surface texture's combined sampler at binding 1.
    fn write_gbuffer_descriptors(
        &self,
        set: vk::VkDescriptorSet,
        camera_buffer: vk::VkBuffer,
        image_view: Option<vk::VkImageView>,
    ) {
        let camera_info = vk::VkDescriptorBufferInfo {
            buffer: camera_buffer,
            offset: 0,
            range: std::mem::size_of::<MeshUniform>() as u64,
        };
        let image_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view: image_view.unwrap_or(std::ptr::null_mut()),
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let writes = [
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                p_image_info: std::ptr::null(),
                p_buffer_info: &camera_info,
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                p_image_info: &image_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            },
        ];
        unsafe {
            (self.loader.vk_update_descriptor_sets.unwrap())(
                self.device,
                2,
                writes.as_ptr(),
                0,
                std::ptr::null(),
            );
        }
    }

    /// The lighting pass's descriptor set writes: the lights' block at
    /// binding 0 and the three G-buffer attachments' combined samplers at
    /// bindings 1-3 (the albedo, the normal, the world position).
    fn write_lighting_descriptors(
        &self,
        set: vk::VkDescriptorSet,
        lights_buffer: vk::VkBuffer,
        albedo_view: vk::VkImageView,
        normal_view: vk::VkImageView,
        world_view: vk::VkImageView,
    ) {
        let lights_info = vk::VkDescriptorBufferInfo {
            buffer: lights_buffer,
            offset: 0,
            range: LightsUniform::SIZE as u64,
        };
        let albedo_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view: albedo_view,
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let normal_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view: normal_view,
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let world_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view: world_view,
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let writes = [
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                p_image_info: std::ptr::null(),
                p_buffer_info: &lights_info,
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                p_image_info: &albedo_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 2,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                p_image_info: &normal_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 3,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                p_image_info: &world_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            },
        ];
        unsafe {
            (self.loader.vk_update_descriptor_sets.unwrap())(
                self.device,
                4,
                writes.as_ptr(),
                0,
                std::ptr::null(),
            );
        }
    }

    /// The mesh's descriptor set writes: the camera block at binding 0 and
    /// the texture's combined image sampler at binding 1 (the default
    /// texture's view when no texture is bound).
    fn write_mesh_descriptors(
        &self,
        set: vk::VkDescriptorSet,
        uniform_buffer: vk::VkBuffer,
        uniform_size: usize,
        image_view: Option<vk::VkImageView>,
        sampler: vk::VkSampler,
    ) {
        let buffer_info = vk::VkDescriptorBufferInfo {
            buffer: uniform_buffer,
            offset: 0,
            range: uniform_size as u64,
        };
        // The default texture's view: the 1x1 white texture the setup
        // created (a None means the texture has no view, the descriptor
        // keeps the null view and the draw reports the violation instead
        // of crashing).
        let image_info = vk::VkDescriptorImageInfo {
            sampler,
            image_view: image_view.unwrap_or(std::ptr::null_mut()),
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let writes = [
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                p_image_info: std::ptr::null(),
                p_buffer_info: &buffer_info,
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                p_image_info: &image_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            },
        ];
        unsafe {
            (self.loader.vk_update_descriptor_sets.unwrap())(
                self.device,
                2,
                writes.as_ptr(),
                0,
                std::ptr::null(),
            );
        }
    }

    /// The mesh's descriptor's texture write (binding 1 only): the draw
    /// path rewrites the sampler's image view when the bind_texture state
    /// changed between the upload and the draw.
    fn write_mesh_texture_descriptor(
        &self,
        set: vk::VkDescriptorSet,
        image_view: vk::VkImageView,
    ) {
        let image_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view,
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let write = vk::VkWriteDescriptorSet {
            s_type: vk::StructureType::WriteDescriptorSet,
            p_next: std::ptr::null(),
            dst_set: set,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::CombinedImageSampler,
            p_image_info: &image_info,
            p_buffer_info: std::ptr::null(),
            p_texel_buffer_view: std::ptr::null(),
        };
        unsafe {
            (self.loader.vk_update_descriptor_sets.unwrap())(
                self.device,
                1,
                &write,
                0,
                std::ptr::null(),
            );
        }
    }

    /// The generic draw path: the mesh's buffers, the descriptor set (the
    /// camera block the set_camera state writes, the model matrix the
    /// submission carries) and the indexed draw into the open frame's
    /// command buffer. A draw with no open frame counts one violation (the
    /// same rule the diagnostic draw's clear-first contract follows).
    fn draw_mesh_by_id(&mut self, mesh_id: usize, model: Mat4) {
        self.draw_mesh_inner(mesh_id, model, false);
    }

    /// The lit draw (M5): the same path, the lit pipeline and the
    /// lights' uniform block bound (the submit path carries the lights).
    fn draw_mesh_lit_by_id(&mut self, mesh_id: usize, model: Mat4) {
        self.draw_mesh_inner(mesh_id, model, true);
    }

    /// The current lights block's copy (the read the draw's write makes).
    fn lit_uniform_block(&self) -> LightsUniform {
        self.lit_block
    }

    /// The mesh draw's shared body: the unlit path binds the mesh
    /// pipeline and the mesh descriptor set; the lit path binds the lit
    /// pipeline and a descriptor set that carries the lights' block at
    /// binding 2 (the per-mesh set's second buffer write, or the lit
    /// set). The first lit step reuses the mesh descriptor set: the lit
    /// layout's binding 2 write lands once per frame (the buffer write
    /// is idempotent at the same offsets).
    fn draw_mesh_inner(&mut self, mesh_id: usize, model: Mat4, lit: bool) {
        if self.command_buffer.is_null() || !self.pass_open {
            self.violations += 1;
            return;
        }
        let index = match self.meshes.get(mesh_id) {
            Some(Some(_state)) => mesh_id,
            _ => {
                self.violations += 1;
                return;
            }
        };
        if !self.mesh_pipeline_ready {
            return;
        }
        if lit && !self.lit_pipeline_ready {
            return;
        }
        // The uniform block: the view-projection matrix from the
        // set_camera state (the same matrices the camera path builds) and
        // the model matrix the submission carries.
        let aspect = self.config.width as f32 / self.config.height as f32;
        let camera = self.camera;
        let view = Mat4::look_at(camera.eye, camera.target, camera.up);
        let projection =
            Mat4::perspective(camera.fov_y_rad, aspect, camera.near, camera.far);
        let view_projection = projection.multiply(view);
        let uniform = MeshUniform {
            view_projection: view_projection.m,
            model: model.m,
        };
        // The lit path's lights block: the shared buffer's copy runs
        // before the draw (the uniform's write the frame's draws read;
        // the buffer stays host-visible, the write is immediate).
        if lit {
            let block = self.lit_uniform_block();
            if !self.lit_lights_map.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &block as *const LightsUniform as *const u8,
                        self.lit_lights_map as *mut u8,
                        LightsUniform::SIZE,
                    );
                }
            }
            if !self.lit_camera_map.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &uniform as *const MeshUniform as *const u8,
                        self.lit_camera_map as *mut u8,
                        std::mem::size_of::<MeshUniform>(),
                    );
                }
            }
        }
        {
            let state = self.meshes[index].as_ref().unwrap();
            if !state.uniform_map.is_null() {
                let map = state.uniform_map;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &uniform as *const MeshUniform as *const u8,
                        map as *mut u8,
                        std::mem::size_of::<MeshUniform>(),
                    );
                }
            }
            // The bound texture's view: the bind_texture state's index (the
            // default texture's view when the None carries). The write
            // runs each draw (the descriptor write is cheap; the
            // alternative is a per-draw set, which the generic path's
            // first step does not need).
            let bound_view = self
                .bound_texture
                .and_then(|i| self.textures_2d.get(i))
                .filter(|t| !t.released)
                .map(|t| t.view)
                .or_else(|| self.default_texture.as_ref().map(|t| t.view));
            if let Some(view) = bound_view {
                let set = if lit {
                    self.lit_descriptor_set
                } else {
                    state.descriptor_set
                };
                self.write_mesh_texture_descriptor(set, view);
            }
        }
        let (vertex_buffer, index_buffer, descriptor_set, index_count) = {
            let state = self.meshes[index].as_ref().unwrap();
            (
                state.vertex_buffer.handle,
                state.index_buffer.handle,
                if lit {
                    self.lit_descriptor_set
                } else {
                    state.descriptor_set
                },
                state.index_count,
            )
        };
        let viewports = [vk::VkViewport {
            x: 0.0,
            y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        }];
        let offsets = [0u64];
        let descriptor_sets = [descriptor_set];
        let (pipeline, layout) = if lit {
            (self.lit_pipeline, self.lit_pipeline_layout)
        } else {
            (self.mesh_pipeline, self.mesh_pipeline_layout)
        };
        unsafe {
            (self.loader.vk_cmd_bind_pipeline.unwrap())(
                self.command_buffer,
                vk::PipelineBindPoint::Graphics,
                pipeline,
            );
            (self.loader.vk_cmd_set_viewport.unwrap())(
                self.command_buffer,
                0,
                1,
                viewports.as_ptr(),
            );
            (self.loader.vk_cmd_set_scissor.unwrap())(
                self.command_buffer,
                0,
                1,
                scissors.as_ptr(),
            );
            (self.loader.vk_cmd_bind_descriptor_sets.unwrap())(
                self.command_buffer,
                vk::PipelineBindPoint::Graphics,
                layout,
                0,
                1,
                descriptor_sets.as_ptr(),
                0,
                std::ptr::null(),
            );
            (self.loader.vk_cmd_bind_vertex_buffers.unwrap())(
                self.command_buffer,
                0,
                1,
                &vertex_buffer,
                offsets.as_ptr(),
            );
            (self.loader.vk_cmd_bind_index_buffer.unwrap())(
                self.command_buffer,
                index_buffer,
                0,
                vk::IndexType::Uint32,
            );
            (self.loader.vk_cmd_draw_indexed.unwrap())(
                self.command_buffer,
                index_count,
                1,
                0,
                0,
                0,
            );
        }
    }

    /// The triangle's mesh buffers, uniform buffer and descriptor set.
    /// The vertex data is the three vertices of the diagnostic triangle in
    /// NDC; the uniform buffer carries the std140 block the shader library
    /// declares.
    fn create_triangle_resources(&mut self) -> Result<(), RenderError> {
        // The three vertices in NDC, stride 12 (vec3 position).
        let vertices: [f32; 9] = [
            -0.5, -0.5, 0.0, //
            0.5, -0.5, 0.0, //
            0.0, 0.5, 0.0,
        ];
        let vertex_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * 4,
            )
        };
        let (vertex_buffer, vertex_memory) = self.create_host_buffer(
            vertex_bytes.len(),
            vk::BufferUsage::VertexBuffer,
        )?;
        self.tri_mesh_buffers = Some(VulkanResource {
            handle: vertex_buffer,
            memory: vertex_memory,
            released: false,
        });
        let vertex_map = self.map_memory(vertex_memory)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                vertex_bytes.as_ptr(),
                vertex_map as *mut u8,
                vertex_bytes.len(),
            );
        }
        // The uniform buffer: one std140 block (64 bytes MVP + 16 bytes
        // colour).
        let uniform_size = std::mem::size_of::<TriUniform>();
        let (uniform_buffer, uniform_memory) = self.create_host_buffer(
            uniform_size,
            vk::BufferUsage::UniformBuffer,
        )?;
        self.tri_uniform_buffer = Some(VulkanResource {
            handle: uniform_buffer,
            memory: uniform_memory,
            released: false,
        });
        self.tri_uniform_map = self.map_memory(uniform_memory)?;
        // The descriptor pool: one uniform buffer descriptor.
        let pool_size = vk::VkDescriptorPoolSize {
            descriptor_type: vk::DescriptorType::UniformBuffer,
            descriptor_count: 1,
        };
        let pool_info = vk::VkDescriptorPoolCreateInfo {
            s_type: vk::StructureType::DescriptorPoolCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut pool = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_pool.unwrap())(
                self.device,
                &pool_info,
                std::ptr::null(),
                &mut pool,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkCreateDescriptorPool failed",
            ));
        }
        self.tri_descriptor_pool = pool;
        let set_layouts = [self.tri_descriptor_set_layout];
        let alloc_info = vk::VkDescriptorSetAllocateInfo {
            s_type: vk::StructureType::DescriptorSetAllocateInfo,
            p_next: std::ptr::null(),
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
        };
        let mut set = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_descriptor_sets.unwrap())(
                self.device,
                &alloc_info,
                &mut set,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: vkAllocateDescriptorSets failed",
            ));
        }
        self.tri_descriptor_set = set;
        let buffer_info = vk::VkDescriptorBufferInfo {
            buffer: uniform_buffer,
            offset: 0,
            range: uniform_size as u64,
        };
        let write = vk::VkWriteDescriptorSet {
            s_type: vk::StructureType::WriteDescriptorSet,
            p_next: std::ptr::null(),
            dst_set: set,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::UniformBuffer,
            p_image_info: std::ptr::null(),
            p_buffer_info: &buffer_info,
            p_texel_buffer_view: std::ptr::null(),
        };
        unsafe {
            (self.loader.vk_update_descriptor_sets.unwrap())(
                self.device,
                1,
                &write,
                0,
                std::ptr::null(),
            );
        }
        Ok(())
    }

    /// The post chain's ensure (M7): the two pipelines, the buffers and
    /// the two descriptor sets. The same lazy rule the deferred path's
    /// ensure follows (the creation before the frame's first post draw;
    /// creation inside an open frame's recording crashes the software
    /// rasterizer — the M5 lesson).
    fn ensure_post_pipeline(&mut self) -> bool {
        if self.post_pipeline_ready {
            return true;
        }
        if self.create_post_pipeline().is_err() {
            return false;
        }
        if self.create_post_composite_pipeline().is_err() {
            return false;
        }
        // The geometry pass's camera buffer (the same MeshUniform block
        // the deferred path's camera buffer carries).
        let (camera_buffer, camera_memory) = match self.create_host_buffer(
            std::mem::size_of::<MeshUniform>(),
            vk::BufferUsage::UniformBuffer,
        ) {
            Ok(pair) => pair,
            Err(_) => return false,
        };
        let camera_map = match self.map_memory(camera_memory) {
            Ok(map) => map,
            Err(_) => return false,
        };
        self.post_camera_buffer = Some(VulkanResource {
            handle: camera_buffer,
            memory: camera_memory,
            released: false,
        });
        self.post_camera_map = camera_map;
        // The geometry pass's lights buffer (the same 256-byte block).
        let (lights_buffer, lights_memory) = match self
            .create_host_buffer(LightsUniform::SIZE, vk::BufferUsage::UniformBuffer)
        {
            Ok(pair) => pair,
            Err(_) => return false,
        };
        let lights_map = match self.map_memory(lights_memory) {
            Ok(map) => map,
            Err(_) => return false,
        };
        let default = LightsUniform::none();
        unsafe {
            std::ptr::copy_nonoverlapping(
                &default as *const LightsUniform as *const u8,
                lights_map as *mut u8,
                LightsUniform::SIZE,
            );
        }
        self.post_lights_buffer = Some(VulkanResource {
            handle: lights_buffer,
            memory: lights_memory,
            released: false,
        });
        self.post_lights_map = lights_map;
        // The geometry pass's volume buffer (the 48-byte std140 block
        // the POST_GBUFFER fragment's Volume block reads).
        let (volume_buffer, volume_memory) = match self.create_host_buffer(
            std::mem::size_of::<VolumeUniform>(),
            vk::BufferUsage::UniformBuffer,
        ) {
            Ok(pair) => pair,
            Err(_) => return false,
        };
        let volume_map = match self.map_memory(volume_memory) {
            Ok(map) => map,
            Err(_) => return false,
        };
        self.post_volume_buffer = Some(VulkanResource {
            handle: volume_buffer,
            memory: volume_memory,
            released: false,
        });
        self.post_volume_map = volume_map;
        // The composite pass's config buffer (the three vec4 params the
        // POST_COMPOSITE shader's Config block carries — 48 bytes).
        let (config_buffer, config_memory) = match self.create_host_buffer(
            48,
            vk::BufferUsage::UniformBuffer,
        ) {
            Ok(pair) => pair,
            Err(_) => return false,
        };
        let config_map = match self.map_memory(config_memory) {
            Ok(map) => map,
            Err(_) => return false,
        };
        self.post_config_buffer = Some(VulkanResource {
            handle: config_buffer,
            memory: config_memory,
            released: false,
        });
        self.post_config_map = config_map;
        // The geometry pass's descriptor set: the camera at binding 0,
        // the lights at binding 1, the surface texture at binding 2.
        let pool_sizes = [
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 3,
            },
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 2,
            },
        ];
        let pool_info = vk::VkDescriptorPoolCreateInfo {
            s_type: vk::StructureType::DescriptorPoolCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 2,
            p_pool_sizes: pool_sizes.as_ptr(),
        };
        let mut pool = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_pool.unwrap())(
                self.device,
                &pool_info,
                std::ptr::null(),
                &mut pool,
            )
        };
        if result != vk::Result::Success {
            return false;
        }
        self.post_gbuffer_descriptor_pool = pool;
        let set_layouts = [self.post_gbuffer_set_layout];
        let alloc_info = vk::VkDescriptorSetAllocateInfo {
            s_type: vk::StructureType::DescriptorSetAllocateInfo,
            p_next: std::ptr::null(),
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
        };
        let mut set = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_descriptor_sets.unwrap())(
                self.device,
                &alloc_info,
                &mut set,
            )
        };
        if result != vk::Result::Success {
            return false;
        }
        self.post_gbuffer_descriptor_set = set;
        let default_view = self.default_texture.as_ref().map(|t| t.view);
        let default_volume_view = self
            .light_volume
            .as_ref()
            .map(|t| t.view);
        self.write_post_gbuffer_descriptors(
            set,
            camera_buffer,
            lights_buffer,
            self.post_volume_buffer
                .as_ref()
                .map(|r| r.handle)
                .unwrap_or(std::ptr::null_mut()),
            default_view,
            default_volume_view,
        );
        // The composite pass's descriptor set: the config at binding 0,
        // the radiance and the world attachments at bindings 1-2 (the
        // views land on the pass's draw).
        let pool_sizes = [
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
            },
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 2,
            },
        ];
        let pool_info = vk::VkDescriptorPoolCreateInfo {
            s_type: vk::StructureType::DescriptorPoolCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 2,
            p_pool_sizes: pool_sizes.as_ptr(),
        };
        let mut pool = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_pool.unwrap())(
                self.device,
                &pool_info,
                std::ptr::null(),
                &mut pool,
            )
        };
        if result != vk::Result::Success {
            return false;
        }
        self.post_composite_descriptor_pool = pool;
        let set_layouts = [self.post_composite_set_layout];
        let alloc_info = vk::VkDescriptorSetAllocateInfo {
            s_type: vk::StructureType::DescriptorSetAllocateInfo,
            p_next: std::ptr::null(),
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
        };
        let mut set = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_descriptor_sets.unwrap())(
                self.device,
                &alloc_info,
                &mut set,
            )
        };
        if result != vk::Result::Success {
            return false;
        }
        self.post_composite_descriptor_set = set;
        self.post_pipeline_ready = true;
        true
    }

    /// The post chain's geometry descriptor writes (M7): the camera
    /// block at binding 0 (the vertex stage reads it), the lights'
    /// block at binding 1 and the surface texture's combined sampler
    /// at binding 2 (the fragment stage's reads — the same bindings
    /// the POST_GBUFFER fragment declares).
    fn write_post_gbuffer_descriptors(
        &self,
        set: vk::VkDescriptorSet,
        camera_buffer: vk::VkBuffer,
        lights_buffer: vk::VkBuffer,
        volume_buffer: vk::VkBuffer,
        image_view: Option<vk::VkImageView>,
        volume_view: Option<vk::VkImageView>,
    ) {
        let camera_info = vk::VkDescriptorBufferInfo {
            buffer: camera_buffer,
            offset: 0,
            range: std::mem::size_of::<MeshUniform>() as u64,
        };
        let lights_info = vk::VkDescriptorBufferInfo {
            buffer: lights_buffer,
            offset: 0,
            range: LightsUniform::SIZE as u64,
        };
        let volume_info = vk::VkDescriptorBufferInfo {
            buffer: volume_buffer,
            offset: 0,
            range: std::mem::size_of::<VolumeUniform>() as u64,
        };
        let image_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view: image_view.unwrap_or(std::ptr::null_mut()),
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let volume_image_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view: volume_view.unwrap_or(std::ptr::null_mut()),
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let writes = [
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                p_image_info: std::ptr::null(),
                p_buffer_info: &camera_info,
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                p_image_info: std::ptr::null(),
                p_buffer_info: &lights_info,
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 2,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                p_image_info: &image_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 3,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                p_image_info: std::ptr::null(),
                p_buffer_info: &volume_info,
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 4,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                p_image_info: &volume_image_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            },
        ];
        unsafe {
            (self.loader.vk_update_descriptor_sets.unwrap())(
                self.device,
                5,
                writes.as_ptr(),
                0,
                std::ptr::null(),
            );
        }
    }

    /// The post chain's surface texture write (M7): the combined
    /// sampler at binding 2 (the POST_GBUFFER fragment's declaration).
    fn write_post_texture_descriptor(
        &self,
        set: vk::VkDescriptorSet,
        image_view: vk::VkImageView,
    ) {
        let image_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view,
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let write = vk::VkWriteDescriptorSet {
            s_type: vk::StructureType::WriteDescriptorSet,
            p_next: std::ptr::null(),
            dst_set: set,
            dst_binding: 2,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::CombinedImageSampler,
            p_image_info: &image_info,
            p_buffer_info: std::ptr::null(),
            p_texel_buffer_view: std::ptr::null(),
        };
        unsafe {
            (self.loader.vk_update_descriptor_sets.unwrap())(
                self.device,
                1,
                &write,
                0,
                std::ptr::null(),
            );
        }
    }
    /// The composite pass's descriptor writes: the config block at
    /// binding 0 and the HDR target's two attachments' combined
    /// samplers at bindings 1-2.
    fn write_post_composite_descriptors(
        &self,
        set: vk::VkDescriptorSet,
        config_buffer: vk::VkBuffer,
        radiance_view: vk::VkImageView,
        world_view: vk::VkImageView,
    ) {
        let config_info = vk::VkDescriptorBufferInfo {
            buffer: config_buffer,
            offset: 0,
            range: 48,
        };
        let radiance_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view: radiance_view,
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let world_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view: world_view,
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let writes = [
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                p_image_info: std::ptr::null(),
                p_buffer_info: &config_info,
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                p_image_info: &radiance_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::VkWriteDescriptorSet {
                s_type: vk::StructureType::WriteDescriptorSet,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: 2,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                p_image_info: &world_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            },
        ];
        unsafe {
            (self.loader.vk_update_descriptor_sets.unwrap())(
                self.device,
                3,
                writes.as_ptr(),
                0,
                std::ptr::null(),
            );
        }
    }
    /// The post chain's volume texture write (M8): the combined
    /// sampler at binding 4 (the POST_GBUFFER fragment's declaration).
    fn write_post_volume_descriptor(
        &self,
        set: vk::VkDescriptorSet,
        image_view: vk::VkImageView,
    ) {
        let image_info = vk::VkDescriptorImageInfo {
            sampler: self.mesh_sampler,
            image_view,
            image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
        };
        let write = vk::VkWriteDescriptorSet {
            s_type: vk::StructureType::WriteDescriptorSet,
            p_next: std::ptr::null(),
            dst_set: set,
            dst_binding: 4,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::CombinedImageSampler,
            p_image_info: &image_info,
            p_buffer_info: std::ptr::null(),
            p_texel_buffer_view: std::ptr::null(),
        };
        unsafe {
            (self.loader.vk_update_descriptor_sets.unwrap())(
                self.device,
                1,
                &write,
                0,
                std::ptr::null(),
            );
        }
    }
    /// The per-target render pass and framebuffer: the attachments the
    /// desc's colour formats and depth flag carry, the load Clear (the
    /// clear() the frame's first step runs), the store Store. The
    /// three-attachment MRT desc (the G-buffer) builds against the
    /// gbuffer_render_pass the pipeline creation registered; the single
    /// attachment desc builds its own pass (the same rule the
    /// pipeline's render_pass check follows).
    fn create_target_pass_and_framebuffer(
        &mut self,
        desc: &RenderTargetDesc,
        views: &[VulkanTexture2D],
        depth_view: &Option<VulkanTexture2D>,
    ) -> Result<(vk::VkRenderPass, vk::VkFramebuffer), RenderError> {
        // The G-buffer's MRT desc: the three colour attachments and the
        // depth attachment build against the shared gbuffer_render_pass
        // (the same pass the gbuffer pipeline creation registers; the
        // pipeline's render_pass check matches the framebuffer's pass).
        let is_gbuffer = desc.color_attachment_count == 3
            && desc.depth_attachment;
        // The post chain's HDR desc (M7): the two Rgba16F attachments
        // plus depth build against the shared post_gbuffer_render_pass
        // (the same rule the G-buffer's desc follows).
        let is_post_hdr = desc.color_attachment_count == 2
            && desc.depth_attachment
            && desc.color_formats[0] == PixelFormat::Rgba16F;
        let render_pass = if is_gbuffer {
            if self.gbuffer_render_pass.is_null() {
                self.gbuffer_render_pass = self.create_gbuffer_render_pass()?;
            }
            self.gbuffer_render_pass
        } else if is_post_hdr {
            if self.post_gbuffer_render_pass.is_null() {
                self.post_gbuffer_render_pass = self.create_post_render_pass()?;
            }
            self.post_gbuffer_render_pass
        } else {
            self.create_render_pass_for_format(desc)?
        };
        // The framebuffer: the target's views plus the depth view.
        let mut attachments: Vec<vk::VkImageView> = views
            .iter()
            .map(|v| v.view)
            .collect();
        if let Some(depth) = depth_view {
            attachments.push(depth.view);
        }
        let framebuffer_info = vk::VkFramebufferCreateInfo {
            s_type: vk::StructureType::FramebufferCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width: desc.width,
            height: desc.height,
            layers: 1,
        };
        let mut framebuffer = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_framebuffer.unwrap())(
                self.device,
                &framebuffer_info,
                std::ptr::null(),
                &mut framebuffer,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "create_render_target: vkCreateFramebuffer failed",
            ));
        }
        Ok((render_pass, framebuffer))
    }

    /// The G-buffer's render pass: three colour attachments (the albedo
    /// Rgba8, the normal Rgba8, the world Rgba16F) plus the depth
    /// attachment. The final layouts stay ColorAttachmentOptimal: the
    /// lighting pass's shader-read handoff runs through the pipeline
    /// barrier the submit_lighting_pass records (the same rule the
    /// readback's layout table follows).
    fn create_gbuffer_render_pass(
        &mut self,
    ) -> Result<vk::VkRenderPass, RenderError> {
        let color_formats = [
            vk::Format::R8g8b8a8Unorm,
            vk::Format::R8g8b8a8Unorm,
            vk::Format::R16g16b16a16Sfloat,
        ];
        let mut attachments: Vec<vk::VkAttachmentDescription> = Vec::new();
        for format in color_formats.iter() {
            attachments.push(vk::VkAttachmentDescription {
                flags: 0,
                format: *format as u32,
                samples: vk::SampleCount::One,
                load_op: vk::AttachmentLoadOp::Clear,
                store_op: vk::AttachmentStoreOp::Store,
                stencil_load_op: vk::AttachmentLoadOp::DontCare,
                stencil_store_op: vk::AttachmentStoreOp::DontCare,
                initial_layout: vk::ImageLayout::Undefined,
                final_layout: vk::ImageLayout::ColorAttachmentOptimal,
            });
        }
        attachments.push(vk::VkAttachmentDescription {
            flags: 0,
            format: self.depth_format as u32,
            samples: vk::SampleCount::One,
            load_op: vk::AttachmentLoadOp::Clear,
            store_op: vk::AttachmentStoreOp::Store,
            stencil_load_op: vk::AttachmentLoadOp::Clear,
            stencil_store_op: vk::AttachmentStoreOp::DontCare,
            initial_layout: vk::ImageLayout::Undefined,
            final_layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
        });
        let color_refs = [
            vk::VkAttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::ColorAttachmentOptimal,
            },
            vk::VkAttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::ColorAttachmentOptimal,
            },
            vk::VkAttachmentReference {
                attachment: 2,
                layout: vk::ImageLayout::ColorAttachmentOptimal,
            },
        ];
        let depth_ref = vk::VkAttachmentReference {
            attachment: 3,
            layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
        };
        let subpass = vk::VkSubpassDescription {
            flags: 0,
            pipeline_bind_point: vk::PipelineBindPoint::Graphics,
            input_attachment_count: 0,
            p_input_attachments: std::ptr::null(),
            color_attachment_count: 3,
            p_color_attachments: color_refs.as_ptr(),
            p_resolve_attachments: std::ptr::null(),
            p_depth_stencil_attachment: &depth_ref,
            preserve_attachment_count: 0,
            p_preserve_attachments: std::ptr::null(),
        };
        // The external dependency: the geometry pass's colour writes
        // finish before the lighting pass's shader reads (the same
        // handoff the submit_lighting_pass's barrier records; the
        // dependency keeps the pass boundary's implicit barrier honest).
        let dependency = vk::VkSubpassDependency {
            // VK_SUBPASS_EXTERNAL is the spec's all-ones value (the
            // loader's constants stay the ones the declarations carry;
            // the value stays the same).
            src_subpass: u32::MAX,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStage::ColorAttachmentOutput as u32,
            dst_stage_mask: vk::PipelineStage::ColorAttachmentOutput as u32
                | vk::PipelineStage::EarlyFragmentTests as u32,
            src_access_mask: vk::Access::ColorAttachmentWrite as u32,
            dst_access_mask: vk::Access::ColorAttachmentRead as u32
                | vk::Access::DepthStencilAttachmentRead as u32,
            dependency_flags: 0,
        };
        let pass_info = vk::VkRenderPassCreateInfo {
            s_type: vk::StructureType::RenderPassCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &dependency,
        };
        let mut pass = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_render_pass.unwrap())(
                self.device,
                &pass_info,
                std::ptr::null(),
                &mut pass,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "create_render_target: the G-buffer's vkCreateRenderPass \
                 failed",
            ));
        }
        Ok(pass)
    }

    /// The post chain's HDR render pass (M7): two colour attachments
    /// (the radiance and the world position, both Rgba16F) plus the
    /// depth attachment. The same shape the G-buffer's pass carries,
    /// with the two HDR formats.
    /// The post chain's HDR render pass: the two Rgba16F colour
    /// attachments (the radiance and the world position) plus the depth
    /// attachment — the same shape the G-buffer's pass carries, with
    /// the HDR formats the post target's images carry.
    fn create_post_render_pass(&mut self) -> Result<vk::VkRenderPass, RenderError> {
        let color_formats = [
            vk::Format::R16g16b16a16Sfloat,
            vk::Format::R16g16b16a16Sfloat,
        ];
        let mut attachments: Vec<vk::VkAttachmentDescription> = Vec::new();
        for format in color_formats.iter() {
            attachments.push(vk::VkAttachmentDescription {
                flags: 0,
                format: *format as u32,
                samples: vk::SampleCount::One,
                load_op: vk::AttachmentLoadOp::Clear,
                store_op: vk::AttachmentStoreOp::Store,
                stencil_load_op: vk::AttachmentLoadOp::DontCare,
                stencil_store_op: vk::AttachmentStoreOp::DontCare,
                initial_layout: vk::ImageLayout::Undefined,
                final_layout: vk::ImageLayout::ColorAttachmentOptimal,
            });
        }
        attachments.push(vk::VkAttachmentDescription {
            flags: 0,
            format: self.depth_format as u32,
            samples: vk::SampleCount::One,
            load_op: vk::AttachmentLoadOp::Clear,
            store_op: vk::AttachmentStoreOp::Store,
            stencil_load_op: vk::AttachmentLoadOp::Clear,
            stencil_store_op: vk::AttachmentStoreOp::DontCare,
            initial_layout: vk::ImageLayout::Undefined,
            final_layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
        });
        let color_refs = [
            vk::VkAttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::ColorAttachmentOptimal,
            },
            vk::VkAttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::ColorAttachmentOptimal,
            },
        ];
        let depth_ref = vk::VkAttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
        };
        let subpass = vk::VkSubpassDescription {
            flags: 0,
            pipeline_bind_point: vk::PipelineBindPoint::Graphics,
            input_attachment_count: 0,
            p_input_attachments: std::ptr::null(),
            color_attachment_count: 2,
            p_color_attachments: color_refs.as_ptr(),
            p_resolve_attachments: std::ptr::null(),
            p_depth_stencil_attachment: &depth_ref,
            preserve_attachment_count: 0,
            p_preserve_attachments: std::ptr::null(),
        };
        let dependency = vk::VkSubpassDependency {
            src_subpass: u32::MAX,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStage::ColorAttachmentOutput as u32,
            dst_stage_mask: vk::PipelineStage::ColorAttachmentOutput as u32
                | vk::PipelineStage::EarlyFragmentTests as u32,
            src_access_mask: vk::Access::ColorAttachmentWrite as u32,
            dst_access_mask: vk::Access::ColorAttachmentRead as u32
                | vk::Access::DepthStencilAttachmentRead as u32,
            dependency_flags: 0,
        };
        let pass_info = vk::VkRenderPassCreateInfo {
            s_type: vk::StructureType::RenderPassCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &dependency,
        };
        let mut pass = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_render_pass.unwrap())(
                self.device,
                &pass_info,
                std::ptr::null(),
                &mut pass,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "VulkanBackend: the post chain's vkCreateRenderPass failed",
            ));
        }
        Ok(pass)
    }

    /// The single-attachment desc's own render pass: the same shape the
    /// default framebuffer's pass carries (one colour attachment of the
    /// desc's format plus the depth attachment when the desc asks for
    /// one), the clear load, the store.
    fn create_render_pass_for_format(
        &mut self,
        desc: &RenderTargetDesc,
    ) -> Result<vk::VkRenderPass, RenderError> {
        let color = vk::VkAttachmentDescription {
            flags: 0,
            format: self.to_vk_format(desc.color_formats[0]) as u32,
            samples: vk::SampleCount::One,
            load_op: vk::AttachmentLoadOp::Clear,
            store_op: vk::AttachmentStoreOp::Store,
            stencil_load_op: vk::AttachmentLoadOp::DontCare,
            stencil_store_op: vk::AttachmentStoreOp::DontCare,
            initial_layout: vk::ImageLayout::Undefined,
            final_layout: vk::ImageLayout::ColorAttachmentOptimal,
        };
        let color_ref = vk::VkAttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::ColorAttachmentOptimal,
        };
        let mut attachments: Vec<vk::VkAttachmentDescription> = vec![color];
        let mut depth_ref: Option<vk::VkAttachmentReference> = None;
        if desc.depth_attachment {
            let depth = vk::VkAttachmentDescription {
                flags: 0,
                format: self.depth_format as u32,
                samples: vk::SampleCount::One,
                load_op: vk::AttachmentLoadOp::Clear,
                store_op: vk::AttachmentStoreOp::Store,
                stencil_load_op: vk::AttachmentLoadOp::Clear,
                stencil_store_op: vk::AttachmentStoreOp::DontCare,
                initial_layout: vk::ImageLayout::Undefined,
                final_layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
            };
            let depth_reference = vk::VkAttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
            };
            attachments.push(depth);
            depth_ref = Some(depth_reference);
        }
        let color_count = 1;
        let _ = color_count;
        let subpass = vk::VkSubpassDescription {
            flags: 0,
            pipeline_bind_point: vk::PipelineBindPoint::Graphics,
            input_attachment_count: 0,
            p_input_attachments: std::ptr::null(),
            color_attachment_count: 1,
            p_color_attachments: &color_ref,
            p_resolve_attachments: std::ptr::null(),
            p_depth_stencil_attachment: depth_ref
                .as_ref()
                .map(|r| r as *const _)
                .unwrap_or(std::ptr::null()),
            preserve_attachment_count: 0,
            p_preserve_attachments: std::ptr::null(),
        };
        let pass_info = vk::VkRenderPassCreateInfo {
            s_type: vk::StructureType::RenderPassCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 0,
            p_dependencies: std::ptr::null(),
        };
        let mut pass = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_render_pass.unwrap())(
                self.device,
                &pass_info,
                std::ptr::null(),
                &mut pass,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "create_render_target: vkCreateRenderPass failed",
            ));
        }
        Ok(pass)
    }


}

impl RenderBackend for VulkanBackend {
    fn describe(&self) -> String {
        if self.instance.is_null() {
            return "vulkan (not initialised)".to_string();
        }
        // The device's identity and the API version the describe() reports
        // (the same fields the C++ backend records).
        let api_major = (self.api_version >> 22) & 0x3FFu32;
        let api_minor = (self.api_version >> 12) & 0x3FFu32;
        let device_name = self.device_name_string();
        format!(
            "vulkan {}.{}; vendor={:#x}; renderer={}",
            api_major, api_minor, self.vendor_id, device_name
        )
    }

    fn set_clear_color(&mut self, color: ClearColor) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.clear_color = color;
    }

    fn clear_color(&self) -> ClearColor {
        self.clear_color
    }

    fn set_depth_test_enabled(&mut self, enabled: bool) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.depth_test_enabled = enabled;
    }

    fn clear(&mut self) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        // The clear opens the frame's command buffer and the render pass
        // (the attachments load the clear colour), then the diagnostic
        // draw runs on the default framebuffer only. The frame stays open:
        // the generic draw path records into the same pass, and the frame's
        // end (the swap_buffers path's submit) closes and submits the
        // command buffer (the same rule the OpenGL backend's clear
        // follows).
        self.begin_frame();
        if self.config.draw_triangle {
            self.draw_triangle();
        }
    }

    fn width(&self) -> u32 {
        self.config.width
    }

    fn height(&self) -> u32 {
        self.config.height
    }

    fn shutdown(&mut self) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.shutdown = true;
        unsafe {
            (self.loader.vk_device_wait_idle.unwrap())(self.device);
        }
    }

    fn violation_count(&self) -> u64 {
        self.violations
    }

    fn create_shader_program(&mut self, id: u32) -> Result<ShaderProgram, RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "create_shader_program: the backend is shut down",
            ));
        }
        Ok(ShaderProgram {
            inner: Arc::new(Mutex::new(ShaderProgramInner {
                id,
                compiled: true,
            })),
        })
    }

    fn create_gpu_mesh(&mut self) -> Result<GpuMesh, RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "create_gpu_mesh: the backend is shut down",
            ));
        }
        // The mesh's slot: the handle's id indexes the state vector. The
        // slot stays None until upload_mesh gives it the buffers.
        let slot = self.meshes.len();
        Ok(GpuMesh {
            inner: Arc::new(Mutex::new(GpuMeshInner {
                id: slot,
                vertex_count: 0,
                index_count: 0,
                uploaded: false,
            })),
        })
    }

    fn upload_mesh(
        &mut self,
        mesh: &GpuMesh,
        layout: &VertexLayout,
        vertices: &[u8],
        indices: &[u32],
    ) -> Result<(), RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "upload_mesh: the backend is shut down",
            ));
        }
        if layout.stride == 0 {
            return Err(RenderError::from(
                "upload_mesh: the layout's stride must be positive",
            ));
        }
        if vertices.len() as u32 % layout.stride != 0 {
            return Err(RenderError::from(
                "upload_mesh: the vertex data is not a multiple of the stride",
            ));
        }
        let mesh_id = mesh.inner.lock().unwrap().id;
        while self.meshes.len() <= mesh_id {
            self.meshes.push(None);
        }
        if self.meshes[mesh_id].is_some() {
            return Err(RenderError::from(
                "upload_mesh: the mesh was already uploaded",
            ));
        }
        if indices.is_empty() {
            return Err(RenderError::from(
                "upload_mesh: the index data is empty",
            ));
        }
        // The pipeline's descriptor set layout and the sampler must exist
        // before the per-mesh descriptor set is allocated (the mesh path's
        // ensure, the same lazy rule the diagnostic triangle follows).
        self.ensure_mesh_pipeline();
        if !self.mesh_pipeline_ready {
            return Err(RenderError::from(
                "upload_mesh: the mesh pipeline could not be created",
            ));
        }
        // The lit pipeline's ensure (M5): the same before-the-frame rule
        // (the pipeline and the buffers' creation inside an open frame's
        // recording is the software rasterizer's crash the debug run
        // caught).
        self.ensure_lit_pipeline();
        let vertex_size = vertices.len();
        let index_size = indices.len() * 4;
        let uniform_size = std::mem::size_of::<MeshUniform>();
        let (vertex_buffer, vertex_memory) = self.create_host_buffer(
            vertex_size,
            vk::BufferUsage::VertexBuffer,
        )?;
        let (index_buffer, index_memory) = self.create_host_buffer(
            index_size,
            vk::BufferUsage::IndexBuffer,
        )?;
        let (uniform_buffer, uniform_memory) = self.create_host_buffer(
            uniform_size,
            vk::BufferUsage::UniformBuffer,
        )?;
        let vertex_map = self.map_memory(vertex_memory)?;
        let index_map = self.map_memory(index_memory)?;
        let uniform_map = self.map_memory(uniform_memory)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr(),
                vertex_map as *mut u8,
                vertex_size,
            );
            std::ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                index_map as *mut u8,
                index_size,
            );
        }
        // The per-mesh descriptor set: the camera block at binding 0 (the
        // uniform buffer) and the default texture's combined image sampler
        // at binding 1 (the material path rewrites the write when a draw
        // binds a texture).
        let pool_sizes = [
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
            },
            vk::VkDescriptorPoolSize {
                descriptor_type: vk::DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
            },
        ];
        let pool_info = vk::VkDescriptorPoolCreateInfo {
            s_type: vk::StructureType::DescriptorPoolCreateInfo,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 2,
            p_pool_sizes: pool_sizes.as_ptr(),
        };
        let mut pool = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_create_descriptor_pool.unwrap())(
                self.device,
                &pool_info,
                std::ptr::null(),
                &mut pool,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "upload_mesh: vkCreateDescriptorPool failed",
            ));
        }
        let set_layouts = [self.mesh_set_layout];
        let alloc_info = vk::VkDescriptorSetAllocateInfo {
            s_type: vk::StructureType::DescriptorSetAllocateInfo,
            p_next: std::ptr::null(),
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
        };
        let mut set = std::ptr::null_mut();
        let result = unsafe {
            (self.loader.vk_allocate_descriptor_sets.unwrap())(
                self.device,
                &alloc_info,
                &mut set,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "upload_mesh: vkAllocateDescriptorSets failed",
            ));
        }
        let default_view = self.default_texture.as_ref().map(|t| t.view);
        self.write_mesh_descriptors(
            set,
            uniform_buffer,
            uniform_size,
            default_view,
            self.mesh_sampler,
        );
        self.meshes[mesh_id] = Some(VulkanMeshState {
            stride: layout.stride,
            vertex_buffer: VulkanResource {
                handle: vertex_buffer,
                memory: vertex_memory,
                released: false,
            },
            index_buffer: VulkanResource {
                handle: index_buffer,
                memory: index_memory,
                released: false,
            },
            uniform_buffer: VulkanResource {
                handle: uniform_buffer,
                memory: uniform_memory,
                released: false,
            },
            uniform_map,
            descriptor_set: set,
            vertex_count: (vertices.len() as u32) / layout.stride,
            index_count: indices.len() as u32,
        });
        {
            let mut inner = mesh.inner.lock().unwrap();
            inner.vertex_count = (vertices.len() as u32) / layout.stride;
            inner.index_count = indices.len() as u32;
            inner.uploaded = true;
        }
        Ok(())
    }

    fn draw(&mut self, mesh: &GpuMesh, _program: &ShaderProgram) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        let inner = mesh.inner.lock().unwrap();
        if !inner.uploaded {
            self.violations += 1;
            return;
        }
        let mesh_id = inner.id;
        drop(inner);
        self.draw_mesh_by_id(mesh_id, Mat4::IDENTITY);
    }

    fn submit(&mut self, submission: &DrawSubmission) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        for light in &submission.lights {
            if light.validate().is_err() {
                self.violations += 1;
            }
        }
        // The generic path: the model matrix from the submission, the
        // camera block from the set_camera state (the same uniform buffer
        // the draw() path writes).
        let mesh_id = submission.mesh.inner.lock().unwrap().id;
        let has_lights = submission.sun.is_some() || !submission.lights.is_empty();
        if has_lights {
            // The lit path (M5): the lights' block from the submission's
            // sun and point lights (the sun absent means intensity zero),
            // the lit pipeline draws.
            let default_sun = DirectionalLight {
                direction: Vec3::new(0.0, -1.0, 0.0),
                color: Vec3::new(0.0, 0.0, 0.0),
                intensity: 0.0,
            };
            let sun = submission.sun.as_ref().unwrap_or(&default_sun);
            self.lit_block = LightsUniform::from_lights(sun, &submission.lights);
            if self.ensure_lit_pipeline() {
                self.draw_mesh_lit_by_id(mesh_id, submission.model);
                return;
            }
            // The lit path's failure falls back to the unlit draw (the
            // render keeps going; the fallback is the milestone's first
            // step's forgiveness).
        }
        self.draw_mesh_by_id(mesh_id, submission.model);
    }

    /// The deferred lighting pass (M6): the two internal steps the
    /// contract's submit_lighting_pass carries. Step one (the geometry
    /// pass) opens the G-buffer target's render pass, binds the G-buffer
    /// pipeline, writes the camera block and the bound texture's view,
    /// draws the mesh and closes the pass; the colour attachments hand
    /// off to the shader reads through the pipeline barrier the handoff
    /// records. Step two (the lighting pass) writes the lights' block,
    /// rewrites the three G-buffer attachments' samplers, opens the
    /// default framebuffer's render pass, binds the lighting pipeline,
    /// draws the fullscreen triangle (the three vertices the vertex
    /// index generates) and closes the pass. The frame stays open across
    /// the two steps (the same rule the generic draw path follows).
    fn submit_lighting_pass(
        &mut self,
        submission: &DrawSubmission,
        gbuffer: &RenderTarget,
    ) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        for light in &submission.lights {
            if light.validate().is_err() {
                self.violations += 1;
            }
        }
        let target_index = match self
            .targets
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &gbuffer.inner))
        {
            Some(index) => index,
            None => {
                // A foreign target is a protocol violation (the same rule
                // the target handles follow).
                self.violations += 1;
                return;
            }
        };
        if self.targets[target_index].released {
            self.violations += 1;
            return;
        }
        let mesh_id = submission.mesh.inner.lock().unwrap().id;
        if self.meshes.get(mesh_id).map(|m| m.is_none()).unwrap_or(true) {
            self.violations += 1;
            return;
        }
        if !self.ensure_deferred_pipeline() {
            return;
        }
        if self.command_buffer.is_null() || self.frame_open {
            // The deferred pass runs inside the frame's open command
            // buffer (the clear() the frame's first step runs opened it);
            // a missing frame is a violation, an already-open frame is
            // the frame the pass records into.
            if self.command_buffer.is_null() {
                self.violations += 1;
                return;
            }
        }
        // The lights' block: the same from_lights the forward path's
        // block carries (the sun absent means the intensity zero; the
        // point lights' four slots).
        let default_sun = DirectionalLight {
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::new(0.0, 0.0, 0.0),
            intensity: 0.0,
        };
        let sun = submission.sun.as_ref().unwrap_or(&default_sun);
        let lights_block = LightsUniform::from_lights(sun, &submission.lights);
        if !self.lighting_lights_map.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &lights_block as *const LightsUniform as *const u8,
                    self.lighting_lights_map as *mut u8,
                    LightsUniform::SIZE,
                );
            }
        }
        // The camera block: the view-projection matrix from the set_camera
        // state and the model matrix the submission carries.
        let aspect = self.config.width as f32 / self.config.height as f32;
        let camera = self.camera;
        let view = Mat4::look_at(camera.eye, camera.target, camera.up);
        let projection =
            Mat4::perspective(camera.fov_y_rad, aspect, camera.near, camera.far);
        let view_projection = projection.multiply(view);
        let uniform = MeshUniform {
            view_projection: view_projection.m,
            model: submission.model.m,
        };
        if !self.gbuffer_camera_map.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &uniform as *const MeshUniform as *const u8,
                    self.gbuffer_camera_map as *mut u8,
                    std::mem::size_of::<MeshUniform>(),
                );
            }
        }
        // The bound texture's view: the bind_texture state's index (the
        // default texture's view when the None carries).
        let bound_view = self
            .bound_texture
            .and_then(|i| self.textures_2d.get(i))
            .filter(|t| !t.released)
            .map(|t| t.view)
            .or_else(|| self.default_texture.as_ref().map(|t| t.view));
        if let Some(view) = bound_view {
            self.write_mesh_texture_descriptor(
                self.gbuffer_descriptor_set,
                view,
            );
        }
        // The lighting pass's three G-buffer attachments' samplers: the
        // target's views (the albedo at 0, the normal at 1, the world
        // position at 2).
        {
            let target = &self.targets[target_index];
            let views: Vec<vk::VkImageView> = target
                .views
                .iter()
                .take(3)
                .map(|v| v.view)
                .collect();
            if views.len() == 3 {
                let lights_buffer = self
                    .lighting_lights_buffer
                    .as_ref()
                    .map(|r| r.handle)
                    .unwrap_or(std::ptr::null_mut());
                self.write_lighting_descriptors(
                    self.lighting_descriptor_set,
                    lights_buffer,
                    views[0],
                    views[1],
                    views[2],
                );
            }
        }
        let (vertex_buffer, index_buffer, index_count) = {
            let state = self.meshes[mesh_id].as_ref().unwrap();
            (
                state.vertex_buffer.handle,
                state.index_buffer.handle,
                state.index_count,
            )
        };
        let target_pass = self.targets[target_index].render_pass;
        let target_framebuffer = self.targets[target_index].framebuffer;
        let command_buffer = self.command_buffer;
        // ---- Step one: the geometry pass ----
        // The default framebuffer's render pass the begin_frame opened
        // (the clear's first step) closes: the two deferred passes record
        // into the same frame's command buffer, and a render pass opens
        // only after the previous one closes (the same rule the
        // submit_frame's end follows).
        if self.pass_open {
            unsafe {
                (self.loader.vk_cmd_end_render_pass.unwrap())(
                    command_buffer,
                );
            }
            self.pass_open = false;
        }
        // The G-buffer target's render pass opens with the clear values
        // (the clear colour the target carries, the depth the clear state
        // carries). The frame's command buffer stays open.
        let mut clear_values = [
            vk::VkClearValue {
                color: [0.0, 0.0, 0.0, 0.0],
            },
            vk::VkClearValue {
                color: [0.0, 0.0, 0.0, 0.0],
            },
            vk::VkClearValue {
                color: [0.0, 0.0, 0.0, 0.0],
            },
            vk::VkClearValue {
                depth_stencil: vk::VkClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
        ];
        let colour = self.clear_color;
        clear_values[0].color = [
            colour.r as f32 / 255.0,
            colour.g as f32 / 255.0,
            colour.b as f32 / 255.0,
            colour.a as f32 / 255.0,
        ];
        clear_values[3].depth_stencil = vk::VkClearDepthStencilValue {
            depth: self.clear_depth,
            stencil: self.clear_stencil as u32,
        };
        let area = vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        };
        let pass_begin = vk::VkRenderPassBeginInfo {
            s_type: vk::StructureType::RenderPassBeginInfo,
            p_next: std::ptr::null(),
            render_pass: target_pass,
            framebuffer: target_framebuffer,
            render_area: area,
            clear_value_count: 4,
            p_clear_values: clear_values.as_ptr(),
        };
        unsafe {
            (self.loader.vk_cmd_begin_render_pass.unwrap())(
                command_buffer,
                &pass_begin,
                0,
            );
        }
        self.pass_open = true;
        // The G-buffer pipeline binds and the mesh draws (the same body
        // the draw_mesh_inner's record carries, with the gbuffer
        // pipeline and the gbuffer descriptor set).
        let viewports = [vk::VkViewport {
            x: 0.0,
            y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        }];
        let offsets = [0u64];
        let descriptor_sets = [self.gbuffer_descriptor_set];
        unsafe {
            (self.loader.vk_cmd_bind_pipeline.unwrap())(
                command_buffer,
                vk::PipelineBindPoint::Graphics,
                self.gbuffer_pipeline,
            );
            (self.loader.vk_cmd_set_viewport.unwrap())(
                command_buffer,
                0,
                1,
                viewports.as_ptr(),
            );
            (self.loader.vk_cmd_set_scissor.unwrap())(
                command_buffer,
                0,
                1,
                scissors.as_ptr(),
            );
            (self.loader.vk_cmd_bind_descriptor_sets.unwrap())(
                command_buffer,
                vk::PipelineBindPoint::Graphics,
                self.gbuffer_pipeline_layout,
                0,
                1,
                descriptor_sets.as_ptr(),
                0,
                std::ptr::null(),
            );
            (self.loader.vk_cmd_bind_vertex_buffers.unwrap())(
                command_buffer,
                0,
                1,
                &vertex_buffer,
                offsets.as_ptr(),
            );
            (self.loader.vk_cmd_bind_index_buffer.unwrap())(
                command_buffer,
                index_buffer,
                0,
                vk::IndexType::Uint32,
            );
            (self.loader.vk_cmd_draw_indexed.unwrap())(
                command_buffer,
                index_count,
                1,
                0,
                0,
                0,
            );
        }
        // The geometry pass closes; the colour attachments hand off to
        // the lighting pass's shader reads through the pipeline barrier
        // (the three colour attachments' ColorAttachmentOptimal ->
        // ShaderReadOnlyOptimal transition; the depth attachment stays
        // DepthStencilAttachmentOptimal — the lighting pass's triangle
        // never writes depth and the next frame's geometry pass opens
        // the pass with the clear load op).
        unsafe {
            (self.loader.vk_cmd_end_render_pass.unwrap())(command_buffer);
        }
        self.pass_open = false;
        let target_images: Vec<vk::VkImage> = self.targets[target_index]
            .views
            .iter()
            .take(3)
            .map(|v| v.image)
            .collect();
        for image in target_images.iter() {
            self.transition_image_layout(
                command_buffer,
                *image,
                vk::ImageAspect::Color as u32,
                vk::ImageLayout::ColorAttachmentOptimal,
                vk::ImageLayout::ShaderReadOnlyOptimal,
            );
        }
        // ---- Step two: the lighting pass ----
        // The default framebuffer's render pass opens (the same pass the
        // begin_frame opens; the clear values land on the swap's present
        // path).
        let mut default_clear_values = [
            vk::VkClearValue {
                color: [0.0, 0.0, 0.0, 0.0],
            },
            vk::VkClearValue {
                depth_stencil: vk::VkClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
        ];
        default_clear_values[0].color = [
            colour.r as f32 / 255.0,
            colour.g as f32 / 255.0,
            colour.b as f32 / 255.0,
            colour.a as f32 / 255.0,
        ];
        default_clear_values[1].depth_stencil = vk::VkClearDepthStencilValue {
            depth: self.clear_depth,
            stencil: self.clear_stencil as u32,
        };
        let default_pass_begin = vk::VkRenderPassBeginInfo {
            s_type: vk::StructureType::RenderPassBeginInfo,
            p_next: std::ptr::null(),
            render_pass: self.render_pass,
            framebuffer: self.framebuffer,
            render_area: area,
            clear_value_count: 2,
            p_clear_values: default_clear_values.as_ptr(),
        };
        unsafe {
            (self.loader.vk_cmd_begin_render_pass.unwrap())(
                command_buffer,
                &default_pass_begin,
                0,
            );
        }
        self.pass_open = true;
        // The lighting pipeline binds and the fullscreen triangle draws
        // (the three vertices the vertex index generates, no vertex
        // buffer, no index buffer).
        let descriptor_sets = [self.lighting_descriptor_set];
        unsafe {
            (self.loader.vk_cmd_bind_pipeline.unwrap())(
                command_buffer,
                vk::PipelineBindPoint::Graphics,
                self.lighting_pipeline,
            );
            (self.loader.vk_cmd_set_viewport.unwrap())(
                command_buffer,
                0,
                1,
                viewports.as_ptr(),
            );
            (self.loader.vk_cmd_set_scissor.unwrap())(
                command_buffer,
                0,
                1,
                scissors.as_ptr(),
            );
            (self.loader.vk_cmd_bind_descriptor_sets.unwrap())(
                command_buffer,
                vk::PipelineBindPoint::Graphics,
                self.lighting_pipeline_layout,
                0,
                1,
                descriptor_sets.as_ptr(),
                0,
                std::ptr::null(),
            );
            (self.loader.vk_cmd_draw.unwrap())(command_buffer, 3, 1, 0, 0);
        }
        // The lighting pass closes. The frame stays open (the swap's
        // present path closes and submits it; the same rule the generic
        // draw path follows).
        unsafe {
            (self.loader.vk_cmd_end_render_pass.unwrap())(command_buffer);
        }
        self.pass_open = false;
        // The G-buffer attachments hand back to ColorAttachmentOptimal:
        // the next frame's geometry pass opens the pass with the clear
        // load op (the same rule the readback's layout table follows).
        for image in target_images.iter() {
            self.transition_image_layout(
                command_buffer,
                *image,
                vk::ImageAspect::Color as u32,
                vk::ImageLayout::ShaderReadOnlyOptimal,
                vk::ImageLayout::ColorAttachmentOptimal,
            );
        }
    }

    fn submit_post_chain(
        &mut self,
        submission: &DrawSubmission,
        hdr_target: &RenderTarget,
        config: &PostChainConfig,
    ) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        for light in &submission.lights {
            if light.validate().is_err() {
                self.violations += 1;
            }
        }
        let target_index = match self
            .targets
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &hdr_target.inner))
        {
            Some(index) => index,
            None => {
                self.violations += 1;
                return;
            }
        };
        if self.targets[target_index].released {
            self.violations += 1;
            return;
        }
        let mesh_id = submission.mesh.inner.lock().unwrap().id;
        if mesh_id >= self.meshes.len() || self.meshes[mesh_id].is_none() {
            self.violations += 1;
            return;
        }
        if !self.ensure_post_pipeline() {
            return;
        }
        if !self.frame_open {
            self.violations += 1;
            return;
        }
        // The lights' block: the same from_lights the forward path's
        // block carries.
        let default_sun = DirectionalLight {
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::new(0.0, 0.0, 0.0),
            intensity: 0.0,
        };
        let sun = submission.sun.as_ref().unwrap_or(&default_sun);
        let lights_block = LightsUniform::from_lights(sun, &submission.lights);
        if !self.post_lights_map.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &lights_block as *const LightsUniform as *const u8,
                    self.post_lights_map as *mut u8,
                    LightsUniform::SIZE,
                );
            }
        }
        // The camera block: the same MeshUniform the mesh path carries.
        let aspect = self.config.width as f32 / self.config.height as f32;
        let camera = self.camera;
        let view = Mat4::look_at(camera.eye, camera.target, camera.up);
        let projection =
            Mat4::perspective(camera.fov_y_rad, aspect, camera.near, camera.far);
        let view_projection = projection.multiply(view);
        let uniform = MeshUniform {
            view_projection: view_projection.m,
            model: submission.model.m,
        };
        if !self.post_camera_map.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &uniform as *const MeshUniform as *const u8,
                    self.post_camera_map as *mut u8,
                    std::mem::size_of::<MeshUniform>(),
                );
            }
        }
        // The config block: the three vec4 params the POST_COMPOSITE
        // shader's Config block carries (the toggles, the bloom's
        // parameters, the focus's parameters).
        let config_block = [
            [
                if config.tonemapping { 1.0f32 } else { 0.0 },
                if config.bloom { 1.0 } else { 0.0 },
                if config.depth_of_field { 1.0 } else { 0.0 },
                if config.ssao { 1.0 } else { 0.0 },
            ],
            [
                config.bloom_threshold,
                config.bloom_knee,
                config.bloom_intensity,
                config.ssao_radius,
            ],
            [
                config.focal_distance,
                config.focus_near_range,
                config.focus_far_range,
                if config.anti_aliasing { 1.0 } else { 0.0 },
            ],
        ];
        if !self.post_config_map.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    config_block.as_ptr() as *const u8,
                    self.post_config_map as *mut u8,
                    48,
                );
            }
        }
        // The volume's block: the origin, the dims, the cell size and
        // the indirect light's toggle and intensity the POST_GBUFFER
        // fragment's Volume block reads (the 48-byte std140 block).
        let volume_block = VolumeUniform {
            origin: self.volume_uniform.origin,
            dims: self.volume_uniform.dims,
            params: [
                self.volume_uniform.params[0],
                config.indirect_intensity,
                if config.indirect_light { 1.0f32 } else { 0.0 },
                0.0,
            ],
        };
        if !self.post_volume_map.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &volume_block as *const VolumeUniform as *const u8,
                    self.post_volume_map as *mut u8,
                    std::mem::size_of::<VolumeUniform>(),
                );
            }
        }
        // The bound texture's view (the surface texture the geometry
        // pass samples).
        let bound_view = self
            .bound_texture
            .and_then(|i| self.textures_2d.get(i))
            .filter(|t| !t.released)
            .map(|t| t.view)
            .or_else(|| self.default_texture.as_ref().map(|t| t.view));
        if let Some(view) = bound_view {
            self.write_post_texture_descriptor(
                self.post_gbuffer_descriptor_set,
                view,
            );
        }
        // The volume's sampler: the light volume's 3D texture the
        // set_light_volume call carries (the toggle's violation when
        // the config asks for the indirect light with no volume set).
        if config.indirect_light {
            match self.light_volume.as_ref() {
                Some(volume) => {
                    self.write_post_volume_descriptor(
                        self.post_gbuffer_descriptor_set,
                        volume.view,
                    );
                }
                None => {
                    self.violations += 1;
                }
            }
        }
        // The composite pass's two HDR attachments' samplers.
        {
            let target = &self.targets[target_index];
            if target.views.len() >= 2 {
                self.write_post_composite_descriptors(
                    self.post_composite_descriptor_set,
                    self.post_config_buffer
                        .as_ref()
                        .map(|r| r.handle)
                        .unwrap_or(std::ptr::null_mut()),
                    target.views[0].view,
                    target.views[1].view,
                );
            }
        }
        let (vertex_buffer, index_buffer, index_count) = {
            let state = self.meshes[mesh_id].as_ref().unwrap();
            (
                state.vertex_buffer.handle,
                state.index_buffer.handle,
                state.index_count,
            )
        };
        let command_buffer = self.command_buffer;
        // The default framebuffer's pass closes (the same rule the
        // deferred path's step one follows).
        if self.pass_open {
            unsafe {
                (self.loader.vk_cmd_end_render_pass.unwrap())(
                    command_buffer,
                );
            }
            self.pass_open = false;
        }
        // The HDR render pass's clear values (the radiance attachment
        // clears to the clear colour's radiance; the world attachment
        // clears to zero — the alpha 0 marks the background).
        let colour = self.clear_color;
        let clear_r = colour.r as f32 / 255.0;
        let clear_g = colour.g as f32 / 255.0;
        let clear_b = colour.b as f32 / 255.0;
        let clear_a = colour.a as f32 / 255.0;
        let clear_values = [
            vk::VkClearValue {
                color: [clear_r, clear_g, clear_b, clear_a],
            },
            vk::VkClearValue {
                color: [0.0, 0.0, 0.0, 0.0],
            },
            vk::VkClearValue {
                depth_stencil: vk::VkClearDepthStencilValue {
                    depth: self.clear_depth,
                    stencil: self.clear_stencil as u32,
                },
            },
        ];
        let area = vk::VkRect2D {
            offset: vk::VkOffset2D { x: 0, y: 0 },
            extent: vk::VkExtent2D {
                width: self.config.width,
                height: self.config.height,
            },
        };
        let viewports = [vk::VkViewport {
            x: 0.0,
            y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [area];
        let target = &self.targets[target_index];
        // The target's own framebuffer: the desc's two-attachment HDR
        // shape built against the shared post_gbuffer_render_pass (the
        // same rule the G-buffer's desc follows).
        let post_framebuffer = target.framebuffer;
        let pass_begin = vk::VkRenderPassBeginInfo {
            s_type: vk::StructureType::RenderPassBeginInfo,
            p_next: std::ptr::null(),
            render_pass: self.post_gbuffer_render_pass,
            framebuffer: post_framebuffer,
            render_area: area,
            clear_value_count: 3,
            p_clear_values: clear_values.as_ptr(),
        };
        unsafe {
            (self.loader.vk_cmd_begin_render_pass.unwrap())(
                command_buffer,
                &pass_begin,
                0,
            );
        }
        // The geometry draw: the post pipeline binds (the same PNU
        // vertex path the mesh draws carry).
        let descriptor_sets = [self.post_gbuffer_descriptor_set];
        unsafe {
            (self.loader.vk_cmd_bind_pipeline.unwrap())(
                command_buffer,
                vk::PipelineBindPoint::Graphics,
                self.post_pipeline,
            );
            (self.loader.vk_cmd_set_viewport.unwrap())(
                command_buffer,
                0,
                1,
                viewports.as_ptr(),
            );
            (self.loader.vk_cmd_set_scissor.unwrap())(
                command_buffer,
                0,
                1,
                scissors.as_ptr(),
            );
            (self.loader.vk_cmd_bind_descriptor_sets.unwrap())(
                command_buffer,
                vk::PipelineBindPoint::Graphics,
                self.post_gbuffer_pipeline_layout,
                0,
                1,
                descriptor_sets.as_ptr(),
                0,
                std::ptr::null(),
            );
            let vertex_offsets = [0u64];
            (self.loader.vk_cmd_bind_vertex_buffers.unwrap())(
                command_buffer,
                0,
                1,
                [vertex_buffer].as_ptr(),
                vertex_offsets.as_ptr(),
            );
            (self.loader.vk_cmd_bind_index_buffer.unwrap())(
                command_buffer,
                index_buffer,
                0,
                vk::IndexType::Uint32,
            );
            (self.loader.vk_cmd_draw_indexed.unwrap())(
                command_buffer,
                index_count,
                1,
                0,
                0,
                0,
            );
        }
        unsafe {
            (self.loader.vk_cmd_end_render_pass.unwrap())(command_buffer);
        }
        // The two HDR attachments hand off to the composite pass's
        // shader reads (the ColorAttachmentOptimal ->
        // ShaderReadOnlyOptimal transition the deferred path records).
        let target_images: Vec<vk::VkImage> = self.targets[target_index]
            .views
            .iter()
            .take(2)
            .map(|v| v.image)
            .collect();
        for image in target_images.iter() {
            self.transition_image_layout(
                command_buffer,
                *image,
                vk::ImageAspect::Color as u32,
                vk::ImageLayout::ColorAttachmentOptimal,
                vk::ImageLayout::ShaderReadOnlyOptimal,
            );
        }
        // ---- The composite pass ----
        // The default framebuffer's render pass opens (the composite
        // writes every pixel the triangle covers).
        let default_clear_values = [
            vk::VkClearValue {
                color: [clear_r, clear_g, clear_b, clear_a],
            },
            vk::VkClearValue {
                depth_stencil: vk::VkClearDepthStencilValue {
                    depth: self.clear_depth,
                    stencil: self.clear_stencil as u32,
                },
            },
        ];
        let default_pass_begin = vk::VkRenderPassBeginInfo {
            s_type: vk::StructureType::RenderPassBeginInfo,
            p_next: std::ptr::null(),
            render_pass: self.render_pass,
            framebuffer: self.framebuffer,
            render_area: area,
            clear_value_count: 2,
            p_clear_values: default_clear_values.as_ptr(),
        };
        unsafe {
            (self.loader.vk_cmd_begin_render_pass.unwrap())(
                command_buffer,
                &default_pass_begin,
                0,
            );
        }
        self.pass_open = true;
        let descriptor_sets = [self.post_composite_descriptor_set];
        unsafe {
            (self.loader.vk_cmd_bind_pipeline.unwrap())(
                command_buffer,
                vk::PipelineBindPoint::Graphics,
                self.post_composite_pipeline,
            );
            (self.loader.vk_cmd_set_viewport.unwrap())(
                command_buffer,
                0,
                1,
                viewports.as_ptr(),
            );
            (self.loader.vk_cmd_set_scissor.unwrap())(
                command_buffer,
                0,
                1,
                scissors.as_ptr(),
            );
            (self.loader.vk_cmd_bind_descriptor_sets.unwrap())(
                command_buffer,
                vk::PipelineBindPoint::Graphics,
                self.post_composite_pipeline_layout,
                0,
                1,
                descriptor_sets.as_ptr(),
                0,
                std::ptr::null(),
            );
            (self.loader.vk_cmd_draw.unwrap())(command_buffer, 3, 1, 0, 0);
        }
        unsafe {
            (self.loader.vk_cmd_end_render_pass.unwrap())(command_buffer);
        }
        self.pass_open = false;
        // The HDR attachments hand back to ColorAttachmentOptimal (the
        // same rule the deferred path follows).
        for image in target_images.iter() {
            self.transition_image_layout(
                command_buffer,
                *image,
                vk::ImageAspect::Color as u32,
                vk::ImageLayout::ShaderReadOnlyOptimal,
                vk::ImageLayout::ColorAttachmentOptimal,
            );
        }
    }

    fn create_render_target(
        &mut self,
        desc: &RenderTargetDesc,
    ) -> Result<RenderTarget, RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "create_render_target: the backend is shut down",
            ));
        }
        desc.validate()?;
        let mut views = Vec::new();
        for i in 0..desc.color_attachment_count as usize {
            // The Sampled bit joins the colour attachment bit: the
            // deferred lighting pass's samplers read the attachments (the
            // descriptor's CombinedImageSampler requires the view's
            // image to carry the Sampled bit — the same rule the
            // render_target_texture's sampling handles follow).
            let (image, memory, view) = self.create_offscreen_image(
                desc.width,
                desc.height,
                self.to_vk_format(desc.color_formats[i]),
                vk::ImageUsage::ColorAttachment | vk::ImageUsage::Sampled,
            )?;
            views.push(VulkanTexture2D {
                inner: Arc::new(Mutex::new(TextureViewInner {
                    width: desc.width,
                    height: desc.height,
                    depth: 1,
                    format: desc.color_formats[i],
                    is_depth: false,
                    released: false,
                })),
                image,
                memory,
                view,
                released: false,
            });
        }
        let depth_view = if desc.depth_attachment {
            let (image, memory, view) = self.create_offscreen_image(
                desc.width,
                desc.height,
                self.depth_format,
                vk::ImageUsage::DepthStencilAttachment as u32,
            )?;
            Some(VulkanTexture2D {
                inner: Arc::new(Mutex::new(TextureViewInner {
                    width: desc.width,
                    height: desc.height,
                    depth: 1,
                    format: desc.depth_format,
                    is_depth: true,
                    released: false,
                })),
                image,
                memory,
                view,
                released: false,
            })
        } else {
            None
        };
        // The per-target render pass and framebuffer: the pipeline
        // creation reads the pass the draw-time check compares (the
        // pipeline's render_pass must match the framebuffer's pass). The
        // pass's shape follows the desc: one colour attachment per
        // desc entry plus the depth attachment when the desc asks for
        // one. The three-attachment MRT desc (the G-buffer) builds
        // against the gbuffer_render_pass the pipeline creation
        // registers; the single-attachment desc builds its own pass.
        let (target_pass, framebuffer) = self.create_target_pass_and_framebuffer(
            desc,
            &views,
            &depth_view,
        )?;
        let target = VulkanRenderTarget {
            inner: Arc::new(Mutex::new(RenderTargetInner {
                width: desc.width,
                height: desc.height,
                color_attachment_count: desc.color_attachment_count,
                depth_attachment: desc.depth_attachment,
                released: false,
            })),
            views,
            depth_view,
            render_pass: target_pass,
            framebuffer,
            released: false,
        };
        let handle = RenderTarget {
            inner: target.inner.clone(),
        };
        self.targets.push(target);
        Ok(handle)
    }

    fn begin_render_pass(&mut self, target: Option<&RenderTarget>) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        match target {
            None => {
                self.bound_target = None;
            }
            Some(t) => {
                match self
                    .targets
                    .iter()
                    .position(|x| Arc::ptr_eq(&x.inner, &t.inner))
                {
                    Some(index) => {
                        if self.targets[index].released {
                            self.violations += 1;
                        } else {
                            self.bound_target = Some(t.inner.clone());
                        }
                    }
                    None => {
                        self.violations += 1;
                    }
                }
            }
        }
    }

    fn end_render_pass(&mut self) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.bound_target = None;
    }

    fn release_render_target(&mut self, target: &RenderTarget) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        match self
            .targets
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &target.inner))
        {
            Some(index) => {
                if self.targets[index].released {
                    self.violations += 1;
                } else {
                    self.targets[index].released = true;
                    target.inner.lock().unwrap().released = true;
                }
            }
            None => {
                self.violations += 1;
            }
        }
    }

    fn render_target_texture(&self, target: &RenderTarget) -> Option<TextureView> {
        match self
            .targets
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &target.inner))
        {
            Some(index) => {
                let t = &self.targets[index];
                if t.released {
                    None
                } else {
                    t.views.first().map(|v| TextureView {
                        inner: v.inner.clone(),
                    })
                }
            }
            None => None,
        }
    }

    fn render_target_texture_at(
        &self,
        target: &RenderTarget,
        attachment: u32,
    ) -> Option<TextureView> {
        match self
            .targets
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &target.inner))
        {
            Some(index) => {
                let t = &self.targets[index];
                if t.released {
                    return None;
                }
                if attachment == DEPTH_ATTACHMENT {
                    return t.depth_view.as_ref().map(|d| TextureView {
                        inner: d.inner.clone(),
                    });
                }
                t.views.get(attachment as usize).map(|v| TextureView {
                    inner: v.inner.clone(),
                })
            }
            None => None,
        }
    }

    fn bind_texture(&mut self, view: Option<&TextureView>, unit: u32) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        // The first sampler unit: the mesh path's descriptor set samples
        // the bound texture (the default texture's view when none is
        // bound). The later units record the bound view without the
        // descriptor write (the generic path's first step carries one
        // unit).
        if unit != 0 {
            return;
        }
        match view {
            None => {
                self.bound_texture = None;
            }
            Some(v) => {
                let index = self
                    .textures_2d
                    .iter()
                    .position(|x| Arc::ptr_eq(&x.inner, &v.inner));
                self.bound_texture = index.map(|i| i);
            }
        }
    }

    fn create_texture_2d(&mut self, desc: &Texture2DDesc) -> Result<TextureView, RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "create_texture_2d: the backend is shut down",
            ));
        }
        let (image, memory, view) = self.create_offscreen_image(
            desc.width,
            desc.height,
            self.to_vk_format(desc.format),
            // The Sampled bit joins the transfer destination: the fill
            // writes through the transfer path and the shader reads through
            // the sampled view (the same usage pair every sampled texture
            // needs).
            vk::ImageUsage::TransferDst as u32 | vk::ImageUsage::Sampled as u32,
        )?;
        let texture = VulkanTexture2D {
            inner: Arc::new(Mutex::new(TextureViewInner {
                width: desc.width,
                height: desc.height,
                depth: 1,
                format: desc.format,
                is_depth: false,
                released: false,
            })),
            image,
            memory,
            view,
            released: false,
        };
        let handle = TextureView {
            inner: texture.inner.clone(),
        };
        self.textures_2d.push(texture);
        Ok(handle)
    }

    fn release_texture_2d(&mut self, view: &TextureView) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        match self
            .textures_2d
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &view.inner))
        {
            Some(index) => {
                if self.textures_2d[index].released {
                    self.violations += 1;
                } else {
                    self.textures_2d[index].released = true;
                    view.inner.lock().unwrap().released = true;
                }
            }
            None => {
                self.violations += 1;
            }
        }
    }

    fn fill_texture_2d(&mut self, view: &TextureView, data: &[u8]) -> Result<(), RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "fill_texture_2d: the backend is shut down",
            ));
        }
        let _index = match self
            .textures_2d
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &view.inner))
        {
            Some(index) => index,
            None => {
                self.violations += 1;
                return Err(RenderError::from(
                    "fill_texture_2d: the texture is not owned by this backend",
                ));
            }
        };
        let inner = view.inner.lock().unwrap();
        let expected = (inner.width as usize) * (inner.height as usize)
            * bytes_per_pixel(inner.format) as usize;
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "fill_texture_2d: the byte count is {} but the texture is {}",
                data.len(),
                expected
            )));
        }
        if self.readback_command.is_null() {
            return Err(RenderError::from(
                "fill_texture_2d: the command path is not initialised",
            ));
        }
        // The staging upload: one host-visible buffer, the CPU writes the
        // texels, the one-off command buffer copies it into the image (the
        // transition the write needs: Undefined -> TransferDst -> the
        // shader read).
        let (width, height, format) = {
            let inner = view.inner.lock().unwrap();
            (inner.width, inner.height, inner.format)
        };
        let index = _index;
        let image = self.textures_2d[index].image;
        let (staging, staging_memory) = self.create_host_buffer(
            data.len(),
            vk::BufferUsage::TransferSrc,
        )?;
        let map = self.map_memory(staging_memory)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                map as *mut u8,
                data.len(),
            );
        }
        // The one-off command buffer: the reset, the begin, the copy.
        unsafe {
            (self.loader.vk_reset_command_buffer.unwrap())(
                self.readback_command,
                0,
            );
        }
        let begin_info = vk::VkCommandBufferBeginInfo {
            s_type: vk::StructureType::CommandBufferBeginInfo,
            p_next: std::ptr::null(),
            flags: vk::CommandBufferUsage::OneTimeSubmit as u32,
            p_inheritance_info: std::ptr::null(),
        };
        let result = unsafe {
            (self.loader.vk_begin_command_buffer.unwrap())(
                self.readback_command,
                &begin_info,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "fill_texture_2d: the command buffer could not begin",
            ));
        }
        let copy = vk::VkBufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::VkImageSubresourceLayers {
                aspect_mask: vk::ImageAspect::Color as u32,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::VkOffset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::VkExtent3D {
                width,
                height,
                depth: 1,
            },
        };
        self.transition_image_layout(
            self.readback_command,
            image,
            vk::ImageAspect::Color as u32,
            vk::ImageLayout::Undefined,
            vk::ImageLayout::TransferDstOptimal,
        );
        unsafe {
            (self.loader.vk_cmd_copy_buffer_to_image.unwrap())(
                self.readback_command,
                staging,
                image,
                vk::ImageLayout::TransferDstOptimal,
                1,
                &copy,
            );
        }
        self.transition_image_layout(
            self.readback_command,
            image,
            vk::ImageAspect::Color as u32,
            vk::ImageLayout::TransferDstOptimal,
            vk::ImageLayout::ShaderReadOnlyOptimal,
        );
        let result = unsafe {
            (self.loader.vk_end_command_buffer.unwrap())(
                self.readback_command,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "fill_texture_2d: the command buffer could not end",
            ));
        }
        let submit = vk::VkSubmitInfo {
            s_type: vk::StructureType::SubmitInfo,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &self.readback_command,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };
        unsafe {
            (self.loader.vk_queue_submit.unwrap())(
                self.queue,
                1,
                &submit,
                std::ptr::null_mut(),
            );
            (self.loader.vk_queue_wait_idle.unwrap())(self.queue);
        }
        // The texture's usage must carry the Sampled bit for the shader
        // read: the creation path's TransferDst-only usage cannot satisfy
        // the descriptor, so the fill records the stage the shader reads
        // and the draw path's validation catches the missing bit.
        let _ = (width, height, format);
        Ok(())
    }

    fn read_texture_2d(
        &self,
        view: &TextureView,
        data: &mut [u8],
    ) -> Result<bool, RenderError> {
        let index = match self
            .textures_2d
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &view.inner))
        {
            Some(index) => index,
            None => {
                return Err(RenderError::from(
                    "read_texture_2d: the texture is not owned by this backend",
                ));
            }
        };
        if self.textures_2d[index].released {
            return Ok(false);
        }
        let inner = view.inner.lock().unwrap();
        let expected = (inner.width as usize) * (inner.height as usize)
            * bytes_per_pixel(inner.format) as usize;
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "read_texture_2d: the byte count is {} but the texture is {}",
                data.len(),
                expected
            )));
        }
        Ok(true)
    }

    fn create_texture_3d(
        &mut self,
        desc: &Texture3DDesc,
    ) -> Result<TextureView, RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "create_texture_3d: the backend is shut down",
            ));
        }
        desc.validate()?;
        let (image, memory, view) = self.create_offscreen_image_3d(
            desc.width,
            desc.height,
            desc.depth,
            self.to_vk_format(desc.format),
            // The Sampled bit joins the transfer destination: the fill
            // writes through the transfer path and the shader reads
            // through the sampled view (the same usage pair the 2D
            // textures carry).
            vk::ImageUsage::TransferDst as u32 | vk::ImageUsage::Sampled as u32,
        )?;
        let texture = VulkanTexture2D {
            inner: Arc::new(Mutex::new(TextureViewInner {
                width: desc.width,
                height: desc.height,
                depth: desc.depth,
                format: desc.format,
                is_depth: false,
                released: false,
            })),
            image,
            memory,
            view,
            released: false,
        };
        let handle = TextureView {
            inner: texture.inner.clone(),
        };
        self.textures_3d.push(texture);
        Ok(handle)
    }
    fn release_texture_3d(&mut self, view: &TextureView) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        match self
            .textures_3d
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &view.inner))
        {
            Some(index) => {
                if self.textures_3d[index].released {
                    self.violations += 1;
                } else {
                    self.textures_3d[index].released = true;
                    view.inner.lock().unwrap().released = true;
                }
            }
            None => {
                self.violations += 1;
            }
        }
    }

    fn fill_texture_3d(&mut self, view: &TextureView, data: &[u8])
        -> Result<(), RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "fill_texture_3d: the backend is shut down",
            ));
        }
        let index = match self
            .textures_3d
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &view.inner))
        {
            Some(index) => index,
            None => {
                self.violations += 1;
                return Err(RenderError::from(
                    "fill_texture_3d: the texture is not owned by this backend",
                ));
            }
        };
        let inner = view.inner.lock().unwrap();
        let expected = (inner.width as usize) * (inner.height as usize)
            * (inner.depth as usize) * bytes_per_pixel(inner.format) as usize;
        let (width, height, depth) = (inner.width, inner.height, inner.depth);
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "fill_texture_3d: the byte count is {} but the texture is {}",
                data.len(),
                expected
            )));
        }
        if self.readback_command.is_null() {
            return Err(RenderError::from(
                "fill_texture_3d: the command path is not initialised",
            ));
        }
        // The staging upload: the same one-off command buffer the 2D
        // fill follows (the transition Undefined -> TransferDst ->
        // the shader read).
        let image = self.textures_3d[index].image;
        let (staging, staging_memory) = self.create_host_buffer(
            data.len(),
            vk::BufferUsage::TransferSrc,
        )?;
        let map = self.map_memory(staging_memory)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                map as *mut u8,
                data.len(),
            );
        }
        unsafe {
            (self.loader.vk_reset_command_buffer.unwrap())(
                self.readback_command,
                0,
            );
        }
        let begin_info = vk::VkCommandBufferBeginInfo {
            s_type: vk::StructureType::CommandBufferBeginInfo,
            p_next: std::ptr::null(),
            flags: vk::CommandBufferUsage::OneTimeSubmit as u32,
            p_inheritance_info: std::ptr::null(),
        };
        let result = unsafe {
            (self.loader.vk_begin_command_buffer.unwrap())(
                self.readback_command,
                &begin_info,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "fill_texture_3d: the command buffer could not begin",
            ));
        }
        // The single full-depth copy: the buffer's packed layout packs
        // the depth slices back to back and the copy's extent carries
        // the whole depth (the driver walks the slices itself).
        let copies = vec![vk::VkBufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::VkImageSubresourceLayers {
                aspect_mask: vk::ImageAspect::Color as u32,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::VkOffset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::VkExtent3D {
                width,
                height,
                depth,
            },
        }];
        self.transition_image_layout(
            self.readback_command,
            image,
            vk::ImageAspect::Color as u32,
            vk::ImageLayout::Undefined,
            vk::ImageLayout::TransferDstOptimal,
        );
        unsafe {
            (self.loader.vk_cmd_copy_buffer_to_image.unwrap())(
                self.readback_command,
                staging,
                image,
                vk::ImageLayout::TransferDstOptimal,
                copies.len() as u32,
                copies.as_ptr(),
            );
        }
        self.transition_image_layout(
            self.readback_command,
            image,
            vk::ImageAspect::Color as u32,
            vk::ImageLayout::TransferDstOptimal,
            vk::ImageLayout::ShaderReadOnlyOptimal,
        );
        let result = unsafe {
            (self.loader.vk_end_command_buffer.unwrap())(
                self.readback_command,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "fill_texture_3d: the command buffer could not end",
            ));
        }
        let submit = vk::VkSubmitInfo {
            s_type: vk::StructureType::SubmitInfo,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            signal_semaphore_count: 0,
            p_command_buffers: &self.readback_command,
            command_buffer_count: 1,
            p_signal_semaphores: std::ptr::null(),
        };
        unsafe {
            (self.loader.vk_queue_submit.unwrap())(
                self.queue,
                1,
                &submit,
                std::ptr::null_mut(),
            );
            (self.loader.vk_queue_wait_idle.unwrap())(self.queue);
        }
        let _ = (width, height, depth);
        Ok(())
    }

    fn read_texture_3d(
        &mut self,
        view: &TextureView,
        data: &mut [u8],
    ) -> Result<bool, RenderError> {
        let index = match self
            .textures_3d
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &view.inner))
        {
            Some(index) => index,
            None => {
                return Err(RenderError::from(
                    "read_texture_3d: the texture is not owned by this backend",
                ));
            }
        };
        if self.textures_3d[index].released {
            return Ok(false);
        }
        let inner = view.inner.lock().unwrap();
        let expected = (inner.width as usize) * (inner.height as usize)
            * (inner.depth as usize) * bytes_per_pixel(inner.format) as usize;
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "read_texture_3d: the byte count is {} but the texture is {}",
                data.len(),
                expected
            )));
        }
        // The real readback: the same one-off command buffer the fill
        // carries — the transfer-src handoff, the copy into a host
        // buffer, the queue wait the fill's submit follows.
        let image = self.textures_3d[index].image;
        let (staging, staging_memory) = self.create_host_buffer(
            data.len(),
            vk::BufferUsage::TransferDst,
        )?;
        unsafe {
            (self.loader.vk_reset_command_buffer.unwrap())(
                self.readback_command,
                0,
            );
        }
        let begin_info = vk::VkCommandBufferBeginInfo {
            s_type: vk::StructureType::CommandBufferBeginInfo,
            p_next: std::ptr::null(),
            flags: vk::CommandBufferUsage::OneTimeSubmit as u32,
            p_inheritance_info: std::ptr::null(),
        };
        let result = unsafe {
            (self.loader.vk_begin_command_buffer.unwrap())(
                self.readback_command,
                &begin_info,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "read_texture_3d: the command buffer could not begin",
            ));
        }
        let (width, height, depth) = {
            let inner = view.inner.lock().unwrap();
            (inner.width, inner.height, inner.depth)
        };
        // DEBUG: one copy per slice with the z offsets (the decisive
        // probe with the single-copy upload isolates the readback side).
        let slice_bytes = (width * height * 4 * 2) as u32;
        let mut copies = Vec::with_capacity(depth as usize);
        for layer in 0..depth {
            copies.push(vk::VkBufferImageCopy {
                buffer_offset: (layer * slice_bytes) as u64,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::VkImageSubresourceLayers {
                    aspect_mask: vk::ImageAspect::Color as u32,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::VkOffset3D { x: 0, y: 0, z: layer as i32 },
                image_extent: vk::VkExtent3D {
                    width,
                    height,
                    depth: 1,
                },
            });
        }
        // The transfer-src handoff: the shader-read state the fill's
        // last transition left, into the copy's source state.
        self.transition_image_layout(
            self.readback_command,
            image,
            vk::ImageAspect::Color as u32,
            vk::ImageLayout::ShaderReadOnlyOptimal,
            vk::ImageLayout::TransferSrcOptimal,
        );
        unsafe {
            (self.loader.vk_cmd_copy_image_to_buffer.unwrap())(
                self.readback_command,
                image,
                vk::ImageLayout::TransferSrcOptimal,
                staging,
                copies.len() as u32,
                copies.as_ptr(),
            );
        }
        // The handback: the Undefined state the next fill's
        // Undefined -> TransferDst transition expects.
        self.transition_image_layout(
            self.readback_command,
            image,
            vk::ImageAspect::Color as u32,
            vk::ImageLayout::TransferSrcOptimal,
            vk::ImageLayout::Undefined,
        );
        let result = unsafe {
            (self.loader.vk_end_command_buffer.unwrap())(
                self.readback_command,
            )
        };
        if result != vk::Result::Success {
            return Err(RenderError::from(
                "read_texture_3d: the command buffer could not end",
            ));
        }
        let submit = vk::VkSubmitInfo {
            s_type: vk::StructureType::SubmitInfo,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            signal_semaphore_count: 0,
            p_command_buffers: &self.readback_command,
            command_buffer_count: 1,
            p_signal_semaphores: std::ptr::null(),
        };
        unsafe {
            (self.loader.vk_queue_submit.unwrap())(
                self.queue,
                1,
                &submit,
                std::ptr::null_mut(),
            );
            (self.loader.vk_queue_wait_idle.unwrap())(self.queue);
        }
        let map = self.map_memory(staging_memory)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                map as *const u8,
                data.as_mut_ptr(),
                data.len(),
            );
        }
        Ok(true)
    }

    fn set_light_volume(&mut self, volume: &LightVolume) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        if volume.dims[0] == 0 || volume.dims[1] == 0 || volume.dims[2] == 0
            || volume.cell_size <= 0.0
        {
            self.violations += 1;
            return;
        }
        // The volume's 3D texture: the same Rgba16F path the standalone
        // 3D textures carry (the dims z * 6 layers, the face `f` of the
        // cell `(x, y, z)` at the layer `z * 6 + f`).
        let desc = Texture3DDesc {
            width: volume.dims[0],
            height: volume.dims[1],
            depth: volume.dims[2] * 6,
            format: PixelFormat::Rgba16F,
        };
        let view = match self.create_texture_3d(&desc) {
            Ok(view) => view,
            Err(_) => {
                self.violations += 1;
                return;
            }
        };
        let packed = volume.pack_texture_3d();
        if self.fill_texture_3d(&view, &packed).is_err() {
            self.violations += 1;
            return;
        }
        // The uniform block's values: the origin, the grid's dims (the
        // depth carrying the z * 6 layers) and the cell size (the
        // indirect intensity and the toggle the submit writes).
        self.volume_uniform = VolumeUniform {
            origin: [
                volume.origin[0],
                volume.origin[1],
                volume.origin[2],
                0.0,
            ],
            dims: [
                volume.dims[0] as f32,
                volume.dims[1] as f32,
                (volume.dims[2] * 6) as f32,
                0.0,
            ],
            params: [volume.cell_size, 0.0, 0.0, 0.0],
        };
        // The old volume's texture releases (the one-volume rule the
        // contract's set_light_volume carries).
        if let Some(old) = self.light_volume.take() {
            let handle = TextureView {
                inner: old.inner.clone(),
            };
            self.release_texture_3d(&handle);
        }
        self.light_volume = Some(VulkanTexture2D {
            inner: view.inner,
            image: self.textures_3d.last().unwrap().image,
            memory: self.textures_3d.last().unwrap().memory,
            view: self.textures_3d.last().unwrap().view,
            released: false,
        });
    }

    fn set_camera(&mut self, camera: CameraParams) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        if let Err(_error) = camera.validate() {
            self.violations += 1;
            return;
        }
        self.camera = camera;
    }

    fn camera(&self) -> CameraParams {
        self.camera
    }

    fn swap_buffers(&mut self) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        // The frame's end: the render pass closes, the command buffer
        // ends and the queue submits (the same rule the OpenGL backend's
        // swap follows; the diagnostic triangle's old path submitted the
        // frame inside the clear — the generic draw path needs the pass
        // open across the contract's calls).
        self.submit_frame();
        // The readback happens before the present (the same pre-present
        // rule the OpenGL backend follows). The off-screen frame is not
        // presented: its command buffer already ran at the frame's end.
        self.readback();
    }

    fn queue_readback(&mut self, x: u32, y: u32) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.readback_queue.push((x, y));
    }

    fn readback(&mut self) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        // The readback path copies the centre pixel from the off-screen
        // colour attachment through the staging buffer (the host-visible
        // mapping the backend owns).
        self.last_readback.clear();
        let queue = std::mem::take(&mut self.readback_queue);
        if self.config.readback_center || !queue.is_empty() {
            self.readback_pixel(
                self.config.width / 2,
                self.config.height / 2,
            );
        }
        for (x, y) in queue {
            self.readback_pixel(x, y);
        }
    }

    fn last_readback(&self) -> &[ReadbackSample] {
        &self.last_readback
    }

    fn has_center_pixel(&self) -> bool {
        self.has_center
    }

    fn center_pixel(&self) -> ClearColor {
        self.center
    }

    fn center_depth(&self) -> f32 {
        self.center_depth
    }

    fn center_stencil(&self) -> i32 {
        self.center_stencil
    }
}

/// The platform's Vulkan library: the loader's dlopen() surface. The
/// backend resolves the entry points through the library's
/// vkGetInstanceProcAddr, so the runtime link-time dependency is only the
/// platform's libvulkan.so.1 (the same rule the OpenGL backend follows).
///
/// On Windows the same surface is LoadLibraryW over vulkan-1.dll (the
/// platform's own loading primitives, the same rule): the functions are
/// the kernel32 imports the platform itself provides, gated behind
/// cfg(windows) with the non-Windows stubs keeping the module linkable.
unsafe fn load_library() -> Result<*mut std::os::raw::c_void, RenderError> {
    #[cfg(windows)]
    {
        // The UTF-16 "vulkan-1.dll" literal (the loader's canonical
        // Windows name, the ICD loader's own export).
        let name: [u16; 13] = [
            0x0076, 0x0075, 0x006c, 0x006b, 0x0061, 0x006e, 0x002d,
            0x0031, 0x002e, 0x0064, 0x006c, 0x006c, 0x0000,
        ];
        let handle = LoadLibraryW(name.as_ptr());
        if handle.is_null() {
            return Err(RenderError::from(
                "VulkanBackend: the platform's vulkan-1.dll could not be \
                 loaded",
            ));
        }
        return Ok(handle);
    }
    #[cfg(not(windows))]
    {
        let names = [
            b"libvulkan.so.1\0".as_ref(),
            b"libvulkan.so\0".as_ref(),
        ];
        for name in &names {
            let handle = libc::dlopen(name.as_ptr() as *const _,
                                      libc::RTLD_NOW | libc::RTLD_LOCAL);
            if !handle.is_null() {
                return Ok(handle);
            }
        }
        Err(RenderError::from(
            "VulkanBackend: the platform's libvulkan.so.1 could not be \
             loaded",
        ))
    }
}

// The kernel32 imports the Windows path needs (the platform's own
// loading primitives). The dlsym shape carries over: on Windows the
// GetProcAddress replaces dlsym.
#[cfg(windows)]
extern "system" {
    fn LoadLibraryW(
        name: *const u16,
    ) -> *mut std::os::raw::c_void;
    // The name parameter is the C char (the same signature the D3D12
    // loader's kernel32 extern carries: the two declarations share the
    // one symbol, the redeclaration mismatch is the linker's clash the
    // combined-feature builds warn about).
    fn GetProcAddress(
        module: *mut std::os::raw::c_void,
        name: *const std::os::raw::c_char,
    ) -> *mut std::os::raw::c_void;
}

/// One library symbol by name. On Windows the GetProcAddress is the
/// dlsym's counterpart; on the other platforms the dlsym stays.
unsafe fn get_library_proc(
    library: *mut std::os::raw::c_void,
    name: &str,
) -> *mut std::os::raw::c_void {
    let mut bytes = Vec::with_capacity(name.len() + 1);
    bytes.extend_from_slice(name.as_bytes());
    bytes.push(0);
    #[cfg(windows)]
    {
        GetProcAddress(library, bytes.as_ptr() as *const std::os::raw::c_char)
    }
    #[cfg(not(windows))]
    {
        libc::dlsym(library, bytes.as_ptr() as *const _)
    }
}

// The non-Windows stubs for the kernel32 imports keep the module
// linkable where the backend never reaches the Windows path (the
// backend reports its load error before any call).
// The reserved names are the platform's own entry-point spellings; the
// allow keeps the never-called stubs quiet (the cfg pair makes each
// half compile on exactly one platform).
#[cfg(not(windows))]
#[allow(non_snake_case, dead_code)]
unsafe extern "system" fn LoadLibraryW(
    _name: *const u16,
) -> *mut std::os::raw::c_void {
    std::ptr::null_mut()
}
#[cfg(not(windows))]
#[allow(non_snake_case, dead_code)]
unsafe extern "system" fn GetProcAddress(
    _module: *mut std::os::raw::c_void,
    _name: *const std::os::raw::c_char,
) -> *mut std::os::raw::c_void {
    std::ptr::null_mut()
}
