//! The Metal render backend: the render contract's third live
//! implementation, on the platform's Metal framework (macOS/iOS).
//!
//! RULE (lesson 3): the engine's own loader, own code, no Apple SDK
//! headers and no third-party binding. The `MtlLoader` resolves the
//! Objective-C runtime's C entry points (objc_getClass, sel_registerName,
//! objc_msgSend) and the Metal framework's one published C function
//! (MTLCreateSystemDefaultDevice) at runtime; every Metal class handle is
//! an opaque pointer and every call a runtime message send.
//!
//! The backend is headless, the same rule the Vulkan and the OpenGL
//! backends follow: the frame target is an off-screen texture (the colour
//! attachment plus the depth/stencil attachment), no windowing system and
//! no drawable. The frame flow matches the other backends: `clear()` opens
//! the frame (the render pass's load actions carry the clear values), the
//! diagnostic triangle draws into the pass, and `swap_buffers()` runs the
//! readback before the frame ends. The readback copies the centre pixel
//! through a blit command encoder into a staging buffer with the six-word
//! layout the Vulkan readback validated (colour at offset 0, depth at
//! offset 16, stencil at offset 20).
//!
//! Validation scope: the module compiles on any platform (the loader
//! declares its own types, so no Apple SDK is needed) and the hardware
//! run needs a Mac — the M20 recorded scope; nothing here claims
//! otherwise. The loader's failure path is the one behaviour a
//! non-Apple host can exercise: `new()` reports the missing framework in
//! the `RenderError` message.

use super::mtl_loader::{
    MtlClearColor, MtlLoader, MtlOrigin, MtlSize,
    MtlRegion,
    MTLPX_RGBA8_UNORM, MTLPX_RGBA16_FLOAT, MTLPX_R8_UNORM,
    MTLPX_DEPTH24_UNORM_STENCIL8, MTLPX_DEPTH32_FLOAT_STENCIL8,
    MTLLA_CLEAR, MTLSA_STORE,
    MTLPT_TRIANGLE,
    MTLVF_FLOAT2, MTLVF_FLOAT3, MTLSF_PER_VERTEX, MTLSM_SHARED, MTLSM_PRIVATE,
    MTLSMF_NEAREST, MTLSAM_CLAMP_TO_EDGE, MTLLA_LOAD,
    MTLIT_UINT32, MTLCF_LESS,
    MTLBLOPT_DEPTH_FROM_DEPTH_STENCIL, MTLBLOPT_STENCIL_FROM_DEPTH_STENCIL,
    MTLCU_RENDER_TARGET, MTLCU_SHADER_READ,
};
use super::super::{
    bytes_per_pixel, CameraParams, ClearBuffer, ClearColor,
    DirectionalLight, DrawSubmission, DynamicLight, GpuMesh, GpuMeshInner,
    PixelFormat, PostChainConfig, ReadbackSample, RenderBackend,
    RenderError, RenderTarget,
    RenderTargetDesc, RenderTargetInner, ShaderProgram, ShaderProgramInner,
    Texture2DDesc, Texture3DDesc, TextureView, TextureViewInner, VertexLayout,
    DEPTH_ATTACHMENT,
};
use crate::math::{Mat4, Vec3};
use crate::lightvolume::LightVolume;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

/// The diagnostic triangle's uniform block (the same bytes the Vulkan and
/// the OpenGL blocks carry: the MVP column-major followed by the RGBA
/// colour).
#[repr(C)]
struct TriUniform {
    mvp: [[f32; 4]; 4],
    color: [f32; 4],
}

/// The generic mesh's uniform block (the same two column-major matrices
/// the Vulkan and the D3D12 blocks carry: the view-projection the
/// set_camera state builds and the model matrix the submission writes).
#[repr(C)]
struct MeshUniform {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

/// The lights' uniform block (M5): the same layout every backend maps —
/// the sun's direction (xyz + pad), the sun's colour (xyz) and intensity
/// (w), the four point lights' position (xyz) and radius (w), the four
/// point lights' colour (xyz) and intensity (w), the params (x = the
/// live point-light count). The tail pad completes the 256 bytes the
/// other backends' uniform buffers carry (the copies' 256-byte writes
/// stay in bounds).
#[repr(C)]
#[derive(Clone, Copy)]
struct LightsUniform {
    sun_direction: [f32; 4],
    sun_color_intensity: [f32; 4],
    point_position_radius: [[f32; 4]; 4],
    point_color_intensity: [[f32; 4]; 4],
    params: [f32; 4],
    _pad: [f32; 20],
}

impl LightsUniform {
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
            // first lit step carries the point lights).
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
/// The backend-owned state of one uploaded generic mesh: the vertex and
/// index buffers (shared storage, the CPU writes them directly), the
/// per-mesh uniform buffer and the counts.
#[allow(dead_code)]
struct MetalMeshState {
    stride: u32,
    vertex_buffer: super::mtl_loader::MtlBufferHandle,
    index_buffer: super::mtl_loader::MtlBufferHandle,
    uniform_buffer: super::mtl_loader::MtlBufferHandle,
    vertex_count: u32,
    index_count: u32,
}

/// The mesh path's pipeline resources (the MSL library pair the mesh
/// shaders compile to, the depth/stencil state and the sampler). The
/// `ensure_mesh_resources` creates them lazily on the first upload.
#[allow(dead_code)]
struct MetalMeshResources {
    pipeline: super::mtl_loader::MtlRenderPipelineStateHandle,
    depth_state: super::mtl_loader::MtlDepthStencilStateHandle,
    sampler: super::mtl_loader::MtlSamplerStateHandle,
}

/// The Metal backend's configuration. The same fields the other backends'
/// configs carry.
#[derive(Clone, Debug)]
pub struct MetalBackendConfig {
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

impl Default for MetalBackendConfig {
    fn default() -> Self {
        MetalBackendConfig {
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

/// The Metal backend's own texture: a device texture plus the contract's
/// shared handle state.
// The texture handle is written at creation; the release path (the next
// milestones) releases it.
#[allow(dead_code)]
struct MetalTexture2D {
    inner: Arc<Mutex<TextureViewInner>>,
    texture: super::mtl_loader::MtlTextureHandle,
    released: bool,
}

/// The Metal backend's own render target: the colour attachments plus the
/// depth/stencil attachment, each a device texture.
// The texture handles are written at creation; the release path (the next
// milestones) releases them.
#[allow(dead_code)]
struct MetalRenderTarget {
    inner: Arc<Mutex<RenderTargetInner>>,
    views: Vec<MetalTexture2D>,
    depth_view: Option<MetalTexture2D>,
    released: bool,
}

/// The headless Metal render backend.
// The device/queue handles and the diagnostic-triangle bookkeeping fields
// are written at setup; the release path (the next milestones) reads them.
#[allow(dead_code)]
pub struct MetalBackend {
    config: MetalBackendConfig,
    loader: MtlLoader,
    device: super::mtl_loader::MtlDeviceHandle,
    queue: super::mtl_loader::MtlCommandQueueHandle,
    // The device's identity (the describe() reports it).
    device_name: String,
    // The off-screen frame target's colour and depth/stencil textures.
    color_texture: super::mtl_loader::MtlTextureHandle,
    depth_texture: super::mtl_loader::MtlTextureHandle,
    // The depth format the device picked (24/8 where the device answers
    // yes, 32F/8 elsewhere).
    depth_format: super::mtl_loader::NsUInteger,
    // The diagnostic triangle's pipeline and its uniform buffer.
    tri_pipeline: super::mtl_loader::MtlRenderPipelineStateHandle,
    tri_uniform_buffer: super::mtl_loader::MtlBufferHandle,
    tri_ready: bool,
    /// The first failure's stage in the triangle resources' setup, with
    /// the API's own error text where one exists (the hardware run's
    /// diagnostic: the contract's render-path failures are silent
    /// unless they surface here).
    tri_error: String,
    // The frame's open command buffer (the clear pass's record; the
    // swap's commit closes the frame the way the Vulkan submit_frame
    // carries).
    frame_command_buffer: super::mtl_loader::MtlCommandBufferHandle,
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
    targets: Vec<MetalRenderTarget>,
    textures_2d: Vec<MetalTexture2D>,
    textures_3d: Vec<MetalTexture2D>,
    next_mesh_id: usize,
    /// The generic mesh path's state: one slot per create_gpu_mesh, the
    /// mesh pipeline's resources and the default texture the draws fall
    /// back to (the same structure the Vulkan and the OpenGL paths
    /// carry).
    meshes: Vec<Option<MetalMeshState>>,
    mesh_resources: Option<MetalMeshResources>,
    mesh_ready: bool,
    mesh_error: String,
    // The lit mesh path's state (M5): the lit pipeline's resources (the
    // same depth/stencil state and sampler the unlit path shares) and
    // the lights' shared buffer (one 256-byte block the draws write and
    // the fragment stage's [[buffer(1)]] reads).
    lit_resources: Option<MetalMeshResources>,
    lit_ready: bool,
    lit_error: String,
    lit_lights_buffer: *mut c_void,
    lit_block: LightsUniform,
    // The deferred path's state (M6): the two pipelines (the G-buffer's
    // three-attachment pipeline and the lighting pass's single-attachment
    // one) and the lights' shared buffer the lighting pass writes. The
    // depth/stencil state and the sampler are the mesh path's own (the
    // deferred pipelines share them).
    gbuffer_pipeline: super::mtl_loader::MtlRenderPipelineStateHandle,
    lighting_pipeline: super::mtl_loader::MtlRenderPipelineStateHandle,
    lighting_lights_buffer: *mut c_void,
    deferred_ready: bool,
    deferred_error: String,
    // The post-processing chain's state (M7): the HDR geometry pipeline
    // (the two Rgba16F attachments) and the composite pipeline (the
    // fullscreen triangle over the default framebuffer) plus the lights'
    // and the config's shared buffers the passes write. The depth/stencil
    // state and the sampler are the mesh path's own (the post pipelines
    // share them).
    post_gbuffer_pipeline: super::mtl_loader::MtlRenderPipelineStateHandle,
    post_composite_pipeline: super::mtl_loader::MtlRenderPipelineStateHandle,
    post_lights_buffer: *mut c_void,
    post_config_buffer: *mut c_void,
    post_volume_buffer: *mut c_void,
    light_volume: Option<MetalTexture2D>,
    volume_uniform: [f32; 12],
    post_ready: bool,
    post_error: String,
    /// The default texture's index into `textures_2d` (the 1x1 white
    /// texture the setup created; the draws fall back to it when no
    /// texture is bound) and the bound texture's index (the
    /// bind_texture state).
    default_texture: Option<usize>,
    bound_texture: Option<usize>,
    readback_queue: Vec<(u32, u32)>,
    last_readback: Vec<ReadbackSample>,
    has_center: bool,
    center: ClearColor,
    center_depth: f32,
    center_stencil: i32,
}

impl MetalBackend {
    /// Creates a backend instance. Throws `RenderError` when the platform
    /// has no Metal framework, no default device can be created, or the
    /// frame textures cannot be allocated.
    pub fn new(config: MetalBackendConfig) -> Result<Self, RenderError> {
        let clear_depth = config.clear_depth;
        let clear_stencil = config.clear_stencil;
        if config.width == 0 || config.height == 0 {
            return Err(RenderError::from(
                "MetalBackend: width and height must be positive",
            ));
        }
        let mut loader = MtlLoader::new();
        let mut missing = String::new();
        if !loader.load(&mut missing) {
            return Err(RenderError::from(format!(
                "MetalBackend: the platform's Metal framework could not be \
                 loaded ({})",
                missing
            )));
        }
        let device = unsafe {
            loader
                .mtl_create_system_default_device
                .expect("the loader resolved the device factory")()
        };
        if device.is_null() {
            return Err(RenderError::from(
                "MetalBackend: the platform has no default Metal device",
            ));
        }
        let device_name = unsafe { loader.device_name(device) };
        let queue =
            unsafe { loader.send0(device, loader.sel("newCommandQueue")) };
        if queue.is_null() {
            return Err(RenderError::from(
                "MetalBackend: the device's command queue could not be \
                 created",
            ));
        }
        let mut backend = MetalBackend {
            config,
            loader,
            device,
            queue,
            device_name,
            color_texture: std::ptr::null_mut(),
            depth_texture: std::ptr::null_mut(),
            depth_format: MTLPX_DEPTH24_UNORM_STENCIL8,
            tri_pipeline: std::ptr::null_mut(),
            tri_uniform_buffer: std::ptr::null_mut(),
            tri_ready: false,
            tri_error: String::new(),
            frame_command_buffer: std::ptr::null_mut(),
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
            next_mesh_id: 0,
            meshes: Vec::new(),
            mesh_resources: None,
            mesh_ready: false,
            mesh_error: String::new(),
            lit_resources: None,
            lit_ready: false,
            lit_error: String::new(),
            lit_lights_buffer: std::ptr::null_mut(),
            lit_block: LightsUniform::none(),
            gbuffer_pipeline: std::ptr::null_mut(),
            lighting_pipeline: std::ptr::null_mut(),
            lighting_lights_buffer: std::ptr::null_mut(),
            deferred_ready: false,
            deferred_error: String::new(),
            post_gbuffer_pipeline: std::ptr::null_mut(),
            post_composite_pipeline: std::ptr::null_mut(),
            post_lights_buffer: std::ptr::null_mut(),
            post_config_buffer: std::ptr::null_mut(),
            post_volume_buffer: std::ptr::null_mut(),
            light_volume: None,
            volume_uniform: [0.0; 12],
            post_ready: false,
            post_error: String::new(),
            default_texture: None,
            bound_texture: None,
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

    /// The first failure's stage in the triangle resources' setup (an
    /// empty string means the setup succeeded; the diagnostic the
    /// smoke examples surface before the frame's checks).
    pub fn triangle_error(&self) -> &str {
        &self.tri_error
    }

    /// The off-screen frame target's textures: the colour attachment (RGBA8
    /// unorm) and the depth/stencil attachment (the format the device
    /// picks). The textures are private storage: the CPU never maps them;
    /// the readback blit copies into a shared staging buffer.
    fn initialise(&mut self) -> Result<(), RenderError> {
        // The depth format pick: the 24-bit packed depth/stencil format is
        // macOS only; where the device does not answer yes, the universal
        // 32-bit float format carries the attachment.
        let depth24_ok = unsafe {
            self.loader.send_ret_bool(
                self.device,
                self.loader.sel("isDepth24Stencil8PixelFormatSupported"),
            )
        };
        self.depth_format = if depth24_ok {
            MTLPX_DEPTH24_UNORM_STENCIL8
        } else {
            MTLPX_DEPTH32_FLOAT_STENCIL8
        };
        self.color_texture = self.create_device_texture(
            self.config.width,
            self.config.height,
            MTLPX_RGBA8_UNORM,
        )?;
        self.depth_texture = self.create_device_texture(
            self.config.width,
            self.config.height,
            self.depth_format,
        )?;
        // The default texture: one 1x1 opaque white texel the mesh path's
        // draws fall back to when no texture is bound (the same default
        // the Vulkan and the OpenGL paths record).
        let view = self.create_texture_2d(&Texture2DDesc {
            width: 1,
            height: 1,
            format: PixelFormat::Rgba8,
        })?;
        self.fill_texture_2d(&view, &[255, 255, 255, 255])?;
        let index = self
            .textures_2d
            .iter()
            .position(|t| Arc::ptr_eq(&t.inner, &view.inner))
            .expect("the default texture's slot");
        self.default_texture = Some(index);
        Ok(())
    }

    /// Allocates one device texture: a descriptor object carries the size,
    /// the format and the usage; the device's newTextureWithDescriptor:
    /// call allocates it.
    fn create_device_texture(
        &mut self,
        width: u32,
        height: u32,
        format: super::mtl_loader::NsUInteger,
    ) -> Result<super::mtl_loader::MtlTextureHandle, RenderError> {
        let descriptor = unsafe {
            self.loader.alloc_init(self.loader.cls_mtl_texture_descriptor)
        };
        if descriptor.is_null() {
            return Err(RenderError::from(
                "MetalBackend: the texture descriptor could not be created",
            ));
        }
        // The 2D type never reaches the descriptor: the texture type
        // property's own default is the 2D type (the M8 rule: the depth
        // and the type stay at the defaults, and the depth/stencil
        // attachments' 2D requirement is the descriptor's own).
        unsafe {
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setWidth:"),
                width as super::mtl_loader::NsUInteger,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setHeight:"),
                height as super::mtl_loader::NsUInteger,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setPixelFormat:"),
                format,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setStorageMode:"),
                MTLSM_PRIVATE,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setUsage:"),
                MTLCU_RENDER_TARGET | MTLCU_SHADER_READ,
            );
        }
        let texture = unsafe {
            self.loader.send_ptr_arg(
                self.device,
                self.loader.sel("newTextureWithDescriptor:"),
                descriptor,
            )
        };
        if texture.is_null() {
            return Err(RenderError::from(
                "MetalBackend: the device texture could not be allocated",
            ));
        }
        Ok(texture)
    }

    /// The 3D texture's creation path: the same descriptor the 2D path
    /// builds, with the texture type set to the 3D type and the depth
    /// carrying the volume's layers (the usage is the sampled-read one
    /// the fill path's replaceRegion writes through; the 3D texture is
    /// never a render target).
    fn create_device_texture_3d(
        &mut self,
        width: u32,
        height: u32,
        depth: u32,
        format: super::mtl_loader::NsUInteger,
    ) -> Result<super::mtl_loader::MtlTextureHandle, RenderError> {
        let descriptor = unsafe {
            self.loader.alloc_init(self.loader.cls_mtl_texture_descriptor)
        };
        if descriptor.is_null() {
            return Err(RenderError::from(
                "MetalBackend: the texture descriptor could not be created",
            ));
        }
        unsafe {
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setWidth:"),
                width as super::mtl_loader::NsUInteger,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setHeight:"),
                height as super::mtl_loader::NsUInteger,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setDepth:"),
                depth as super::mtl_loader::NsUInteger,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setTextureType:"),
                super::mtl_loader::MTLT_3D,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setPixelFormat:"),
                format,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setStorageMode:"),
                MTLSM_PRIVATE,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setUsage:"),
                MTLCU_SHADER_READ,
            );
        }
        let texture = unsafe {
            self.loader.send_ptr_arg(
                self.device,
                self.loader.sel("newTextureWithDescriptor:"),
                descriptor,
            )
        };
        if texture.is_null() {
            return Err(RenderError::from(
                "MetalBackend: the device texture could not be allocated",
            ));
        }
        Ok(texture)
    }

    /// The contract's pixel format mapping (the contract's enum onto the
    /// Metal pixel format).
    fn to_mtl_format(&self, format: PixelFormat) -> super::mtl_loader::NsUInteger {
        match format {
            PixelFormat::Rgba8 => MTLPX_RGBA8_UNORM,
            PixelFormat::Rgba16F => MTLPX_RGBA16_FLOAT,
            PixelFormat::R8 => MTLPX_R8_UNORM,
            PixelFormat::Depth24Stencil8 => self.depth_format,
        }
    }

    /// Compiles the two MSL sources into one MTLLibrary (the device's own
    /// compiler; a nil return is the compile failure's signal).
    fn compile_stage(
        &self,
        source: &str,
        stage: &'static str,
    ) -> Result<super::mtl_loader::MTLLibraryHandle, String> {
        // The compile failure's text travels back to the caller as the
        // Err value (the lit path's failures must land in `lit_error`,
        // not the triangle path's `tri_error` — the upload's error
        // message reads the lit one, and an empty message hides the
        // compile failure's text).
        let source_string = self.make_ns_string(source);
        if source_string.is_null() {
            return Err(format!("{}: the source's NSString failed", stage));
        }
        let options = unsafe {
            self.loader
                .alloc_init(self.loader.cls_mtl_compile_options)
        };
        // The device's real library-creation method is
        // newLibraryWithSource:options:error: (the compile-error out-slot
        // is the NSError** the caller passes by pointer; a nil return is
        // the compile failure's signal). The earlier compileSource:
        // options: selector does not exist in the API, so the runtime
        // raised the NSInvalidArgumentException the Rust runtime cannot
        // catch — the hardware run's crash right after the describe:.
        let mut compile_error: *mut c_void = std::ptr::null_mut();
        let library = unsafe {
            self.loader.send_ptr_arg3(
                self.device,
                self.loader.sel("newLibraryWithSource:options:error:"),
                source_string,
                options,
                &mut compile_error as *mut *mut c_void as *mut c_void,
            )
        };
        if library.is_null() {
            Err(if compile_error.is_null() {
                format!("{}: the library creation returned nil", stage)
            } else {
                format!("{}: {}", stage, self.ns_error_text(compile_error))
            })
        } else {
            Ok(library)
        }
    }

    /// Reads an NSError's message (the localizedDescription property's
    /// NSString through UTF8String, the same read the device name uses).
    fn ns_error_text(&self, error: *mut c_void) -> String {
        let description = unsafe {
            self.loader.send0(error, self.loader.sel("localizedDescription"))
        };
        if description.is_null() {
            return "an unnamed error".to_string();
        }
        let text = unsafe {
            self.loader
                .send_ret_ptr(description, self.loader.sel("UTF8String"))
        };
        if text.is_null() {
            return "an unnamed error".to_string();
        }
        unsafe {
            std::ffi::CStr::from_ptr(text as *const std::os::raw::c_char)
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Creates an NSString from a Rust string slice (the runtime's
    /// stringWithUTF8String class method).
    fn make_ns_string(&self, text: &str) -> *mut c_void {
        let cls = unsafe {
            self.loader.objc_get_class.unwrap()(
                b"NSString\0".as_ptr() as *const std::os::raw::c_char,
            )
        };
        if cls.is_null() {
            return std::ptr::null_mut();
        }
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        unsafe {
            self.loader.send_ptr_arg(
                cls,
                self.loader.sel("stringWithUTF8String:"),
                bytes.as_ptr() as *mut c_void,
            )
        }
    }

    /// Fetches a named function out of a compiled library.
    fn new_function(
        &self,
        library: super::mtl_loader::MTLLibraryHandle,
        name: &str,
    ) -> Option<super::mtl_loader::MtlFunctionHandle> {
        let name_string = self.make_ns_string(name);
        if name_string.is_null() {
            return None;
        }
        let function = unsafe {
            self.loader.send_ptr_arg(
                library,
                self.loader.sel("newFunctionWithName:"),
                name_string,
            )
        };
        if function.is_null() {
            None
        } else {
            Some(function)
        }
    }

    /// Builds the triangle's uniform block bytes (the MVP column-major
    /// followed by the colour).
    fn build_tri_uniform(&self) -> Vec<u8> {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let projection = Mat4::perspective(
            std::f32::consts::FRAC_PI_2,
            aspect,
            0.1,
            100.0,
        );
        let view = Mat4::look_at(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let mvp = projection.multiply(view);
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
        unsafe {
            std::slice::from_raw_parts(
                &uniform as *const TriUniform as *const u8,
                std::mem::size_of::<TriUniform>(),
            )
            .to_vec()
        }
    }

    /// Builds the mesh's uniform block bytes (the view-projection matrix
    /// the set_camera state builds and the model matrix the submission
    /// writes — the same two matrices the Vulkan and the D3D12 blocks
    /// carry).
    fn build_mesh_uniform(&self, model: Mat4) -> Vec<u8> {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let camera = self.camera;
        let view = Mat4::look_at(camera.eye, camera.target, camera.up);
        let projection = Mat4::perspective(
            camera.fov_y_rad,
            aspect,
            camera.near,
            camera.far,
        );
        let view_projection = projection.multiply(view);
        let uniform = MeshUniform {
            view_projection: view_projection.m,
            model: model.m,
        };
        unsafe {
            std::slice::from_raw_parts(
                &uniform as *const MeshUniform as *const u8,
                std::mem::size_of::<MeshUniform>(),
            )
            .to_vec()
        }
    }

    /// Allocates a shared-storage buffer with the given bytes (the CPU
    /// writes, the GPU reads: Metal's shared storage is GPU-visible, no
    /// staging copy the Vulkan path needs exists here).
    fn create_shared_buffer(&self, bytes: &[u8]) -> super::mtl_loader::MtlBufferHandle {
        let buffer = unsafe {
            self.loader.send_ul2_ret_ptr(
                self.device,
                self.loader.sel("newBufferWithLength:options:"),
                bytes.len() as super::mtl_loader::NsUInteger,
                MTLSM_SHARED,
            )
        };
        if buffer.is_null() {
            return std::ptr::null_mut();
        }
        let contents = unsafe {
            self.loader.send0(buffer, self.loader.sel("contents"))
        };
        if !contents.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    contents as *mut u8,
                    bytes.len(),
                );
            }
        }
        buffer
    }

    /// The mesh path's sampler state: one NEAREST sampler, clamp-to-edge
    /// on both axes (the same filter the Vulkan and the D3D12 paths
    /// record).
    fn create_mesh_sampler(&self) -> super::mtl_loader::MtlSamplerStateHandle {
        let descriptor = unsafe {
            self.loader.alloc_init(self.loader.cls_mtl_sampler_descriptor)
        };
        if descriptor.is_null() {
            return std::ptr::null_mut();
        }
        let sampler = unsafe {
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setMinFilter:"),
                MTLSMF_NEAREST,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setMagFilter:"),
                MTLSMF_NEAREST,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setSAddressMode:"),
                MTLSAM_CLAMP_TO_EDGE,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setTAddressMode:"),
                MTLSAM_CLAMP_TO_EDGE,
            );
            self.loader.send_ptr_arg(
                self.device,
                self.loader.sel("newSamplerStateWithDescriptor:"),
                descriptor,
            )
        };
        if sampler.is_null() {
            return std::ptr::null_mut();
        }
        sampler
    }

    /// The mesh path's depth/stencil state: the depth test on (LESS, the
    /// same rule the Vulkan and the D3D12 paths record), the depth write
    /// on, the stencil test off.
    fn create_mesh_depth_state(&self) -> super::mtl_loader::MtlDepthStencilStateHandle {
        let descriptor = unsafe {
            self.loader.alloc_init(
                self.loader.cls_mtl_depth_stencil_descriptor,
            )
        };
        if descriptor.is_null() {
            return std::ptr::null_mut();
        }
        let state = unsafe {
            self.loader.send_void_bool(
                descriptor,
                self.loader.sel("setDepthWriteEnabled:"),
                true,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setDepthCompareFunction:"),
                MTLCF_LESS,
            );
            self.loader.send_ptr_arg(
                self.device,
                self.loader.sel("newDepthStencilStateWithDescriptor:"),
                descriptor,
            )
        };
        if state.is_null() {
            return std::ptr::null_mut();
        }
        state
    }

    /// The mesh path's pipeline: the two MSL libraries compile (the
    /// Camera struct both stages declare), the vertex descriptor carries
    /// the PNU layout's three attributes (position 0, normal 12, uv 24)
    /// at buffer index 1 (the uniform sits at buffer index 0), the
    /// colour attachment's pixel format follows the frame target's.
    fn ensure_mesh_resources(&mut self) -> bool {
        if self.mesh_ready {
            return true;
        }
        let vertex_library = match self.compile_stage(
            super::shader_library::MESH_VERTEX_SHADER,
            "the mesh vertex stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.mesh_error = message;
                return false;
            }
        };
        let vertex_function =
            match self.new_function(vertex_library, "meshVertexMain") {
                Some(function) => function,
                None => {
                    self.mesh_error =
                        "meshVertexMain lookup failed".to_string();
                    return false;
                }
            };
        let fragment_library = match self.compile_stage(
            super::shader_library::MESH_FRAGMENT_SHADER,
            "the mesh fragment stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.mesh_error = message;
                return false;
            }
        };
        let fragment_function =
            match self.new_function(fragment_library, "meshFragmentMain") {
                Some(function) => function,
                None => {
                    self.mesh_error =
                        "meshFragmentMain lookup failed".to_string();
                    return false;
                }
            };
        let descriptor = unsafe {
            self.loader
                .alloc_init(self.loader.cls_mtl_render_pipeline_descriptor)
        };
        if descriptor.is_null() {
            self.mesh_error =
                "the mesh pipeline descriptor alloc failed".to_string();
            return false;
        }
        unsafe {
            self.loader.send_void_ptr(
                descriptor,
                self.loader.sel("setVertexFunction:"),
                vertex_function,
            );
            self.loader.send_void_ptr(
                descriptor,
                self.loader.sel("setFragmentFunction:"),
                fragment_function,
            );
        }
        // The vertex descriptor: the class method vertexDescriptor (the
        // same rule the triangle path records: the alloc/init pair
        // leaves the arrays unallocated).
        let vertex_descriptor = unsafe {
            self.loader.send0(
                self.loader.cls_mtl_vertex_descriptor,
                self.loader.sel("vertexDescriptor"),
            )
        };
        if vertex_descriptor.is_null() {
            self.mesh_error =
                "the mesh vertex descriptor's constructor failed"
                    .to_string();
            return false;
        }
        let attributes = unsafe {
            self.loader.send0(
                vertex_descriptor,
                self.loader.sel("attributes"),
            )
        };
        let layouts = unsafe {
            self.loader.send0(vertex_descriptor, self.loader.sel("layouts"))
        };
        unsafe {
            // The three attributes: position (Float3, offset 0), normal
            // (Float3, offset 12), uv (Float2, offset 24), all at buffer
            // index 1 (the vertices' buffer; the uniform sits at buffer
            // index 0).
            for (index, (format, offset)) in [
                (MTLVF_FLOAT3, 0usize),
                (MTLVF_FLOAT3, 12usize),
                (MTLVF_FLOAT2, 24usize),
            ]
            .iter()
            .enumerate()
            {
                let attribute = self.loader.send_ul_ret_ptr(
                    attributes,
                    self.loader.sel("objectAtIndexedSubscript:"),
                    index as super::mtl_loader::NsUInteger,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setFormat:"),
                    *format,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setBufferIndex:"),
                    1,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setOffset:"),
                    *offset as super::mtl_loader::NsUInteger,
                );
            }
            // The layout at buffer index 1: the PNU stride (32), per
            // vertex.
            let layout_1 = self.loader.send_ul_ret_ptr(
                layouts,
                self.loader.sel("objectAtIndexedSubscript:"),
                1,
            );
            self.loader.send_void_ul(
                layout_1,
                self.loader.sel("setStride:"),
                32,
            );
            self.loader.send_void_ul(
                layout_1,
                self.loader.sel("setStepFunction:"),
                MTLSF_PER_VERTEX,
            );
            self.loader.send_void_ptr(
                descriptor,
                self.loader.sel("setVertexDescriptor:"),
                vertex_descriptor,
            );
            // The colour attachment's pixel format follows the frame
            // target's RGBA8 unorm, the depth and the stencil attachment
            // formats follow the device's pick (the same rule the
            // triangle pipeline records).
            let colour_attachments = self.loader.send0(
                descriptor,
                self.loader.sel("colorAttachments"),
            );
            let colour_attachment_0 = self.loader.send_ul_ret_ptr(
                colour_attachments,
                self.loader.sel("objectAtIndexedSubscript:"),
                0,
            );
            self.loader.send_void_ul(
                colour_attachment_0,
                self.loader.sel("setPixelFormat:"),
                MTLPX_RGBA8_UNORM,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setDepthAttachmentPixelFormat:"),
                self.depth_format,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setStencilAttachmentPixelFormat:"),
                self.depth_format,
            );
            // The winding rule and the cull mode stay at the API's own
            // defaults (front-facing = counter-clockwise, cull = none —
            // the same forgiving rule the Vulkan path's first step
            // records). The pipeline descriptor carries no winding/cull
            // setters at all: those are render-command-encoder state, and
            // sending setWinding:/setCullMode: to the descriptor raises
            // the unrecognized-selector exception the Rust runtime cannot
            // catch (the hardware run's abort at the descriptor block
            // caught it).
        }
        let pipeline = unsafe {
            let mut pipeline_error: *mut c_void = std::ptr::null_mut();
            let pipeline = self.loader.send_ptr_arg2(
                self.device,
                self.loader
                    .sel("newRenderPipelineStateWithDescriptor:error:"),
                descriptor,
                &mut pipeline_error as *mut *mut c_void as *mut c_void,
            );
            if pipeline.is_null() {
                self.mesh_error = if pipeline_error.is_null() {
                    "the mesh pipeline state creation returned nil"
                        .to_string()
                } else {
                    self.ns_error_text(pipeline_error)
                };
                return false;
            }
            pipeline
        };
        let depth_state = self.create_mesh_depth_state();
        if depth_state.is_null() {
            self.mesh_error =
                "the mesh depth/stencil state creation failed".to_string();
            return false;
        }
        let sampler = self.create_mesh_sampler();
        if sampler.is_null() {
            self.mesh_error =
                "the mesh sampler state creation failed".to_string();
            return false;
        }
        self.mesh_resources = Some(MetalMeshResources {
            pipeline,
            depth_state,
            sampler,
        });
        self.mesh_ready = true;
        true
    }

    /// The lit mesh path's pipeline (M5): the LIT MSL library pair
    /// compiles, the same vertex descriptor (the PNU attributes at
    /// buffer 1), the colour/depth/stencil pixel formats, the depth
    /// state and the sampler the unlit path shares. The same lazy
    /// before-the-frame rule the unlit path follows.
    fn ensure_lit_resources(&mut self) -> bool {
        if self.lit_ready {
            return true;
        }
        let vertex_library = match self.compile_stage(
            super::shader_library::LIT_VERTEX_SHADER,
            "the lit vertex stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.lit_error = message;
                return false;
            }
        };
        let vertex_function =
            match self.new_function(vertex_library, "litVertexMain") {
                Some(function) => function,
                None => {
                    self.lit_error =
                        "litVertexMain lookup failed".to_string();
                    return false;
                }
            };
        let fragment_library = match self.compile_stage(
            super::shader_library::LIT_FRAGMENT_SHADER,
            "the lit fragment stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.lit_error = message;
                return false;
            }
        };
        let fragment_function =
            match self.new_function(fragment_library, "litFragmentMain") {
                Some(function) => function,
                None => {
                    self.lit_error =
                        "litFragmentMain lookup failed".to_string();
                    return false;
                }
            };
        let descriptor = unsafe {
            self.loader
                .alloc_init(self.loader.cls_mtl_render_pipeline_descriptor)
        };
        if descriptor.is_null() {
            self.lit_error =
                "the lit pipeline descriptor alloc failed".to_string();
            return false;
        }
        let vertex_descriptor = unsafe {
            self.loader.send0(
                self.loader.cls_mtl_vertex_descriptor,
                self.loader.sel("vertexDescriptor"),
            )
        };
        if vertex_descriptor.is_null() {
            self.lit_error =
                "the lit vertex descriptor's constructor failed"
                    .to_string();
            return false;
        }
        let attributes = unsafe {
            self.loader.send0(
                vertex_descriptor,
                self.loader.sel("attributes"),
            )
        };
        let layouts = unsafe {
            self.loader.send0(vertex_descriptor, self.loader.sel("layouts"))
        };
        unsafe {
            // The same three attributes the unlit mesh path carries
            // (position, normal, uv at buffer index 1).
            for (index, (format, offset)) in [
                (MTLVF_FLOAT3, 0usize),
                (MTLVF_FLOAT3, 12usize),
                (MTLVF_FLOAT2, 24usize),
            ]
            .iter()
            .enumerate()
            {
                let attribute = self.loader.send_ul_ret_ptr(
                    attributes,
                    self.loader.sel("objectAtIndexedSubscript:"),
                    index as super::mtl_loader::NsUInteger,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setFormat:"),
                    *format,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setBufferIndex:"),
                    1,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setOffset:"),
                    *offset as super::mtl_loader::NsUInteger,
                );
            }
            let layout_1 = self.loader.send_ul_ret_ptr(
                layouts,
                self.loader.sel("objectAtIndexedSubscript:"),
                1,
            );
            self.loader.send_void_ul(
                layout_1,
                self.loader.sel("setStride:"),
                32,
            );
            self.loader.send_void_ul(
                layout_1,
                self.loader.sel("setStepFunction:"),
                MTLSF_PER_VERTEX,
            );
            self.loader.send_void_ptr(
                descriptor,
                self.loader.sel("setVertexDescriptor:"),
                vertex_descriptor,
            );
            let colour_attachments = self.loader.send0(
                descriptor,
                self.loader.sel("colorAttachments"),
            );
            let colour_attachment_0 = self.loader.send_ul_ret_ptr(
                colour_attachments,
                self.loader.sel("objectAtIndexedSubscript:"),
                0,
            );
            self.loader.send_void_ul(
                colour_attachment_0,
                self.loader.sel("setPixelFormat:"),
                MTLPX_RGBA8_UNORM,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setDepthAttachmentPixelFormat:"),
                self.depth_format,
            );
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setStencilAttachmentPixelFormat:"),
                self.depth_format,
            );
            // The winding rule and the cull mode stay at the API's own
            // defaults (the M4 fix's rule: the pipeline descriptor
            // carries no winding/cull setters).
            self.loader.send_void_ptr(
                descriptor,
                self.loader.sel("setVertexFunction:"),
                vertex_function,
            );
            self.loader.send_void_ptr(
                descriptor,
                self.loader.sel("setFragmentFunction:"),
                fragment_function,
            );
        }
        let pipeline = unsafe {
            let mut pipeline_error: *mut c_void = std::ptr::null_mut();
            let pipeline = self.loader.send_ptr_arg2(
                self.device,
                self.loader
                    .sel("newRenderPipelineStateWithDescriptor:error:"),
                descriptor,
                &mut pipeline_error as *mut *mut c_void as *mut c_void,
            );
            if pipeline.is_null() {
                self.lit_error = if pipeline_error.is_null() {
                    "the lit pipeline state creation returned nil"
                        .to_string()
                } else {
                    self.ns_error_text(pipeline_error)
                };
                return false;
            }
            pipeline
        };
        let depth_state = self.create_mesh_depth_state();
        if depth_state.is_null() {
            self.lit_error =
                "the lit depth/stencil state creation failed".to_string();
            return false;
        }
        let sampler = self.create_mesh_sampler();
        if sampler.is_null() {
            self.lit_error =
                "the lit sampler state creation failed".to_string();
            return false;
        }
        self.lit_resources = Some(MetalMeshResources {
            pipeline,
            depth_state,
            sampler,
        });
        self.lit_ready = true;
        true
    }
    /// The deferred path's two pipelines (M6): the G-buffer's
    /// three-attachment pipeline (the same PNU vertex descriptor the mesh
    /// path carries, the depth test on) and the lighting pass's
    /// single-attachment pipeline (no vertex descriptor, the depth test
    /// off — the lighting pass's fullscreen triangle covers the frame).
    /// The depth/stencil state and the sampler are the mesh path's own
    /// (the lazy create the mesh path's first upload ran).
    fn ensure_deferred_resources(&mut self) -> bool {
        if self.deferred_ready {
            return true;
        }
        // The G-buffer pipeline: the two MSL libraries compile (the same
        // compile shape the lit path carries), the vertex descriptor
        // carries the PNU attributes (the same three attributes the mesh
        // path carries), the colour attachments are the three formats the
        // G-buffer target carries (albedo Rgba8, normal Rgba8, world
        // Rgba16F), the depth test on.
        let gbuffer_vertex_library = match self.compile_stage(
            super::shader_library::GBUFFER_VERTEX_SHADER,
            "the gbuffer vertex stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.deferred_error = message;
                return false;
            }
        };
        let gbuffer_vertex_function = match self.new_function(
            gbuffer_vertex_library,
            "gbufferVertexMain",
        ) {
            Some(function) => function,
            None => {
                self.deferred_error =
                    "gbufferVertexMain lookup failed".to_string();
                return false;
            }
        };
        let gbuffer_fragment_library = match self.compile_stage(
            super::shader_library::GBUFFER_FRAGMENT_SHADER,
            "the gbuffer fragment stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.deferred_error = message;
                return false;
            }
        };
        let gbuffer_fragment_function = match self.new_function(
            gbuffer_fragment_library,
            "gbufferFragmentMain",
        ) {
            Some(function) => function,
            None => {
                self.deferred_error =
                    "gbufferFragmentMain lookup failed".to_string();
                return false;
            }
        };
        let gbuffer_descriptor = unsafe {
            self.loader
                .alloc_init(self.loader.cls_mtl_render_pipeline_descriptor)
        };
        if gbuffer_descriptor.is_null() {
            self.deferred_error =
                "the gbuffer pipeline descriptor alloc failed".to_string();
            return false;
        }
        unsafe {
            // The same three attributes the mesh path carries (position,
            // normal, uv at buffer index 1).
            let vertex_descriptor = self.loader.alloc_init(
                self.loader.cls_mtl_vertex_descriptor,
            );
            let attributes = self.loader.send0(
                vertex_descriptor,
                self.loader.sel("attributes"),
            );
            let layouts = self.loader.send0(
                vertex_descriptor,
                self.loader.sel("layouts"),
            );
            for (index, (format, offset)) in [
                (MTLVF_FLOAT3, 0usize),
                (MTLVF_FLOAT3, 12usize),
                (MTLVF_FLOAT2, 24usize),
            ]
            .iter()
            .enumerate()
            {
                let attribute = self.loader.send_ul_ret_ptr(
                    attributes,
                    self.loader.sel("objectAtIndexedSubscript:"),
                    index as super::mtl_loader::NsUInteger,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setFormat:"),
                    *format,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setBufferIndex:"),
                    1,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setOffset:"),
                    *offset as super::mtl_loader::NsUInteger,
                );
            }
            let layout_1 = self.loader.send_ul_ret_ptr(
                layouts,
                self.loader.sel("objectAtIndexedSubscript:"),
                1,
            );
            self.loader.send_void_ul(
                layout_1,
                self.loader.sel("setStride:"),
                32,
            );
            self.loader.send_void_ul(
                layout_1,
                self.loader.sel("setStepFunction:"),
                MTLSF_PER_VERTEX,
            );
            self.loader.send_void_ptr(
                gbuffer_descriptor,
                self.loader.sel("setVertexDescriptor:"),
                vertex_descriptor,
            );
            // The three colour attachments: the G-buffer's albedo,
            // normal and world position formats.
            let colour_attachments = self.loader.send0(
                gbuffer_descriptor,
                self.loader.sel("colorAttachments"),
            );
            for (index, format) in [
                MTLPX_RGBA8_UNORM,
                MTLPX_RGBA8_UNORM,
                MTLPX_RGBA16_FLOAT,
            ]
            .iter()
            .enumerate()
            {
                let attachment = self.loader.send_ul_ret_ptr(
                    colour_attachments,
                    self.loader.sel("objectAtIndexedSubscript:"),
                    index as super::mtl_loader::NsUInteger,
                );
                self.loader.send_void_ul(
                    attachment,
                    self.loader.sel("setPixelFormat:"),
                    *format,
                );
            }
            self.loader.send_void_ul(
                gbuffer_descriptor,
                self.loader.sel("setDepthAttachmentPixelFormat:"),
                self.depth_format,
            );
            self.loader.send_void_ul(
                gbuffer_descriptor,
                self.loader.sel("setStencilAttachmentPixelFormat:"),
                self.depth_format,
            );
            self.loader.send_void_ptr(
                gbuffer_descriptor,
                self.loader.sel("setVertexFunction:"),
                gbuffer_vertex_function,
            );
            self.loader.send_void_ptr(
                gbuffer_descriptor,
                self.loader.sel("setFragmentFunction:"),
                gbuffer_fragment_function,
            );
        }
        let gbuffer_pipeline = unsafe {
            let mut pipeline_error: *mut c_void = std::ptr::null_mut();
            let pipeline = self.loader.send_ptr_arg2(
                self.device,
                self.loader
                    .sel("newRenderPipelineStateWithDescriptor:error:"),
                gbuffer_descriptor,
                &mut pipeline_error as *mut *mut c_void as *mut c_void,
            );
            if pipeline.is_null() {
                self.deferred_error = if pipeline_error.is_null() {
                    "the gbuffer pipeline state creation returned nil"
                        .to_string()
                } else {
                    self.ns_error_text(pipeline_error)
                };
                return false;
            }
            pipeline
        };
        // The lighting pipeline: the two MSL libraries compile, no vertex
        // descriptor (the fullscreen triangle the vertex id generates),
        // one colour attachment (the default framebuffer's Rgba8), no
        // depth attachment (the depth test off).
        let lighting_vertex_library = match self.compile_stage(
            super::shader_library::LIGHTING_VERTEX_SHADER,
            "the lighting vertex stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.deferred_error = message;
                return false;
            }
        };
        let lighting_vertex_function = match self.new_function(
            lighting_vertex_library,
            "lightingVertexMain",
        ) {
            Some(function) => function,
            None => {
                self.deferred_error =
                    "lightingVertexMain lookup failed".to_string();
                return false;
            }
        };
        let lighting_fragment_library = match self.compile_stage(
            super::shader_library::LIGHTING_FRAGMENT_SHADER,
            "the lighting fragment stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.deferred_error = message;
                return false;
            }
        };
        let lighting_fragment_function = match self.new_function(
            lighting_fragment_library,
            "lightingFragmentMain",
        ) {
            Some(function) => function,
            None => {
                self.deferred_error =
                    "lightingFragmentMain lookup failed".to_string();
                return false;
            }
        };
        let lighting_descriptor = unsafe {
            self.loader
                .alloc_init(self.loader.cls_mtl_render_pipeline_descriptor)
        };
        if lighting_descriptor.is_null() {
            self.deferred_error =
                "the lighting pipeline descriptor alloc failed".to_string();
            return false;
        }
        unsafe {
            // One colour attachment (the default framebuffer's Rgba8),
            // no depth attachment (the depth test off — the lighting
            // pass's triangle covers the frame and the depth attachment
            // stays untouched).
            let colour_attachments = self.loader.send0(
                lighting_descriptor,
                self.loader.sel("colorAttachments"),
            );
            let attachment_0 = self.loader.send_ul_ret_ptr(
                colour_attachments,
                self.loader.sel("objectAtIndexedSubscript:"),
                0,
            );
            self.loader.send_void_ul(
                attachment_0,
                self.loader.sel("setPixelFormat:"),
                MTLPX_RGBA8_UNORM,
            );
            self.loader.send_void_ptr(
                lighting_descriptor,
                self.loader.sel("setVertexFunction:"),
                lighting_vertex_function,
            );
            self.loader.send_void_ptr(
                lighting_descriptor,
                self.loader.sel("setFragmentFunction:"),
                lighting_fragment_function,
            );
        }
        let lighting_pipeline = unsafe {
            let mut pipeline_error: *mut c_void = std::ptr::null_mut();
            let pipeline = self.loader.send_ptr_arg2(
                self.device,
                self.loader
                    .sel("newRenderPipelineStateWithDescriptor:error:"),
                lighting_descriptor,
                &mut pipeline_error as *mut *mut c_void as *mut c_void,
            );
            if pipeline.is_null() {
                self.deferred_error = if pipeline_error.is_null() {
                    "the lighting pipeline state creation returned nil"
                        .to_string()
                } else {
                    self.ns_error_text(pipeline_error)
                };
                return false;
            }
            pipeline
        };
        // The lights' shared buffer (one 256-byte block the lighting
        // pass writes and the fragment stage's [[buffer(1)]] reads).
        let lights_buffer = self.create_shared_buffer(&[0u8; 256]);
        if lights_buffer.is_null() {
            self.deferred_error =
                "the lighting lights buffer could not be allocated"
                    .to_string();
            return false;
        }
        self.gbuffer_pipeline = gbuffer_pipeline;
        self.lighting_pipeline = lighting_pipeline;
        self.lighting_lights_buffer = lights_buffer;
        self.deferred_ready = true;
        true
    }
    /// The post-processing chain's resources (M7): the HDR geometry
    /// pipeline (the two Rgba16F attachments) and the composite pipeline
    /// (the fullscreen triangle over the default framebuffer) plus the
    /// lights' and the config's shared buffers the passes write.
    fn ensure_post_resources(&mut self) -> bool {
        if self.post_ready {
            return true;
        }
        // The post geometry pipeline: the two MSL libraries compile (the
        // same compile shape the deferred path carries), the vertex
        // descriptor carries the PNU attributes (the same three
        // attributes the mesh path carries), the colour attachments are
        // the two Rgba16F formats the HDR target carries, the depth test
        // on.
        let vertex_library = match self.compile_stage(
            super::shader_library::GBUFFER_VERTEX_SHADER,
            "the post geometry vertex stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.post_error = message;
                return false;
            }
        };
        let vertex_function =
            match self.new_function(vertex_library, "gbufferVertexMain") {
                Some(function) => function,
                None => {
                    self.post_error =
                        "gbufferVertexMain lookup failed".to_string();
                    return false;
                }
            };
        let fragment_library = match self.compile_stage(
            super::shader_library::POST_GBUFFER_FRAGMENT_SHADER,
            "the post geometry fragment stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.post_error = message;
                return false;
            }
        };
        let fragment_function = match self.new_function(
            fragment_library,
            "postGbufferFragmentMain",
        ) {
            Some(function) => function,
            None => {
                self.post_error =
                    "postGbufferFragmentMain lookup failed".to_string();
                return false;
            }
        };
        let gbuffer_descriptor = unsafe {
            self.loader
                .alloc_init(self.loader.cls_mtl_render_pipeline_descriptor)
        };
        if gbuffer_descriptor.is_null() {
            self.post_error =
                "the post geometry pipeline descriptor alloc failed"
                    .to_string();
            return false;
        }
        unsafe {
            // The same three attributes the mesh path carries (position,
            // normal, uv at buffer index 1).
            let vertex_descriptor = self.loader.alloc_init(
                self.loader.cls_mtl_vertex_descriptor,
            );
            let attributes = self.loader.send0(
                vertex_descriptor,
                self.loader.sel("attributes"),
            );
            let layouts = self.loader.send0(
                vertex_descriptor,
                self.loader.sel("layouts"),
            );
            for (index, (format, offset)) in [
                (MTLVF_FLOAT3, 0usize),
                (MTLVF_FLOAT3, 12usize),
                (MTLVF_FLOAT2, 24usize),
            ]
            .iter()
            .enumerate()
            {
                let attribute = self.loader.send_ul_ret_ptr(
                    attributes,
                    self.loader.sel("objectAtIndexedSubscript:"),
                    index as super::mtl_loader::NsUInteger,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setFormat:"),
                    *format,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setBufferIndex:"),
                    1,
                );
                self.loader.send_void_ul(
                    attribute,
                    self.loader.sel("setOffset:"),
                    *offset as super::mtl_loader::NsUInteger,
                );
            }
            let layout_1 = self.loader.send_ul_ret_ptr(
                layouts,
                self.loader.sel("objectAtIndexedSubscript:"),
                1,
            );
            self.loader.send_void_ul(
                layout_1,
                self.loader.sel("setStride:"),
                32,
            );
            self.loader.send_void_ul(
                layout_1,
                self.loader.sel("setStepFunction:"),
                MTLSF_PER_VERTEX,
            );
            self.loader.send_void_ptr(
                gbuffer_descriptor,
                self.loader.sel("setVertexDescriptor:"),
                vertex_descriptor,
            );
            // The two colour attachments: the HDR target's radiance and
            // world position formats.
            let colour_attachments = self.loader.send0(
                gbuffer_descriptor,
                self.loader.sel("colorAttachments"),
            );
            for (index, format) in
                [MTLPX_RGBA16_FLOAT, MTLPX_RGBA16_FLOAT].iter().enumerate()
            {
                let attachment = self.loader.send_ul_ret_ptr(
                    colour_attachments,
                    self.loader.sel("objectAtIndexedSubscript:"),
                    index as super::mtl_loader::NsUInteger,
                );
                self.loader.send_void_ul(
                    attachment,
                    self.loader.sel("setPixelFormat:"),
                    *format,
                );
            }
            self.loader.send_void_ul(
                gbuffer_descriptor,
                self.loader.sel("setDepthAttachmentPixelFormat:"),
                self.depth_format,
            );
            self.loader.send_void_ul(
                gbuffer_descriptor,
                self.loader.sel("setStencilAttachmentPixelFormat:"),
                self.depth_format,
            );
            self.loader.send_void_ptr(
                gbuffer_descriptor,
                self.loader.sel("setVertexFunction:"),
                vertex_function,
            );
            self.loader.send_void_ptr(
                gbuffer_descriptor,
                self.loader.sel("setFragmentFunction:"),
                fragment_function,
            );
        }
        let gbuffer_pipeline = unsafe {
            let mut pipeline_error: *mut c_void = std::ptr::null_mut();
            let pipeline = self.loader.send_ptr_arg2(
                self.device,
                self.loader
                    .sel("newRenderPipelineStateWithDescriptor:error:"),
                gbuffer_descriptor,
                &mut pipeline_error as *mut *mut c_void as *mut c_void,
            );
            if pipeline.is_null() {
                self.post_error = if pipeline_error.is_null() {
                    "the post geometry pipeline state creation returned nil"
                        .to_string()
                } else {
                    self.ns_error_text(pipeline_error)
                };
                return false;
            }
            pipeline
        };
        // The composite pipeline: the two MSL libraries compile, no
        // vertex descriptor (the fullscreen triangle the vertex id
        // generates), one colour attachment (the default framebuffer's
        // Rgba8), no depth attachment (the depth test off).
        let composite_vertex_library = match self.compile_stage(
            super::shader_library::LIGHTING_VERTEX_SHADER,
            "the composite vertex stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.post_error = message;
                return false;
            }
        };
        let composite_vertex_function = match self.new_function(
            composite_vertex_library,
            "lightingVertexMain",
        ) {
            Some(function) => function,
            None => {
                self.post_error =
                    "lightingVertexMain lookup failed".to_string();
                return false;
            }
        };
        let composite_fragment_library = match self.compile_stage(
            super::shader_library::POST_COMPOSITE_FRAGMENT_SHADER,
            "the composite fragment stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.post_error = message;
                return false;
            }
        };
        let composite_fragment_function = match self.new_function(
            composite_fragment_library,
            "postCompositeFragmentMain",
        ) {
            Some(function) => function,
            None => {
                self.post_error =
                    "postCompositeFragmentMain lookup failed".to_string();
                return false;
            }
        };
        let composite_descriptor = unsafe {
            self.loader
                .alloc_init(self.loader.cls_mtl_render_pipeline_descriptor)
        };
        if composite_descriptor.is_null() {
            self.post_error =
                "the composite pipeline descriptor alloc failed".to_string();
            return false;
        }
        unsafe {
            // One colour attachment (the default framebuffer's Rgba8),
            // no depth attachment (the depth test off — the composite's
            // triangle covers the frame).
            let colour_attachments = self.loader.send0(
                composite_descriptor,
                self.loader.sel("colorAttachments"),
            );
            let attachment_0 = self.loader.send_ul_ret_ptr(
                colour_attachments,
                self.loader.sel("objectAtIndexedSubscript:"),
                0,
            );
            self.loader.send_void_ul(
                attachment_0,
                self.loader.sel("setPixelFormat:"),
                MTLPX_RGBA8_UNORM,
            );
            self.loader.send_void_ptr(
                composite_descriptor,
                self.loader.sel("setVertexFunction:"),
                composite_vertex_function,
            );
            self.loader.send_void_ptr(
                composite_descriptor,
                self.loader.sel("setFragmentFunction:"),
                composite_fragment_function,
            );
        }
        let composite_pipeline = unsafe {
            let mut pipeline_error: *mut c_void = std::ptr::null_mut();
            let pipeline = self.loader.send_ptr_arg2(
                self.device,
                self.loader
                    .sel("newRenderPipelineStateWithDescriptor:error:"),
                composite_descriptor,
                &mut pipeline_error as *mut *mut c_void as *mut c_void,
            );
            if pipeline.is_null() {
                self.post_error = if pipeline_error.is_null() {
                    "the composite pipeline state creation returned nil"
                        .to_string()
                } else {
                    self.ns_error_text(pipeline_error)
                };
                return false;
            }
            pipeline
        };
        // The lights' shared buffer (one 256-byte block the geometry
        // pass writes and the fragment stage's [[buffer(1)]] reads).
        let lights_buffer = self.create_shared_buffer(&[0u8; 256]);
        if lights_buffer.is_null() {
            self.post_error =
                "the post lights buffer could not be allocated".to_string();
            return false;
        }
        // The config's shared buffer (one 48-byte block the composite
        // pass writes and the fragment stage's [[buffer(0)]] reads).
        let config_buffer = self.create_shared_buffer(&[0u8; 48]);
        if config_buffer.is_null() {
            self.post_error =
                "the post config buffer could not be allocated".to_string();
            return false;
        }
        // The volume's shared buffer (one 48-byte block the geometry
        // pass writes and the fragment stage's [[buffer(2)]] reads).
        let volume_buffer = self.create_shared_buffer(&[0u8; 48]);
        if volume_buffer.is_null() {
            self.post_error =
                "the post volume buffer could not be allocated".to_string();
            return false;
        }
        self.post_gbuffer_pipeline = gbuffer_pipeline;
        self.post_composite_pipeline = composite_pipeline;
        self.post_lights_buffer = lights_buffer;
        self.post_config_buffer = config_buffer;
        self.post_volume_buffer = volume_buffer;
        self.post_ready = true;
        true
    }

    /// The diagnostic triangle's pipeline: the MSL library compiles, the
    /// vertex descriptor carries the Float3 attribute, the colour
    /// attachment's pixel format follows the frame target's.
    fn ensure_triangle_resources(&mut self) -> bool {
        if self.tri_ready {
            return true;
        }
        // One library per stage (the same compile shape the C++ Metal
        // backend records): the combined-source library had no failure
        // signal this path could read, so the stage that failed stayed
        // unnamed in the smoke's diagnostic.
        let vertex_library = match self.compile_stage(
            super::shader_library::TRIANGLE_VERTEX_SHADER,
            "the vertex stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.tri_error = message;
                return false;
            }
        };
        let vertex_function =
            match self.new_function(vertex_library, "vertexMain") {
                Some(function) => function,
                None => {
                    self.tri_error = "vertexMain lookup failed".to_string();
                    return false;
                }
            };
        let fragment_library = match self.compile_stage(
            super::shader_library::TRIANGLE_FRAGMENT_SHADER,
            "the fragment stage's library",
        ) {
            Ok(library) => library,
            Err(message) => {
                self.tri_error = message;
                return false;
            }
        };
        let fragment_function =
            match self.new_function(fragment_library, "fragmentMain") {
                Some(function) => function,
                None => {
                    self.tri_error =
                        "fragmentMain lookup failed".to_string();
                    return false;
                }
            };
        let descriptor = unsafe {
            self.loader
                .alloc_init(self.loader.cls_mtl_render_pipeline_descriptor)
        };
        if descriptor.is_null() {
            self.tri_error =
                "pipeline descriptor alloc failed".to_string();
            return false;
        }
        unsafe {
            self.loader.send_void_ptr(
                descriptor,
                self.loader.sel("setVertexFunction:"),
                vertex_function,
            );
            self.loader.send_void_ptr(
                descriptor,
                self.loader.sel("setFragmentFunction:"),
                fragment_function,
            );
        }
        // The vertex descriptor: one Float3 attribute (position) at
        // buffer index 1 (the vertices' buffer; the uniform sits at
        // buffer index 0), per-vertex. The layout array is indexed by
        // buffer index, so the stride lands on layout 1. The
        // constructor is the class method vertexDescriptor (the API's
        // documented form: it returns "a default object with allocated
        // arrays in the attributes and layouts properties") — the
        // alloc/init pair leaves the arrays unallocated, the pipeline's
        // validation nils the state and the triangle draw is silently
        // dropped (the hardware run's missing draw caught it).
        let vertex_descriptor = unsafe {
            self.loader.send0(
                self.loader.cls_mtl_vertex_descriptor,
                self.loader.sel("vertexDescriptor"),
            )
        };
        if vertex_descriptor.is_null() {
            self.tri_error = "the vertex descriptor's constructor failed"
                .to_string();
            return false;
        }
        let attributes = unsafe {
            self.loader.send0(
                vertex_descriptor,
                self.loader.sel("attributes"),
            )
        };
        let layouts = unsafe {
            self.loader.send0(vertex_descriptor, self.loader.sel("layouts"))
        };
        let attribute_0 = unsafe {
            self.loader.send_ul_ret_ptr(
                attributes,
                self.loader.sel("objectAtIndexedSubscript:"),
                0,
            )
        };
        let layout_0 = unsafe {
            self.loader.send_ul_ret_ptr(
                layouts,
                self.loader.sel("objectAtIndexedSubscript:"),
                1,
            )
        };
        unsafe {
            self.loader.send_void_ul(
                attribute_0,
                self.loader.sel("setFormat:"),
                MTLVF_FLOAT3,
            );
            self.loader.send_void_ul(
                attribute_0,
                self.loader.sel("setBufferIndex:"),
                1,
            );
            self.loader.send_void_ul(
                attribute_0,
                self.loader.sel("setOffset:"),
                0,
            );
            self.loader.send_void_ul(
                layout_0,
                self.loader.sel("setStride:"),
                12,
            );
            self.loader.send_void_ul(
                layout_0,
                self.loader.sel("setStepFunction:"),
                MTLSF_PER_VERTEX,
            );
            self.loader.send_void_ptr(
                descriptor,
                self.loader.sel("setVertexDescriptor:"),
                vertex_descriptor,
            );
            // The colour attachment's pixel format follows the frame
            // target's (the off-screen target's RGBA8 unorm). The pipeline
            // descriptor carries the per-attachment formats through its
            // colourAttachments array (the same indexed-subscript pattern
            // the render pass descriptor uses; the hardware run caught the
            // direct-selector forms as Objective-C exceptions the Rust
            // runtime cannot catch).
            let colour_attachments = self.loader.send0(
                    descriptor,
                    self.loader.sel("colorAttachments"),
            );
            let colour_attachment_0 = self.loader.send_ul_ret_ptr(
                    colour_attachments,
                    self.loader.sel("objectAtIndexedSubscript:"),
                    0,
            );
            self.loader.send_void_ul(
                colour_attachment_0,
                self.loader.sel("setPixelFormat:"),
                MTLPX_RGBA8_UNORM,
            );
            // The depth attachment's pixel format (the depth/stencil
            // format the device picked).
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setDepthAttachmentPixelFormat:"),
                self.depth_format,
            );
            // The stencil attachment's pixel format: the combined
            // depth/stencil format carries the pass's stencilAttachment
            // texture too, so the pipeline's descriptor must name the
            // same format (its default MTLPixelFormatInvalid fails the
            // pipeline's match validation, which nils the pipeline state
            // and silently drops the triangle draw).
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setStencilAttachmentPixelFormat:"),
                self.depth_format,
            );
        }
        let pipeline = unsafe {
            let mut pipeline_error: *mut c_void = std::ptr::null_mut();
            let pipeline = self.loader.send_ptr_arg2(
                self.device,
                self.loader
                    .sel("newRenderPipelineStateWithDescriptor:error:"),
                descriptor,
                &mut pipeline_error as *mut *mut c_void as *mut c_void,
            );
            if pipeline.is_null() {
                self.tri_error = if pipeline_error.is_null() {
                    "the pipeline state creation returned nil".to_string()
                } else {
                    self.ns_error_text(pipeline_error)
                };
                return false;
            }
            pipeline
        };
        // The triangle's uniform buffer: the shared-storage buffer the
        // draw binds (the same staging rule the other backends follow).
        let uniform = self.build_tri_uniform();
        let buffer = self.create_shared_buffer(&uniform);
        if buffer.is_null() {
            self.tri_error =
                "uniform buffer allocation failed".to_string();
            return false;
        }
        self.tri_pipeline = pipeline;
        self.tri_uniform_buffer = buffer;
        self.tri_ready = true;
        true
    }

    /// The diagnostic triangle's draw: the pass's render encoder carries
    /// the pipeline, the uniform buffer and the triangle's three
    /// positions. The triangle never writes depth (the same scoped toggle
    /// the other backends apply).
    fn draw_triangle(&mut self, encoder: *mut c_void) {
        if self.shutdown || !self.tri_ready || encoder.is_null() {
            return;
        }
        unsafe {
            // The pipeline state and the uniform buffer bind at the
            // vertex stage's buffer index 0 (the shader's `buffer(0)`
            // qualifier) and the fragment stage's buffer index 0 (the
            // fragment shader's own `buffer(0)` qualifier).
            self.loader.send_void_ptr(
                encoder,
                self.loader.sel("setRenderPipelineState:"),
                self.tri_pipeline,
            );
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setVertexBuffer:offset:atIndex:"),
                self.tri_uniform_buffer,
                0,
                0,
            );
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setFragmentBuffer:offset:atIndex:"),
                self.tri_uniform_buffer,
                0,
                0,
            );
            // The triangle's three vertices (the NDC positions the
            // Vulkan backend's mesh carries), stride 12: the vertex
            // buffer at index 1 (the vertex descriptor's attribute 0
            // points there).
            let vertices: [f32; 9] = [
                -0.5, -0.5, 0.0, //
                0.5, -0.5, 0.0, //
                0.0, 0.5, 0.0,
            ];
            let vertex_bytes = std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * 4,
            );
            let vertex_buffer = self.create_shared_buffer(vertex_bytes);
            if vertex_buffer.is_null() {
                self.violations += 1;
                return;
            }
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setVertexBuffer:offset:atIndex:"),
                vertex_buffer,
                0,
                1,
            );
            self.loader.send_void_ul3(
                encoder,
                self.loader.sel("drawPrimitives:vertexStart:vertexCount:"),
                MTLPT_TRIANGLE,
                0,
                3,
            );
        }
    }

    /// The frame's clear: the render pass's load actions carry the clear
    /// values when the frame's command buffer runs. The command buffer
    /// stays open until the swap's commit (the Vulkan begin/submit
    /// frame's same shape).
    fn clear_frame(&mut self) -> *mut c_void {
        let command_buffer = unsafe {
            self.loader.send0(self.queue, self.loader.sel("commandBuffer"))
        };
        if command_buffer.is_null() {
            self.violations += 1;
            return std::ptr::null_mut();
        }
        // The render pass descriptor: the frame target's two attachments
        // with the clear load actions (the same shape the Vulkan
        // begin_frame's clear values carry).
        let pass_descriptor = unsafe {
            self.loader.alloc_init(
                self.loader.cls_mtl_render_pass_descriptor,
            )
        };
        if pass_descriptor.is_null() {
            self.violations += 1;
            return std::ptr::null_mut();
        }
        let colour_attachments = unsafe {
            self.loader.send0(
                pass_descriptor,
                self.loader.sel("colorAttachments"),
            )
        };
        let colour_attachment = unsafe {
            self.loader.send_ul_ret_ptr(
                colour_attachments,
                self.loader.sel("objectAtIndexedSubscript:"),
                0,
            )
        };
        let depth_attachment = unsafe {
            self.loader.send0(
                pass_descriptor,
                self.loader.sel("depthAttachment"),
            )
        };
        if colour_attachment.is_null() || depth_attachment.is_null() {
            self.violations += 1;
            return std::ptr::null_mut();
        }
        let colour = self.clear_color;
        let mtl_colour = MtlClearColor {
            red: colour.r as f64 / 255.0,
            green: colour.g as f64 / 255.0,
            blue: colour.b as f64 / 255.0,
            alpha: colour.a as f64 / 255.0,
        };
        unsafe {
            self.loader.send_void_ptr(
                colour_attachment,
                self.loader.sel("setTexture:"),
                self.color_texture,
            );
            self.loader.send_void_clear_color(
                colour_attachment,
                self.loader.sel("setClearColor:"),
                mtl_colour,
            );
            self.loader.send_void_ul(
                colour_attachment,
                self.loader.sel("setLoadAction:"),
                MTLLA_CLEAR,
            );
            self.loader.send_void_ul(
                colour_attachment,
                self.loader.sel("setStoreAction:"),
                MTLSA_STORE,
            );
            self.loader.send_void_ptr(
                depth_attachment,
                self.loader.sel("setTexture:"),
                self.depth_texture,
            );
            self.loader.send_void_double(
                depth_attachment,
                self.loader.sel("setClearDepth:"),
                self.clear_depth as f64,
            );
            // The depth attachment's own load/store actions: the
            // descriptor's default is DontCare, so the clear depth never
            // lands without them (the hardware run's NaN readback caught
            // the fix that moved the stencil selectors to the
            // stencilAttachment and left the depth attachment's own
            // actions unset).
            self.loader.send_void_ul(
                depth_attachment,
                self.loader.sel("setLoadAction:"),
                MTLLA_CLEAR,
            );
            self.loader.send_void_ul(
                depth_attachment,
                self.loader.sel("setStoreAction:"),
                MTLSA_STORE,
            );
            // The stencil attachment is a separate descriptor: the
            // depth attachment's descriptor carries no stencil selectors
            // (setClearStencil:, setStencilLoadAction: and
            // setStencilStoreAction: belong to the stencilAttachment,
            // so sending them to the depth attachment raised the
            // NSInvalidArgumentException the Rust runtime cannot catch
            // — the hardware run's crash right after the describe:).
            let stencil_attachment = self.loader.send0(
                pass_descriptor,
                self.loader.sel("stencilAttachment"),
            );
            if stencil_attachment.is_null() {
                self.violations += 1;
                return std::ptr::null_mut();
            }
            self.loader.send_void_ptr(
                stencil_attachment,
                self.loader.sel("setTexture:"),
                self.depth_texture,
            );
            self.loader.send_void_ul(
                stencil_attachment,
                self.loader.sel("setClearStencil:"),
                self.clear_stencil as super::mtl_loader::NsUInteger,
            );
            self.loader.send_void_ul(
                stencil_attachment,
                self.loader.sel("setLoadAction:"),
                MTLLA_CLEAR,
            );
            self.loader.send_void_ul(
                stencil_attachment,
                self.loader.sel("setStoreAction:"),
                MTLSA_STORE,
            );
        }
        let encoder = unsafe {
            self.loader.send_ptr_arg(
                command_buffer,
                self.loader.sel("renderCommandEncoderWithDescriptor:"),
                pass_descriptor,
            )
        };
        if encoder.is_null() {
            self.violations += 1;
            return std::ptr::null_mut();
        }
        // The triangle's draw happens inside the pass (the encoder stays
        // open until the frame's end; the caller closes it).
        if self.config.draw_triangle {
            self.ensure_triangle_resources();
            self.draw_triangle(encoder);
        }
        unsafe {
            self.loader.send_void(encoder, self.loader.sel("endEncoding"));
        }
        command_buffer
    }

    /// The mesh draw's render pass descriptor: the frame target's two
    /// attachments with the LOAD load action (the clear pass's pixels
    /// stay; the same rule the Vulkan path's open render pass records —
    /// the Metal frame carries multiple render passes on one command
    /// buffer, each pass's load action keeps the previous pass's output).
    fn mesh_pass_descriptor(&mut self) -> *mut c_void {
        let pass_descriptor = unsafe {
            self.loader.alloc_init(
                self.loader.cls_mtl_render_pass_descriptor,
            )
        };
        if pass_descriptor.is_null() {
            self.violations += 1;
            return std::ptr::null_mut();
        }
        let colour_attachments = unsafe {
            self.loader.send0(
                pass_descriptor,
                self.loader.sel("colorAttachments"),
            )
        };
        let colour_attachment = unsafe {
            self.loader.send_ul_ret_ptr(
                colour_attachments,
                self.loader.sel("objectAtIndexedSubscript:"),
                0,
            )
        };
        let depth_attachment = unsafe {
            self.loader.send0(
                pass_descriptor,
                self.loader.sel("depthAttachment"),
            )
        };
        let stencil_attachment = unsafe {
            self.loader.send0(
                pass_descriptor,
                self.loader.sel("stencilAttachment"),
            )
        };
        if colour_attachment.is_null()
            || depth_attachment.is_null()
            || stencil_attachment.is_null()
        {
            self.violations += 1;
            return std::ptr::null_mut();
        }
        unsafe {
            self.loader.send_void_ptr(
                colour_attachment,
                self.loader.sel("setTexture:"),
                self.color_texture,
            );
            self.loader.send_void_ul(
                colour_attachment,
                self.loader.sel("setLoadAction:"),
                MTLLA_LOAD,
            );
            self.loader.send_void_ul(
                colour_attachment,
                self.loader.sel("setStoreAction:"),
                MTLSA_STORE,
            );
            self.loader.send_void_ptr(
                depth_attachment,
                self.loader.sel("setTexture:"),
                self.depth_texture,
            );
            self.loader.send_void_ul(
                depth_attachment,
                self.loader.sel("setLoadAction:"),
                MTLLA_LOAD,
            );
            self.loader.send_void_ul(
                depth_attachment,
                self.loader.sel("setStoreAction:"),
                MTLSA_STORE,
            );
            self.loader.send_void_ptr(
                stencil_attachment,
                self.loader.sel("setTexture:"),
                self.depth_texture,
            );
            self.loader.send_void_ul(
                stencil_attachment,
                self.loader.sel("setLoadAction:"),
                MTLLA_LOAD,
            );
            self.loader.send_void_ul(
                stencil_attachment,
                self.loader.sel("setStoreAction:"),
                MTLSA_STORE,
            );
        }
        pass_descriptor
    }

    /// The mesh's draw: the render command encoder the frame's command
    /// buffer carries (the pass's load action LOAD keeps the clear's
    /// pixels), the pipeline state, the depth/stencil state, the uniform
    /// buffer at buffer index 0 (both stages), the vertex buffer at
    /// index 1, the index buffer, the bound texture and the sampler at
    /// the fragment stage's slots 0. The draw is the indexed form
    /// (drawIndexedPrimitives) the uint32 index buffer carries.
    fn draw_mesh_by_id(
        &mut self,
        mesh_id: usize,
        model: Mat4,
    ) {
        self.draw_mesh_inner(mesh_id, model, false);
    }

    /// The lit draw (M5): the same path, the lit pipeline and the
    /// lights' buffer at the fragment stage's buffer index 1.
    fn draw_mesh_lit_by_id(
        &mut self,
        mesh_id: usize,
        model: Mat4,
    ) {
        self.draw_mesh_inner(mesh_id, model, true);
    }

    /// The mesh draw's shared body: the unlit path binds the mesh
    /// pipeline; the lit path binds the lit pipeline, writes the lights'
    /// block into the shared buffer and binds it at the fragment stage's
    /// buffer index 1 (the LIT shader's `buffer(1)` qualifier).
    fn draw_mesh_inner(
        &mut self,
        mesh_id: usize,
        model: Mat4,
        lit: bool,
    ) {
        let command_buffer = self.frame_command_buffer;
        if self.shutdown || command_buffer.is_null() {
            self.violations += 1;
            return;
        }
        let state = match self.meshes.get(mesh_id) {
            Some(Some(state)) => state,
            _ => {
                self.violations += 1;
                return;
            }
        };
        if !self.mesh_ready {
            return;
        }
        if lit && !self.lit_ready {
            return;
        }
        // The uniform block: the view-projection matrix from the
        // set_camera state and the model matrix the submission carries.
        let uniform = self.build_mesh_uniform(model);
        let uniform_bytes = state.uniform_buffer;
        let contents = unsafe {
            self.loader.send0(uniform_bytes, self.loader.sel("contents"))
        };
        if contents.is_null() {
            self.violations += 1;
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                uniform.as_ptr(),
                contents as *mut u8,
                uniform.len(),
            );
        }
        // The pipeline's handles copy out of the option (the pass
        // descriptor's creation needs the &mut self borrow; the handles
        // are plain pointers, the copy is the borrow's clean split).
        let (pipeline, depth_state, sampler) = {
            let resources = if lit {
                self.lit_resources.as_ref().unwrap()
            } else {
                self.mesh_resources.as_ref().unwrap()
            };
            (resources.pipeline, resources.depth_state, resources.sampler)
        };
        // The lights' buffer (the lit path): the block's copy into the
        // shared buffer's contents (the same write the uniform's copy
        // runs).
        let lights_buffer = if lit {
            if self.lit_lights_buffer.is_null() {
                self.violations += 1;
                return;
            }
            let contents = unsafe {
                self.loader.send0(
                    self.lit_lights_buffer,
                    self.loader.sel("contents"),
                )
            };
            if contents.is_null() {
                self.violations += 1;
                return;
            }
            let block = self.lit_block;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &block as *const LightsUniform as *const u8,
                    contents as *mut u8,
                    256,
                );
            }
            self.lit_lights_buffer
        } else {
            std::ptr::null_mut()
        };
        let texture = self
            .bound_texture
            .and_then(|i| self.textures_2d.get(i))
            .filter(|t| !t.released)
            .map(|t| t.texture)
            .or_else(|| {
                self.default_texture
                    .and_then(|i| self.textures_2d.get(i))
                    .filter(|t| !t.released)
                    .map(|t| t.texture)
            });
        let texture = match texture {
            Some(texture) => texture,
            None => {
                self.violations += 1;
                return;
            }
        };
        let (vertex_buffer, index_buffer, index_count) = (
            state.vertex_buffer,
            state.index_buffer,
            state.index_count,
        );
        let pass_descriptor = self.mesh_pass_descriptor();
        if pass_descriptor.is_null() {
            return;
        }
        let encoder = unsafe {
            self.loader.send_ptr_arg(
                command_buffer,
                self.loader.sel("renderCommandEncoderWithDescriptor:"),
                pass_descriptor,
            )
        };
        if encoder.is_null() {
            self.violations += 1;
            return;
        }
        unsafe {
            // The pipeline state and the depth/stencil state.
            self.loader.send_void_ptr(
                encoder,
                self.loader.sel("setRenderPipelineState:"),
                pipeline,
            );
            self.loader.send_void_ptr(
                encoder,
                self.loader.sel("setDepthStencilState:"),
                depth_state,
            );
            // The uniform buffer at the vertex stage's buffer index 0
            // (the shader's `buffer(0)` qualifier) and the fragment
            // stage's own buffer index 0 (the same Camera struct both
            // stages declare).
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setVertexBuffer:offset:atIndex:"),
                uniform_bytes,
                0,
                0,
            );
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setFragmentBuffer:offset:atIndex:"),
                uniform_bytes,
                0,
                0,
            );
            if lit {
                // The lights' buffer at the fragment stage's buffer index
                // 1 (the LIT shader's `buffer(1)` qualifier).
                self.loader.send_void_ptr_ul2(
                    encoder,
                    self.loader.sel("setFragmentBuffer:offset:atIndex:"),
                    lights_buffer,
                    0,
                    1,
                );
            }
            // The vertex buffer at index 1 (the vertex descriptor's
            // attributes point there).
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setVertexBuffer:offset:atIndex:"),
                vertex_buffer,
                0,
                1,
            );
            // The fragment stage's texture and sampler at slot 0 (the
            // shader's `texture(0)` and `sampler(0)` qualifiers).
            self.loader.send_void_ptr_ul(
                encoder,
                self.loader.sel("setFragmentTexture:atIndex:"),
                texture,
                0,
            );
            self.loader.send_void_ptr_ul(
                encoder,
                self.loader.sel("setFragmentSamplerState:atIndex:"),
                sampler,
                0,
            );
            // The indexed draw: the drawIndexedPrimitives call the
            // uint32 index buffer carries.
            self.loader.send_void_ul3_ptr_ul(
                encoder,
                self.loader.sel(
                    "drawIndexedPrimitives:indexCount:indexType:\
                     indexBuffer:indexBufferOffset:",
                ),
                MTLPT_TRIANGLE,
                index_count as super::mtl_loader::NsUInteger,
                MTLIT_UINT32,
                index_buffer,
                0,
            );
            self.loader.send_void(encoder, self.loader.sel("endEncoding"));
        }
    }
    /// The G-buffer pass's render pass descriptor: the target's three
    /// colour attachments (the MRT order the G-buffer's shaders carry)
    /// plus the depth/stencil attachment, each with the CLEAR load action
    /// (the geometry pass's own clear — the same rule the Vulkan
    /// gbuffer_render_pass's clear values carry).
    fn gbuffer_pass_descriptor(&mut self, target_index: usize) -> *mut c_void {
        let pass_descriptor = unsafe {
            self.loader.alloc_init(
                self.loader.cls_mtl_render_pass_descriptor,
            )
        };
        if pass_descriptor.is_null() {
            self.violations += 1;
            return std::ptr::null_mut();
        }
        let colour_attachments = unsafe {
            self.loader.send0(
                pass_descriptor,
                self.loader.sel("colorAttachments"),
            )
        };
        let depth_attachment = unsafe {
            self.loader.send0(
                pass_descriptor,
                self.loader.sel("depthAttachment"),
            )
        };
        let stencil_attachment = unsafe {
            self.loader.send0(
                pass_descriptor,
                self.loader.sel("stencilAttachment"),
            )
        };
        let depth_texture = self.targets[target_index]
            .depth_view
            .as_ref()
            .map(|d| d.texture)
            .unwrap_or(std::ptr::null_mut());
        let colour = self.clear_color;
        let clear = MtlClearColor {
            red: colour.r as f64 / 255.0,
            green: colour.g as f64 / 255.0,
            blue: colour.b as f64 / 255.0,
            alpha: colour.a as f64 / 255.0,
        };
        for (index, view) in self.targets[target_index].views.iter().enumerate()
        {
            let attachment = unsafe {
                self.loader.send_ul_ret_ptr(
                    colour_attachments,
                    self.loader.sel("objectAtIndexedSubscript:"),
                    index as super::mtl_loader::NsUInteger,
                )
            };
            if attachment.is_null() {
                self.violations += 1;
                return std::ptr::null_mut();
            }
            unsafe {
                self.loader.send_void_ptr(
                    attachment,
                    self.loader.sel("setTexture:"),
                    view.texture,
                );
                self.loader.send_void_clear_color(
                    attachment,
                    self.loader.sel("setClearColor:"),
                    clear,
                );
                self.loader.send_void_ul(
                    attachment,
                    self.loader.sel("setLoadAction:"),
                    MTLLA_CLEAR,
                );
                self.loader.send_void_ul(
                    attachment,
                    self.loader.sel("setStoreAction:"),
                    MTLSA_STORE,
                );
            }
        }
        if !depth_texture.is_null() {
            unsafe {
                self.loader.send_void_ptr(
                    depth_attachment,
                    self.loader.sel("setTexture:"),
                    depth_texture,
                );
                self.loader.send_void_double(
                    depth_attachment,
                    self.loader.sel("setClearDepth:"),
                    self.clear_depth as f64,
                );
                self.loader.send_void_ul(
                    depth_attachment,
                    self.loader.sel("setLoadAction:"),
                    MTLLA_CLEAR,
                );
                self.loader.send_void_ul(
                    depth_attachment,
                    self.loader.sel("setStoreAction:"),
                    MTLSA_STORE,
                );
                self.loader.send_void_ptr(
                    stencil_attachment,
                    self.loader.sel("setTexture:"),
                    depth_texture,
                );
                self.loader.send_void_ul(
                    stencil_attachment,
                    self.loader.sel("setClearStencil:"),
                    self.clear_stencil as super::mtl_loader::NsUInteger,
                );
                self.loader.send_void_ul(
                    stencil_attachment,
                    self.loader.sel("setLoadAction:"),
                    MTLLA_CLEAR,
                );
                self.loader.send_void_ul(
                    stencil_attachment,
                    self.loader.sel("setStoreAction:"),
                    MTLSA_STORE,
                );
            }
        }
        pass_descriptor
    }

    /// The post chain's HDR pass descriptor (M7): the two colour
    /// attachments the HDR target carries (the radiance clears to the
    /// clear colour's radiance, the world position clears to zero — the
    /// alpha 0 marks the background) plus the depth attachment.
    fn hdr_pass_descriptor(&mut self, target_index: usize) -> *mut c_void {
        let pass_descriptor = unsafe {
            self.loader.alloc_init(
                self.loader.cls_mtl_render_pass_descriptor,
            )
        };
        if pass_descriptor.is_null() {
            self.violations += 1;
            return std::ptr::null_mut();
        }
        let colour_attachments = unsafe {
            self.loader.send0(
                pass_descriptor,
                self.loader.sel("colorAttachments"),
            )
        };
        let depth_attachment = unsafe {
            self.loader.send0(
                pass_descriptor,
                self.loader.sel("depthAttachment"),
            )
        };
        let stencil_attachment = unsafe {
            self.loader.send0(
                pass_descriptor,
                self.loader.sel("stencilAttachment"),
            )
        };
        let depth_texture = self.targets[target_index]
            .depth_view
            .as_ref()
            .map(|d| d.texture)
            .unwrap_or(std::ptr::null_mut());
        let colour = self.clear_color;
        let clear = MtlClearColor {
            red: colour.r as f64 / 255.0,
            green: colour.g as f64 / 255.0,
            blue: colour.b as f64 / 255.0,
            alpha: colour.a as f64 / 255.0,
        };
        // The two colour attachments: the radiance clears to the clear
        // colour's radiance, the world position clears to zero.
        for (index, clear_value) in [clear, MtlClearColor { red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0 }]
            .iter()
            .enumerate()
        {
            let view = match self.targets[target_index].views.get(index) {
                Some(view) => view,
                None => {
                    self.violations += 1;
                    return std::ptr::null_mut();
                }
            };
            let attachment = unsafe {
                self.loader.send_ul_ret_ptr(
                    colour_attachments,
                    self.loader.sel("objectAtIndexedSubscript:"),
                    index as super::mtl_loader::NsUInteger,
                )
            };
            if attachment.is_null() {
                self.violations += 1;
                return std::ptr::null_mut();
            }
            unsafe {
                self.loader.send_void_ptr(
                    attachment,
                    self.loader.sel("setTexture:"),
                    view.texture,
                );
                self.loader.send_void_clear_color(
                    attachment,
                    self.loader.sel("setClearColor:"),
                    *clear_value,
                );
                self.loader.send_void_ul(
                    attachment,
                    self.loader.sel("setLoadAction:"),
                    MTLLA_CLEAR,
                );
                self.loader.send_void_ul(
                    attachment,
                    self.loader.sel("setStoreAction:"),
                    MTLSA_STORE,
                );
            }
        }
        if !depth_texture.is_null() {
            unsafe {
                self.loader.send_void_ptr(
                    depth_attachment,
                    self.loader.sel("setTexture:"),
                    depth_texture,
                );
                self.loader.send_void_double(
                    depth_attachment,
                    self.loader.sel("setClearDepth:"),
                    self.clear_depth as f64,
                );
                self.loader.send_void_ul(
                    depth_attachment,
                    self.loader.sel("setLoadAction:"),
                    MTLLA_CLEAR,
                );
                self.loader.send_void_ul(
                    depth_attachment,
                    self.loader.sel("setStoreAction:"),
                    MTLSA_STORE,
                );
                self.loader.send_void_ptr(
                    stencil_attachment,
                    self.loader.sel("setTexture:"),
                    depth_texture,
                );
                self.loader.send_void_ul(
                    stencil_attachment,
                    self.loader.sel("setClearStencil:"),
                    self.clear_stencil as super::mtl_loader::NsUInteger,
                );
                self.loader.send_void_ul(
                    stencil_attachment,
                    self.loader.sel("setLoadAction:"),
                    MTLLA_CLEAR,
                );
                self.loader.send_void_ul(
                    stencil_attachment,
                    self.loader.sel("setStoreAction:"),
                    MTLSA_STORE,
                );
            }
        }
        pass_descriptor
    }

    /// The readback's one pixel: the blit command encoder copies the
    /// colour word (offset 0) and the depth/stencil word (offset 16) into
    /// a staging buffer the CPU reads. The depth and the stencil land in
    /// the same packed word the Vulkan readback validated.
    fn readback_pixel(&mut self, x: u32, y: u32) {
        if self.color_texture.is_null() {
            return;
        }
        let command_buffer = unsafe {
            self.loader.send0(self.queue, self.loader.sel("commandBuffer"))
        };
        if command_buffer.is_null() {
            return;
        }
        // The staging buffer: four bytes of colour (RGBA8) plus four
        // bytes of depth/stencil, the same six-word layout the Vulkan
        // readback validated (colour 0..4, depth 16..20, stencil 20..21).
        let staging = self.create_shared_buffer(&[0u8; 24]);
        let blit = unsafe {
            self.loader.send0(
                command_buffer,
                self.loader.sel("blitCommandEncoder"),
            )
        };
        if blit.is_null() {
            return;
        }
        // The depth/stencil copy: the combined formats' copies carry
        // the plane the copy names through the options variant (the
        // plain copy form is the API's forbidden form for these
        // formats — the hardware run's zero depth/stencil readback
        // caught it). The depth plane's copy carries the depth value
        // packed in the low word of each 4-byte destination pixel (the
        // 24/8 format's depth in the bottom three bytes; the 32F/8
        // format's depth float in the whole word), and the stencil
        // plane's copy carries the stencil byte in each 4-byte pixel's
        // bottom byte (the API's packed-copies' convention).
        let copy = self.loader.sel(
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:\
             sourceSize:toBuffer:destinationOffset:\
             destinationBytesPerRow:destinationBytesPerImage:options:",
        );
        unsafe {
            // The colour copy: one pixel at (x, y), one row of 4 bytes.
            let f: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                super::mtl_loader::MtlTextureHandle,
                super::mtl_loader::NsUInteger,
                super::mtl_loader::NsUInteger,
                MtlOrigin,
                MtlSize,
                super::mtl_loader::MtlBufferHandle,
                super::mtl_loader::NsUInteger,
                super::mtl_loader::NsUInteger,
                super::mtl_loader::NsUInteger,
                super::mtl_loader::NsUInteger,
            ) = std::mem::transmute(self.loader.objc_msg_send.unwrap());
            f(
                blit,
                copy,
                self.color_texture,
                0,
                0,
                MtlOrigin {
                    x: x as u64,
                    y: y as u64,
                    z: 0,
                },
                MtlSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                staging,
                0,
                4,
                4,
                0,
            );
            if !self.depth_texture.is_null() {
                f(
                    blit,
                    copy,
                    self.depth_texture,
                    0,
                    0,
                    MtlOrigin {
                        x: x as u64,
                        y: y as u64,
                        z: 0,
                    },
                    MtlSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    staging,
                    16,
                    4,
                    4,
                    MTLBLOPT_DEPTH_FROM_DEPTH_STENCIL,
                );
                // The stencil plane's copy: the API's rule is one plane
                // per copy (the options variant takes either the depth
                // or the stencil option, "but not both"), so the stencil
                // lands in its own copy at offset 20, one 4-byte pixel
                // carrying the stencil byte in its bottom byte.
                f(
                    blit,
                    copy,
                    self.depth_texture,
                    0,
                    0,
                    MtlOrigin {
                        x: x as u64,
                        y: y as u64,
                        z: 0,
                    },
                    MtlSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    staging,
                    20,
                    4,
                    4,
                    MTLBLOPT_STENCIL_FROM_DEPTH_STENCIL,
                );
            }
            self.loader.send_void(blit, self.loader.sel("endEncoding"));
        }
        unsafe {
            self.loader.send_void(command_buffer, self.loader.sel("commit"));
            self.loader.send_void(
                command_buffer,
                self.loader.sel("waitUntilCompleted"),
            );
        }
        let contents =
            unsafe { self.loader.send0(staging, self.loader.sel("contents")) };
        if contents.is_null() {
            return;
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(contents as *const u8, 24)
        };
        let colour = ClearColor {
            r: bytes[0],
            g: bytes[1],
            b: bytes[2],
            a: bytes[3],
        };
        // The depth/stencil word: the packed 4-byte texel the options
        // variant's copies land (the depth plane's copy puts the depth
        // in the low word — the 24/8 format's depth in the bottom three
        // bytes, the 32F/8 format's depth float in the whole word; the
        // stencil plane's copy puts the stencil byte in each pixel's
        // bottom byte).
        let depth_stencil = u32::from_le_bytes([
            bytes[16],
            bytes[17],
            bytes[18],
            bytes[19],
        ]);
        let stencil = (u32::from_le_bytes([
            bytes[20],
            bytes[21],
            bytes[22],
            bytes[23],
        ]) & 0xFF) as i32;
        let depth = if self.depth_format
            == MTLPX_DEPTH24_UNORM_STENCIL8
        {
            let depth24 = depth_stencil >> 8;
            depth24 as f32 / 16_777_215.0
        } else {
            f32::from_bits(depth_stencil)
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

impl RenderBackend for MetalBackend {
    fn describe(&self) -> String {
        format!("metal {}", self.device_name)
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
        self.frame_command_buffer = self.clear_frame();
        self.frame_open = true;
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
    }

    fn violation_count(&self) -> u64 {
        self.violations
    }

    fn create_shader_program(
        &mut self,
        id: u32,
    ) -> Result<ShaderProgram, RenderError> {
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
        let id = self.next_mesh_id;
        self.next_mesh_id += 1;
        Ok(GpuMesh {
            inner: Arc::new(Mutex::new(GpuMeshInner {
                id,
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
                "upload_mesh: the vertex data is not a multiple of the \
                 stride",
            ));
        }
        if indices.is_empty() {
            return Err(RenderError::from(
                "upload_mesh: the index data is empty",
            ));
        }
        // The mesh pipeline's resources: the same lazy ensure the
        // triangle path follows (one pipeline serves every mesh).
        if !self.ensure_mesh_resources() {
            return Err(RenderError::from(format!(
                "upload_mesh: the mesh pipeline could not be created \
                 ({})",
                self.mesh_error
            )));
        }
        // The lit pipeline's ensure (M5): the same before-the-frame rule
        // (the pipeline's creation inside an open frame's encoder is the
        // software path's crash the debug runs caught).
        if !self.ensure_lit_resources() {
            return Err(RenderError::from(format!(
                "upload_mesh: the lit pipeline could not be created ({})",
                self.lit_error
            )));
        }
        // The lights' shared buffer (M5): one 256-byte shared-storage
        // buffer every lit draw writes (the fragment stage's
        // [[buffer(1)]] reads it).
        if self.lit_lights_buffer.is_null() {
            self.lit_lights_buffer =
                self.create_shared_buffer(&[0u8; 256]);
            if self.lit_lights_buffer.is_null() {
                return Err(RenderError::from(
                    "upload_mesh: the lights buffer could not be allocated",
                ));
            }
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
        // The three shared-storage buffers: the PNU bytes, the u32
        // indices and the uniform block (the CPU writes, the GPU reads:
        // Metal's shared storage is GPU-visible, no staging copy the
        // Vulkan path needs exists here).
        let vertex_buffer = self.create_shared_buffer(vertices);
        if vertex_buffer.is_null() {
            return Err(RenderError::from(
                "upload_mesh: the vertex buffer could not be allocated",
            ));
        }
        let mut index_bytes = Vec::with_capacity(indices.len() * 4);
        for index in indices {
            index_bytes.extend_from_slice(&index.to_le_bytes());
        }
        let index_buffer = self.create_shared_buffer(&index_bytes);
        if index_buffer.is_null() {
            return Err(RenderError::from(
                "upload_mesh: the index buffer could not be allocated",
            ));
        }
        let uniform_buffer = self.create_shared_buffer(&[0u8; 128]);
        if uniform_buffer.is_null() {
            return Err(RenderError::from(
                "upload_mesh: the uniform buffer could not be allocated",
            ));
        }
        self.meshes[mesh_id] = Some(MetalMeshState {
            stride: layout.stride,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            vertex_count: (vertices.len() as u32) / layout.stride,
            index_count: indices.len() as u32,
        });
        let mut inner = mesh.inner.lock().unwrap();
        inner.vertex_count = (vertices.len() as u32) / layout.stride;
        inner.index_count = indices.len() as u32;
        inner.uploaded = true;
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
        let inner = submission.mesh.inner.lock().unwrap();
        if !inner.uploaded {
            self.violations += 1;
            return;
        }
        let mesh_id = inner.id;
        drop(inner);
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
            if self.lit_ready {
                self.draw_mesh_lit_by_id(mesh_id, submission.model);
                return;
            }
            // The lit path's failure falls back to the unlit draw (the
            // render keeps going).
        }
        self.draw_mesh_by_id(mesh_id, submission.model);
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
            let texture = self.create_device_texture(
                desc.width,
                desc.height,
                self.to_mtl_format(desc.color_formats[i]),
            )?;
            views.push(MetalTexture2D {
                inner: Arc::new(Mutex::new(TextureViewInner {
                    width: desc.width,
                    height: desc.height,
                    depth: 1,
                    format: desc.color_formats[i],
                    is_depth: false,
                    released: false,
                })),
                texture,
                released: false,
            });
        }
        let depth_view = if desc.depth_attachment {
            let texture = self.create_device_texture(
                desc.width,
                desc.height,
                self.to_mtl_format(desc.depth_format),
            )?;
            Some(MetalTexture2D {
                inner: Arc::new(Mutex::new(TextureViewInner {
                    width: desc.width,
                    height: desc.height,
                    depth: 1,
                    format: desc.depth_format,
                    is_depth: true,
                    released: false,
                })),
                texture,
                released: false,
            })
        } else {
            None
        };
        let target = MetalRenderTarget {
            inner: Arc::new(Mutex::new(RenderTargetInner {
                width: desc.width,
                height: desc.height,
                color_attachment_count: desc.color_attachment_count,
                depth_attachment: desc.depth_attachment,
                released: false,
            })),
            views,
            depth_view,
            released: false,
        };
        let handle = RenderTarget {
            inner: Arc::clone(&target.inner),
        };
        self.targets.push(target);
        Ok(handle)
    }

    fn begin_render_pass(&mut self, target: Option<&RenderTarget>) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.bound_target = target.map(|t| Arc::clone(&t.inner));
        self.pass_open = true;
    }

    fn end_render_pass(&mut self) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.bound_target = None;
        self.pass_open = false;
    }

    fn release_render_target(&mut self, target: &RenderTarget) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        match self
            .targets
            .iter_mut()
            .find(|t| Arc::ptr_eq(&t.inner, &target.inner))
        {
            Some(t) => {
                if t.released {
                    self.violations += 1;
                } else {
                    t.released = true;
                    target.inner.lock().unwrap().released = true;
                }
            }
            None => {
                self.violations += 1;
            }
        }
    }

    fn render_target_texture(&self, target: &RenderTarget) -> Option<TextureView> {
        self.render_target_texture_at(target, 0)
    }

    fn render_target_texture_at(
        &self,
        target: &RenderTarget,
        attachment: u32,
    ) -> Option<TextureView> {
        let index = self
            .targets
            .iter()
            .position(|t| Arc::ptr_eq(&t.inner, &target.inner))?;
        let t = &self.targets[index];
        if t.released {
            return None;
        }
        if attachment == DEPTH_ATTACHMENT {
            return t.depth_view.as_ref().map(|d| TextureView {
                inner: Arc::clone(&d.inner),
            });
        }
        t.views
            .get(attachment as usize)
            .map(|v| TextureView {
                inner: Arc::clone(&v.inner),
            })
    }

    fn bind_texture(&mut self, view: Option<&TextureView>, _unit: u32) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        // The first sampler unit: the mesh path's draw binds the texture
        // the state carries (the same rule the Vulkan and the OpenGL
        // paths record). A None clears the binding and the draw falls
        // back to the default texture.
        if _unit != 0 {
            return;
        }
        match view {
            None => {
                self.bound_texture = None;
            }
            Some(view) => {
                self.bound_texture = self
                    .textures_2d
                    .iter()
                    .position(|t| Arc::ptr_eq(&t.inner, &view.inner));
            }
        }
    }

    fn create_texture_2d(
        &mut self,
        desc: &Texture2DDesc,
    ) -> Result<TextureView, RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "create_texture_2d: the backend is shut down",
            ));
        }
        desc.validate()?;
        let texture = self.create_device_texture(
            desc.width,
            desc.height,
            self.to_mtl_format(desc.format),
        )?;
        let view = MetalTexture2D {
            inner: Arc::new(Mutex::new(TextureViewInner {
                width: desc.width,
                height: desc.height,
                depth: 1,
                format: desc.format,
                is_depth: false,
                released: false,
            })),
            texture,
            released: false,
        };
        let handle = TextureView {
            inner: Arc::clone(&view.inner),
        };
        self.textures_2d.push(view);
        Ok(handle)
    }

    fn release_texture_2d(&mut self, view: &TextureView) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        match self
            .textures_2d
            .iter_mut()
            .find(|t| Arc::ptr_eq(&t.inner, &view.inner))
        {
            Some(t) => {
                if t.released {
                    self.violations += 1;
                } else {
                    t.released = true;
                    view.inner.lock().unwrap().released = true;
                }
            }
            None => {
                self.violations += 1;
            }
        }
    }

    fn fill_texture_2d(
        &mut self,
        view: &TextureView,
        data: &[u8],
    ) -> Result<(), RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "fill_texture_2d: the backend is shut down",
            ));
        }
        let index = self
            .textures_2d
            .iter()
            .position(|t| Arc::ptr_eq(&t.inner, &view.inner))
            .ok_or_else(|| {
                RenderError::from(
                    "fill_texture_2d: the texture is not owned by this \
                     backend",
                )
            })?;
        if self.textures_2d[index].released {
            self.violations += 1;
            return Err(RenderError::from(
                "fill_texture_2d: the texture is released",
            ));
        }
        let inner = view.inner.lock().unwrap();
        let expected = (inner.width as usize)
            * (inner.height as usize)
            * bytes_per_pixel(inner.format) as usize;
        let width = inner.width;
        let height = inner.height;
        let format = inner.format;
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "fill_texture_2d: expected {} bytes, got {}",
                expected, data.len()
            )));
        }
        // The upload: the texture's replaceRegion call (the private
        // texture's CPU write path — the CPU writes the texels through
        // the driver's own staging copy; the same rule the Vulkan path's
        // staging upload records).
        let texture = self.textures_2d[index].texture;
        let region = MtlRegion {
            origin: MtlOrigin {
                x: 0,
                y: 0,
                z: 0,
            },
            size: MtlSize {
                width: width as super::mtl_loader::NsUInteger,
                height: height as super::mtl_loader::NsUInteger,
                depth: 1,
            },
        };
        let bytes_per_row =
            width * bytes_per_pixel(format) as u32;
        unsafe {
            let f: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                MtlRegion,
                super::mtl_loader::NsUInteger,
                *const c_void,
                super::mtl_loader::NsUInteger,
            ) = std::mem::transmute(self.loader.objc_msg_send.unwrap());
            f(
                texture,
                self.loader.sel(
                    "replaceRegion:mipmapLevel:withBytes:bytesPerRow:",
                ),
                region,
                0,
                data.as_ptr() as *const c_void,
                bytes_per_row as super::mtl_loader::NsUInteger,
            );
        }
        Ok(())
    }

    fn read_texture_2d(
        &self,
        view: &TextureView,
        data: &mut [u8],
    ) -> Result<bool, RenderError> {
        let index = self
            .textures_2d
            .iter()
            .position(|t| Arc::ptr_eq(&t.inner, &view.inner))
            .ok_or_else(|| {
                RenderError::from(
                    "read_texture_2d: the texture is not owned by this \
                     backend",
                )
            })?;
        if self.textures_2d[index].released {
            return Ok(false);
        }
        let inner = view.inner.lock().unwrap();
        let expected = (inner.width as usize)
            * (inner.height as usize)
            * bytes_per_pixel(inner.format) as usize;
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "read_texture_2d: expected {} bytes, got {}",
                expected,
                data.len()
            )));
        }
        // The readback is validation-only for now (the same rule the
        // other backends follow).
        Ok(false)
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
        let texture = self.create_device_texture_3d(
            desc.width,
            desc.height,
            desc.depth,
            self.to_mtl_format(desc.format),
        )?;
        let view = MetalTexture2D {
            inner: Arc::new(Mutex::new(TextureViewInner {
                width: desc.width,
                height: desc.height,
                depth: desc.depth,
                format: desc.format,
                is_depth: false,
                released: false,
            })),
            texture,
            released: false,
        };
        let handle = TextureView {
            inner: Arc::clone(&view.inner),
        };
        self.textures_3d.push(view);
        Ok(handle)
    }

    fn release_texture_3d(&mut self, view: &TextureView) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        match self
            .textures_3d
            .iter_mut()
            .find(|t| Arc::ptr_eq(&t.inner, &view.inner))
        {
            Some(t) => {
                if t.released {
                    self.violations += 1;
                } else {
                    t.released = true;
                    view.inner.lock().unwrap().released = true;
                }
            }
            None => {
                self.violations += 1;
            }
        }
    }

    fn fill_texture_3d(
        &mut self,
        view: &TextureView,
        data: &[u8],
    ) -> Result<(), RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "fill_texture_3d: the backend is shut down",
            ));
        }
        let index = self
            .textures_3d
            .iter()
            .position(|t| Arc::ptr_eq(&t.inner, &view.inner))
            .ok_or_else(|| {
                RenderError::from(
                    "fill_texture_3d: the texture is not owned by this \
                     backend",
                )
            })?;
        if self.textures_3d[index].released {
            self.violations += 1;
            return Err(RenderError::from(
                "fill_texture_3d: the texture is released",
            ));
        }
        let inner = view.inner.lock().unwrap();
        let expected = (inner.width as usize)
            * (inner.height as usize)
            * (inner.depth as usize)
            * bytes_per_pixel(inner.format) as usize;
        let (width, height, depth, format) =
            (inner.width, inner.height, inner.depth, inner.format);
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "fill_texture_3d: expected {} bytes, got {}",
                expected,
                data.len()
            )));
        }
        // The upload: the texture's replaceRegion call (the same CPU
        // write path the 2D fill follows, with the region's depth
        // carrying the volume's layers and the bytesPerImage covering
        // one layer's slice — the MtlRegion and the
        // replaceRegion:mipmapLevel:withBytes:bytesPerRow:bytesPerImage:
        // selector the 3D upload takes).
        let texture = self.textures_3d[index].texture;
        let region = MtlRegion {
            origin: MtlOrigin {
                x: 0,
                y: 0,
                z: 0,
            },
            size: MtlSize {
                width: width as super::mtl_loader::NsUInteger,
                height: height as super::mtl_loader::NsUInteger,
                depth: depth as super::mtl_loader::NsUInteger,
            },
        };
        let bytes_per_row = width * bytes_per_pixel(format) as u32;
        let bytes_per_image = bytes_per_row * height;
        unsafe {
            let f: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                MtlRegion,
                super::mtl_loader::NsUInteger,
                super::mtl_loader::NsUInteger,
                *const c_void,
                super::mtl_loader::NsUInteger,
                super::mtl_loader::NsUInteger,
            ) = std::mem::transmute(self.loader.objc_msg_send.unwrap());
            f(
                texture,
                self.loader.sel(
                    "replaceRegion:mipmapLevel:slice:withBytes:\
                     bytesPerRow:bytesPerImage:",
                ),
                region,
                0,
                0,
                data.as_ptr() as *const c_void,
                bytes_per_row as super::mtl_loader::NsUInteger,
                bytes_per_image as super::mtl_loader::NsUInteger,
            );
        }
        Ok(())
    }

    fn read_texture_3d(
        &mut self,
        view: &TextureView,
        data: &mut [u8],
    ) -> Result<bool, RenderError> {
        let index = self
            .textures_3d
            .iter()
            .position(|t| Arc::ptr_eq(&t.inner, &view.inner));
        let index = match index {
            Some(index) => index,
            None => {
                return Err(RenderError::from(
                    "read_texture_3d: the texture is not owned by this \
                     backend",
                ));
            }
        };
        if self.textures_3d[index].released {
            return Ok(false);
        }
        let inner = view.inner.lock().unwrap();
        let expected = (inner.width as usize)
            * (inner.height as usize)
            * (inner.depth as usize)
            * bytes_per_pixel(inner.format) as usize;
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "read_texture_3d: expected {} bytes, got {}",
                expected,
                data.len()
            )));
        }
        // The readback is validation-only for now (the same rule the
        // 2D path records).
        Ok(false)
    }

    fn set_light_volume(&mut self, volume: &LightVolume) {
        if self.shutdown {
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
        // The uniform block's values: the origin, the dims (the depth
        // carrying the z * 6 layers) and the cell size (the indirect
        // intensity and the toggle the submit writes).
        self.volume_uniform = [
            volume.origin[0],
            volume.origin[1],
            volume.origin[2],
            0.0,
            volume.dims[0] as f32,
            volume.dims[1] as f32,
            (volume.dims[2] * 6) as f32,
            0.0,
            volume.cell_size,
            0.0,
            0.0,
            0.0,
        ];
        // The old volume's texture releases (the one-volume rule the
        // contract's set_light_volume carries).
        if let Some(old) = self.light_volume.take() {
            let handle = TextureView {
                inner: old.inner.clone(),
            };
            self.release_texture_3d(&handle);
        }
        let index = self
            .textures_3d
            .iter()
            .position(|t| Arc::ptr_eq(&t.inner, &view.inner))
            .unwrap_or(self.textures_3d.len());
        self.light_volume = Some(MetalTexture2D {
            inner: view.inner,
            texture: self.textures_3d[index].texture,
            released: false,
        });
    }

    fn set_camera(&mut self, camera: CameraParams) {
        if self.shutdown {
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
        // The frame's command buffer commits (the pass's load actions and
        // the triangle's draw run now; the same rule the Vulkan
        // submit_frame's queue submit carries).
        let command_buffer = self.frame_command_buffer;
        self.frame_command_buffer = std::ptr::null_mut();
        if !command_buffer.is_null() {
            unsafe {
                self.loader
                    .send_void(command_buffer, self.loader.sel("commit"));
                self.loader.send_void(
                    command_buffer,
                    self.loader.sel("waitUntilCompleted"),
                );
            }
        }
        // The readback happens before the frame ends (the same
        // pre-present rule the other backends follow).
        self.readback();
        self.frame_open = false;
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
        self.last_readback.clear();
        let queue = std::mem::take(&mut self.readback_queue);
        if self.config.readback_center || !queue.is_empty() {
            self.readback_pixel(self.config.width / 2, self.config.height / 2);
        }
        for (x, y) in queue {
            self.readback_pixel(x, y);
        }
    }

    fn last_readback(&self) -> &[super::super::ReadbackSample] {
        &self.last_readback
    }

    fn has_center_pixel(&self) -> bool {
        self.has_center
    }

    fn center_pixel(&self) -> super::super::ClearColor {
        self.center
    }

    fn center_depth(&self) -> f32 {
        self.center_depth
    }

    fn center_stencil(&self) -> i32 {
        self.center_stencil
    }

    /// The deferred lighting pass (M6): the two internal steps the
    /// contract's submit_lighting_pass carries. Step one (the geometry
    /// pass) opens the G-buffer target's pass descriptor (the three
    /// colour attachments plus the depth attachment, each with the clear
    /// load action), binds the G-buffer pipeline and draws the mesh. Step
    /// two (the lighting pass) opens the default framebuffer's pass
    /// descriptor (the load action LOAD keeps the clear's pixels), binds
    /// the lighting pipeline and draws the fullscreen triangle. The two
    /// passes record into the same frame's command buffer (the same rule
    /// the mesh draw path follows: each pass's load action keeps the
    /// previous pass's output).
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
        if mesh_id >= self.meshes.len() || self.meshes[mesh_id].is_none() {
            self.violations += 1;
            return;
        }
        if !self.mesh_ready || !self.ensure_deferred_resources() {
            return;
        }
        if !self.frame_open {
            // The deferred pass runs inside the frame's open command
            // buffer (the clear() the frame's first step runs opened it);
            // a missing frame is a violation.
            self.violations += 1;
            return;
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
        if !self.lighting_lights_buffer.is_null() {
            let contents = unsafe {
                self.loader.send0(
                    self.lighting_lights_buffer,
                    self.loader.sel("contents"),
                )
            };
            if !contents.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &lights_block as *const LightsUniform as *const u8,
                        contents as *mut u8,
                        std::mem::size_of::<LightsUniform>(),
                    );
                }
            }
        }
        // The camera block: the view-projection matrix from the
        // set_camera state and the model matrix the submission carries
        // (the same bytes the mesh path's uniform buffer carries).
        let uniform = self.build_mesh_uniform(submission.model);
        let mesh_state = self.meshes[mesh_id].as_ref().unwrap();
        let uniform_bytes = mesh_state.uniform_buffer;
        let contents = unsafe {
            self.loader.send0(uniform_bytes, self.loader.sel("contents"))
        };
        if contents.is_null() {
            self.violations += 1;
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                uniform.as_ptr(),
                contents as *mut u8,
                uniform.len(),
            );
        }
        let (vertex_buffer, index_buffer, index_count) = (
            mesh_state.vertex_buffer,
            mesh_state.index_buffer,
            mesh_state.index_count,
        );
        let depth_state = self
            .mesh_resources
            .as_ref()
            .map(|r| r.depth_state)
            .unwrap_or(std::ptr::null_mut());
        let sampler = self
            .mesh_resources
            .as_ref()
            .map(|r| r.sampler)
            .unwrap_or(std::ptr::null_mut());
        let texture = self
            .bound_texture
            .and_then(|i| self.textures_2d.get(i))
            .filter(|t| !t.released)
            .map(|t| t.texture)
            .or_else(|| {
                self.default_texture
                    .and_then(|i| self.textures_2d.get(i))
                    .filter(|t| !t.released)
                    .map(|t| t.texture)
            });
        let texture = match texture {
            Some(texture) => texture,
            None => {
                self.violations += 1;
                return;
            }
        };
        let command_buffer = self.frame_command_buffer;
        let gbuffer_pipeline = self.gbuffer_pipeline;
        let lighting_pipeline = self.lighting_pipeline;
        let lights_buffer = self.lighting_lights_buffer;
        let gbuffer_views: Vec<_> = {
            let target = &self.targets[target_index];
            let views: Vec<_> = target
                .views
                .iter()
                .take(3)
                .map(|v| v.texture)
                .collect();
            views
        };
        if gbuffer_views.len() != 3 {
            self.violations += 1;
            return;
        }
        // ---- Step one: the geometry pass ----
        let pass_descriptor = self.gbuffer_pass_descriptor(target_index);
        if pass_descriptor.is_null() {
            return;
        }
        let encoder = unsafe {
            self.loader.send_ptr_arg(
                command_buffer,
                self.loader.sel("renderCommandEncoderWithDescriptor:"),
                pass_descriptor,
            )
        };
        if encoder.is_null() {
            self.violations += 1;
            return;
        }
        unsafe {
            // The G-buffer pipeline and the mesh path's depth/stencil
            // state (the depth test on).
            self.loader.send_void_ptr(
                encoder,
                self.loader.sel("setRenderPipelineState:"),
                gbuffer_pipeline,
            );
            self.loader.send_void_ptr(
                encoder,
                self.loader.sel("setDepthStencilState:"),
                depth_state,
            );
            // The camera block at the vertex stage's buffer index 0 (the
            // gbuffer vertex shader's `buffer(0)` qualifier).
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setVertexBuffer:offset:atIndex:"),
                uniform_bytes,
                0,
                0,
            );
            // The vertex buffer at buffer index 1 (the vertex
            // descriptor's attributes point there).
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setVertexBuffer:offset:atIndex:"),
                vertex_buffer,
                0,
                1,
            );
            // The bound texture and the sampler at the fragment stage's
            // slot 0 (the gbuffer fragment shader's `texture(0)` and
            // `sampler(0)` qualifiers).
            self.loader.send_void_ptr_ul(
                encoder,
                self.loader.sel("setFragmentTexture:atIndex:"),
                texture,
                0,
            );
            self.loader.send_void_ptr_ul(
                encoder,
                self.loader.sel("setFragmentSamplerState:atIndex:"),
                sampler,
                0,
            );
            // The indexed draw: the same call the mesh path carries.
            self.loader.send_void_ul3_ptr_ul(
                encoder,
                self.loader.sel(
                    "drawIndexedPrimitives:indexCount:indexType:\
                     indexBuffer:indexBufferOffset:",
                ),
                MTLPT_TRIANGLE,
                index_count as super::mtl_loader::NsUInteger,
                MTLIT_UINT32,
                index_buffer,
                0,
            );
            self.loader.send_void(encoder, self.loader.sel("endEncoding"));
        }
        // ---- Step two: the lighting pass ----
        // The default framebuffer's pass descriptor: the LOAD load action
        // keeps the clear's pixels (the same rule the mesh pass
        // descriptor's descriptor records).
        let lighting_pass_descriptor = self.mesh_pass_descriptor();
        if lighting_pass_descriptor.is_null() {
            return;
        }
        let lighting_encoder = unsafe {
            self.loader.send_ptr_arg(
                command_buffer,
                self.loader.sel("renderCommandEncoderWithDescriptor:"),
                lighting_pass_descriptor,
            )
        };
        if lighting_encoder.is_null() {
            self.violations += 1;
            return;
        }
        unsafe {
            // The lighting pipeline binds; no depth/stencil state (the
            // depth test off — the lighting pass's triangle covers the
            // frame and the depth attachment stays untouched).
            self.loader.send_void_ptr(
                lighting_encoder,
                self.loader.sel("setRenderPipelineState:"),
                lighting_pipeline,
            );
            // The lights' buffer at the fragment stage's buffer index 1
            // (the lighting fragment shader's `buffer(1)` qualifier).
            self.loader.send_void_ptr_ul2(
                lighting_encoder,
                self.loader.sel("setFragmentBuffer:offset:atIndex:"),
                lights_buffer,
                0,
                1,
            );
            // The three G-buffer attachments' textures at the fragment
            // stage's slots 0, 1 and 2 (the albedo, the normal, the world
            // position — the lighting fragment shader's `texture(0)`,
            // `texture(1)` and `texture(2)` qualifiers).
            self.loader.send_void_ptr_ul(
                lighting_encoder,
                self.loader.sel("setFragmentTexture:atIndex:"),
                gbuffer_views[0],
                0,
            );
            self.loader.send_void_ptr_ul(
                lighting_encoder,
                self.loader.sel("setFragmentTexture:atIndex:"),
                gbuffer_views[1],
                1,
            );
            self.loader.send_void_ptr_ul(
                lighting_encoder,
                self.loader.sel("setFragmentTexture:atIndex:"),
                gbuffer_views[2],
                2,
            );
            // The sampler at the fragment stage's slot 0 (the lighting
            // fragment shader's `sampler(0)` qualifier).
            self.loader.send_void_ptr_ul(
                lighting_encoder,
                self.loader.sel("setFragmentSamplerState:atIndex:"),
                sampler,
                0,
            );
            // The fullscreen triangle: the three vertices the vertex id
            // generates (no vertex buffer, no index buffer).
            self.loader.send_void_ul3(
                lighting_encoder,
                self.loader
                    .sel("drawPrimitives:vertexStart:vertexCount:"),
                MTLPT_TRIANGLE,
                0,
                3,
            );
            self.loader.send_void(
                lighting_encoder,
                self.loader.sel("endEncoding"),
            );
        }
    }
    /// The post-processing chain (M7): the two internal steps the
    /// contract's submit_post_chain carries. Step one (the geometry pass)
    /// opens the HDR target's pass descriptor (the two Rgba16F colour
    /// attachments plus the depth attachment, each with the clear load
    /// action), binds the post geometry pipeline and draws the mesh into
    /// the radiance and the world attachments. Step two (the composite
    /// pass) opens the default framebuffer's pass descriptor, binds the
    /// composite pipeline and draws the fullscreen triangle that applies
    /// the enabled effects onto the display's range.
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
        if mesh_id >= self.meshes.len() || self.meshes[mesh_id].is_none() {
            self.violations += 1;
            return;
        }
        if !self.mesh_ready || !self.ensure_post_resources() {
            return;
        }
        if !self.frame_open {
            // The post chain runs inside the frame's open command buffer
            // (the clear() the frame's first step runs opened it); a
            // missing frame is a violation.
            self.violations += 1;
            return;
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
        if !self.post_lights_buffer.is_null() {
            let contents = unsafe {
                self.loader.send0(
                    self.post_lights_buffer,
                    self.loader.sel("contents"),
                )
            };
            if !contents.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &lights_block as *const LightsUniform as *const u8,
                        contents as *mut u8,
                        std::mem::size_of::<LightsUniform>(),
                    );
                }
            }
        }
        // The config block: the three vec4 params the composite shader's
        // Config block carries (the toggles, the bloom's parameters, the
        // focus's parameters).
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
        if !self.post_config_buffer.is_null() {
            let contents = unsafe {
                self.loader.send0(
                    self.post_config_buffer,
                    self.loader.sel("contents"),
                )
            };
            if !contents.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        config_block.as_ptr() as *const u8,
                        contents as *mut u8,
                        48,
                    );
                }
            }
        }
        // The volume's block: the origin, the dims, the cell size, the
        // indirect intensity and the toggle the fragment stage's
        // [[buffer(2)]] reads (the 48-byte std140 block).
        let volume_block: [f32; 12] = [
            self.volume_uniform[0],
            self.volume_uniform[1],
            self.volume_uniform[2],
            0.0,
            self.volume_uniform[4],
            self.volume_uniform[5],
            self.volume_uniform[6],
            0.0,
            self.volume_uniform[8],
            config.indirect_intensity,
            if config.indirect_light { 1.0f32 } else { 0.0 },
            0.0,
        ];
        if !self.post_volume_buffer.is_null() {
            let contents = unsafe {
                self.loader.send0(
                    self.post_volume_buffer,
                    self.loader.sel("contents"),
                )
            };
            if !contents.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        volume_block.as_ptr() as *const u8,
                        contents as *mut u8,
                        volume_block.len() * 4,
                    );
                }
            }
        }
        // The camera block: the view-projection matrix from the
        // set_camera state and the model matrix the submission carries
        // (the same bytes the mesh path's uniform buffer carries).
        let uniform = self.build_mesh_uniform(submission.model);
        let mesh_state = self.meshes[mesh_id].as_ref().unwrap();
        let uniform_bytes = mesh_state.uniform_buffer;
        let contents = unsafe {
            self.loader.send0(uniform_bytes, self.loader.sel("contents"))
        };
        if contents.is_null() {
            self.violations += 1;
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                uniform.as_ptr(),
                contents as *mut u8,
                uniform.len(),
            );
        }
        let (vertex_buffer, index_buffer, index_count) = (
            mesh_state.vertex_buffer,
            mesh_state.index_buffer,
            mesh_state.index_count,
        );
        let depth_state = self
            .mesh_resources
            .as_ref()
            .map(|r| r.depth_state)
            .unwrap_or(std::ptr::null_mut());
        let sampler = self
            .mesh_resources
            .as_ref()
            .map(|r| r.sampler)
            .unwrap_or(std::ptr::null_mut());
        let texture = self
            .bound_texture
            .and_then(|i| self.textures_2d.get(i))
            .filter(|t| !t.released)
            .map(|t| t.texture)
            .or_else(|| {
                self.default_texture
                    .and_then(|i| self.textures_2d.get(i))
                    .filter(|t| !t.released)
                    .map(|t| t.texture)
            });
        let texture = match texture {
            Some(texture) => texture,
            None => {
                self.violations += 1;
                return;
            }
        };
        let command_buffer = self.frame_command_buffer;
        let post_gbuffer_pipeline = self.post_gbuffer_pipeline;
        let post_composite_pipeline = self.post_composite_pipeline;
        let post_lights_buffer = self.post_lights_buffer;
        let post_config_buffer = self.post_config_buffer;
        let post_volume_buffer = self.post_volume_buffer;
        let volume_texture = self
            .light_volume
            .as_ref()
            .map(|t| t.texture)
            .unwrap_or(std::ptr::null_mut());
        let hdr_views: Vec<_> = {
            let target = &self.targets[target_index];
            let views: Vec<_> = target
                .views
                .iter()
                .take(2)
                .map(|v| v.texture)
                .collect();
            views
        };
        if hdr_views.len() != 2 {
            self.violations += 1;
            return;
        }
        // ---- Step one: the geometry pass ----
        // The HDR target's pass descriptor: the two colour attachments
        // (the radiance and the world position) plus the depth attachment,
        // each with the clear load action (the world attachment clears to
        // zero — the alpha 0 marks the background).
        let pass_descriptor = self.hdr_pass_descriptor(target_index);
        if pass_descriptor.is_null() {
            return;
        }
        // The volume's toggle with no volume set: the protocol
        // violation the contract's set_light_volume rule carries.
        if config.indirect_light && volume_texture.is_null() {
            self.violations += 1;
            return;
        }
        let encoder = unsafe {
            self.loader.send_ptr_arg(
                command_buffer,
                self.loader.sel("renderCommandEncoderWithDescriptor:"),
                pass_descriptor,
            )
        };
        if encoder.is_null() {
            self.violations += 1;
            return;
        }
        unsafe {
            // The post geometry pipeline and the mesh path's depth/stencil
            // state (the depth test on).
            self.loader.send_void_ptr(
                encoder,
                self.loader.sel("setRenderPipelineState:"),
                post_gbuffer_pipeline,
            );
            self.loader.send_void_ptr(
                encoder,
                self.loader.sel("setDepthStencilState:"),
                depth_state,
            );
            // The camera block at the vertex stage's buffer index 0 (the
            // gbuffer vertex shader's `buffer(0)` qualifier).
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setVertexBuffer:offset:atIndex:"),
                uniform_bytes,
                0,
                0,
            );
            // The vertex buffer at buffer index 1 (the vertex
            // descriptor's attributes point there).
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setVertexBuffer:offset:atIndex:"),
                vertex_buffer,
                0,
                1,
            );
            // The lights' buffer at the fragment stage's buffer index 1
            // (the post geometry fragment shader's `buffer(1)` qualifier).
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setFragmentBuffer:offset:atIndex:"),
                post_lights_buffer,
                0,
                1,
            );
            // The volume's block at the fragment stage's buffer index 2
            // (the post geometry fragment shader's `buffer(2)` qualifier).
            self.loader.send_void_ptr_ul2(
                encoder,
                self.loader.sel("setFragmentBuffer:offset:atIndex:"),
                post_volume_buffer,
                0,
                2,
            );
            // The bound texture and the sampler at the fragment stage's
            // slot 0 (the post geometry fragment shader's `texture(0)` and
            // `sampler(0)` qualifiers).
            self.loader.send_void_ptr_ul(
                encoder,
                self.loader.sel("setFragmentTexture:atIndex:"),
                texture,
                0,
            );
            self.loader.send_void_ptr_ul(
                encoder,
                self.loader.sel("setFragmentSamplerState:atIndex:"),
                sampler,
                0,
            );
            // The volume's 3D texture at the fragment stage's slot 1 (the
            // post geometry fragment shader's `texture(1)` qualifier).
            self.loader.send_void_ptr_ul(
                encoder,
                self.loader.sel("setFragmentTexture:atIndex:"),
                volume_texture,
                1,
            );
            // The indexed draw: the same call the mesh path carries.
            self.loader.send_void_ul3_ptr_ul(
                encoder,
                self.loader.sel(
                    "drawIndexedPrimitives:indexCount:indexType:\
                     indexBuffer:indexBufferOffset:",
                ),
                MTLPT_TRIANGLE,
                index_count as super::mtl_loader::NsUInteger,
                MTLIT_UINT32,
                index_buffer,
                0,
            );
            self.loader.send_void(encoder, self.loader.sel("endEncoding"));
        }
        // ---- Step two: the composite pass ----
        // The default framebuffer's pass descriptor: the LOAD load action
        // keeps the clear's pixels (the same rule the mesh pass
        // descriptor's descriptor records).
        let composite_pass_descriptor = self.mesh_pass_descriptor();
        if composite_pass_descriptor.is_null() {
            return;
        }
        let composite_encoder = unsafe {
            self.loader.send_ptr_arg(
                command_buffer,
                self.loader.sel("renderCommandEncoderWithDescriptor:"),
                composite_pass_descriptor,
            )
        };
        if composite_encoder.is_null() {
            self.violations += 1;
            return;
        }
        unsafe {
            // The composite pipeline binds; no depth/stencil state (the
            // depth test off — the composite's triangle covers the frame).
            self.loader.send_void_ptr(
                composite_encoder,
                self.loader.sel("setRenderPipelineState:"),
                post_composite_pipeline,
            );
            // The config block at the fragment stage's buffer index 0
            // (the composite fragment shader's `buffer(0)` qualifier).
            self.loader.send_void_ptr_ul2(
                composite_encoder,
                self.loader.sel("setFragmentBuffer:offset:atIndex:"),
                post_config_buffer,
                0,
                0,
            );
            // The two HDR attachments' textures at the fragment stage's
            // slots 0 and 1 (the composite fragment shader's `texture(0)`
            // and `texture(1)` qualifiers).
            self.loader.send_void_ptr_ul(
                composite_encoder,
                self.loader.sel("setFragmentTexture:atIndex:"),
                hdr_views[0],
                0,
            );
            self.loader.send_void_ptr_ul(
                composite_encoder,
                self.loader.sel("setFragmentTexture:atIndex:"),
                hdr_views[1],
                1,
            );
            // The sampler at the fragment stage's slot 0 (the composite
            // fragment shader's `sampler(0)` qualifier).
            self.loader.send_void_ptr_ul(
                composite_encoder,
                self.loader.sel("setFragmentSamplerState:atIndex:"),
                sampler,
                0,
            );
            // The fullscreen triangle: the three vertices the vertex id
            // generates (no vertex buffer, no index buffer).
            self.loader.send_void_ul3(
                composite_encoder,
                self.loader
                    .sel("drawPrimitives:vertexStart:vertexCount:"),
                MTLPT_TRIANGLE,
                0,
                3,
            );
            self.loader.send_void(
                composite_encoder,
                self.loader.sel("endEncoding"),
            );
        }
    }
}
