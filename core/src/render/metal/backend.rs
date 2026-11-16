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
    MtlLoader, MtlOrigin, MtlSize,
    MTLPX_RGBA8_UNORM, MTLPX_RGBA16_FLOAT, MTLPX_R8_UNORM,
    MTLPX_DEPTH24_UNORM_STENCIL8, MTLPX_DEPTH32_FLOAT_STENCIL8,
    MTLVF_FLOAT3, MTLSF_PER_VERTEX, MTLSM_SHARED, MTLSM_PRIVATE,
    MTLCU_RENDER_TARGET, MTLCU_SHADER_READ,
};
use super::super::{
    bytes_per_pixel, CameraParams, ClearBuffer, ClearColor, DrawSubmission,
    GpuMesh, GpuMeshInner, PixelFormat, ReadbackSample, RenderBackend,
    RenderError, RenderTarget, RenderTargetDesc, RenderTargetInner,
    ShaderProgram, ShaderProgramInner, Texture2DDesc, TextureView,
    TextureViewInner, VertexLayout, DEPTH_ATTACHMENT,
};
use crate::math::{Mat4, Vec3};
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
    fn compile_library(
        &mut self,
        vertex_source: &str,
        fragment_source: &str,
    ) -> Option<super::mtl_loader::MTLLibraryHandle> {
        let mut source = String::with_capacity(
            vertex_source.len() + fragment_source.len() + 1,
        );
        source.push_str(vertex_source);
        source.push_str(fragment_source);
        let source_string = self.make_ns_string(&source);
        if source_string.is_null() {
            return None;
        }
        let options = unsafe {
            self.loader
                .alloc_init(self.loader.cls_mtl_compile_options)
        };
        let library = unsafe {
            self.loader.send_ptr_arg(
                self.device,
                self.loader.sel("compileSource:options:"),
                source_string,
            )
        };
        let _ = options;
        if library.is_null() {
            None
        } else {
            Some(library)
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

    /// The diagnostic triangle's pipeline: the MSL library compiles, the
    /// vertex descriptor carries the Float3 attribute, the colour
    /// attachment's pixel format follows the frame target's.
    fn ensure_triangle_resources(&mut self) -> bool {
        if self.tri_ready {
            return true;
        }
        let library = match self.compile_library(
            super::shader_library::TRIANGLE_VERTEX_SHADER,
            super::shader_library::TRIANGLE_FRAGMENT_SHADER,
        ) {
            Some(library) => library,
            None => return false,
        };
        let vertex_function = match self.new_function(library, "vertexMain") {
            Some(function) => function,
            None => return false,
        };
        let fragment_function =
            match self.new_function(library, "fragmentMain") {
                Some(function) => function,
                None => return false,
            };
        let descriptor = unsafe {
            self.loader
                .alloc_init(self.loader.cls_mtl_render_pipeline_descriptor)
        };
        if descriptor.is_null() {
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
        // buffer index 0, per-vertex.
        let vertex_descriptor = unsafe {
            self.loader.alloc_init(self.loader.cls_mtl_vertex_descriptor)
        };
        if vertex_descriptor.is_null() {
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
                0,
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
                0,
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
            // target's (the off-screen target's RGBA8 unorm).
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setColorAttachmentPixelFormat:"),
                MTLPX_RGBA8_UNORM,
            );
            // The depth attachment's pixel format (the depth/stencil
            // format the device picked).
            self.loader.send_void_ul(
                descriptor,
                self.loader.sel("setDepthAttachmentPixelFormat:"),
                self.depth_format,
            );
        }
        let pipeline = unsafe {
            self.loader.send_ptr_arg(
                self.device,
                self.loader.sel("newRenderPipelineStateWithDescriptor:"),
                descriptor,
            )
        };
        if pipeline.is_null() {
            return false;
        }
        // The triangle's uniform buffer: the shared-storage buffer the
        // draw binds (the same staging rule the other backends follow).
        let uniform = self.build_tri_uniform();
        let buffer = self.create_shared_buffer(&uniform);
        if buffer.is_null() {
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
    fn draw_triangle(&mut self) {
        if self.shutdown || !self.tri_ready {
            return;
        }
        // The frame's command buffer and encoder exist only inside the
        // frame's clear pass (the next milestones' release path wires the
        // full pass); the resources are ensured once here.
        let _ = self.tri_pipeline;
        let _ = self.tri_uniform_buffer;
    }

    /// The frame's clear: the render pass's load actions carry the clear
    /// values when the frame's command buffer runs.
    fn clear_frame(&mut self) {
        // The recorded state lives until the frame's encode; the pass's
        // load actions read it then.
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
        let copy = self.loader.sel(
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:\
             sourceSize:toBuffer:destinationOffset:\
             destinationBytesPerRow:destinationBytesPerImage:",
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
            );
            // The depth copy: the same one pixel, the depth/stencil
            // plane's packed word landing at offset 16.
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
        // The depth/stencil word: the packed 4-byte texel. The 24/8
        // format's depth lives in the top three bytes, the stencil in the
        // low byte; the 32F/8 format's depth is a float and the stencil
        // stays 0 (the same rule the Vulkan readback records).
        let depth_stencil = u32::from_le_bytes([
            bytes[16],
            bytes[17],
            bytes[18],
            bytes[19],
        ]);
        let (depth, stencil) = if self.depth_format
            == MTLPX_DEPTH24_UNORM_STENCIL8
        {
            let depth24 = depth_stencil >> 8;
            let depth = depth24 as f32 / 16_777_215.0;
            let stencil = (depth_stencil & 0xFF) as i32;
            (depth, stencil)
        } else {
            let depth = f32::from_bits(depth_stencil);
            (depth, 0)
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
        self.clear_frame();
        if self.config.draw_triangle {
            self.ensure_triangle_resources();
            self.draw_triangle();
        }
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
        Ok(GpuMesh {
            inner: Arc::new(Mutex::new(GpuMeshInner {
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
        }
    }

    fn submit(&mut self, submission: &DrawSubmission) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        let inner = submission.mesh.inner.lock().unwrap();
        if !inner.uploaded {
            self.violations += 1;
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

    fn bind_texture(&mut self, _view: Option<&TextureView>, _unit: u32) {
        if self.shutdown {
            self.violations += 1;
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
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "fill_texture_2d: expected {} bytes, got {}",
                expected,
                data.len()
            )));
        }
        // The upload path lands in the next milestones (the release path
        // wires the blit copy); the size validation is the contract's
        // rule here.
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
}
