//! The OpenGL render backend: the render contract's second live
//! implementation, on the platform's EGL + GL 3.3 core stack.
//!
//! RULE (lesson 3): the engine's own loader, own code, no third-party
//! binding. The `GlLoader` resolves the platform's libEGL.so.1 and the GL
//! core entry points; the backend never links a GL binding library.
//!
//! The headless path runs through an EGL pbuffer surface (the platform
//! surfaceless device when the implementation offers it), the same
//! off-screen rule the Vulkan backend follows: the frame renders into the
//! off-screen surface and the readback copies the pixels back through
//! `glReadPixels`. The frame flow matches the Vulkan backend's: `clear()`
//! opens the frame (the clear plus the diagnostic draw), `swap_buffers()`
//! runs the readback before the frame ends.

use super::super::{
    bytes_per_pixel, CameraParams, ClearBuffer, ClearColor, GpuMesh,
    GpuMeshInner, PixelFormat, ReadbackSample, RenderBackend, RenderError,
    RenderTarget, RenderTargetDesc, RenderTargetInner, ShaderProgram,
    ShaderProgramInner, Texture2DDesc, TextureView, TextureViewInner,
    VertexLayout, DEPTH_ATTACHMENT,
};
use super::gl_loader::{GlLoader, *};
use super::shader_library::{TRIANGLE_FRAGMENT_SHADER, TRIANGLE_VERTEX_SHADER};
use crate::math::{Mat4, Vec3};
use std::sync::{Arc, Mutex};
use std::os::raw::{c_char, c_void};

/// The OpenGL backend's configuration. The same fields the Vulkan backend's
/// config carries: the frame size, the diagnostic draw, the clear values
/// and the readback switch.
#[derive(Clone, Debug)]
pub struct OpenGlBackendConfig {
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

impl Default for OpenGlBackendConfig {
    fn default() -> Self {
        OpenGlBackendConfig {
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

/// The OpenGL backend's own texture: a GL texture object plus the
/// contract's shared handle state.
// The texture object is written at creation; the release path (the next
// milestones) deletes it.
#[allow(dead_code)]
struct OpenGlTexture2D {
    inner: Arc<Mutex<TextureViewInner>>,
    texture: u32,
    released: bool,
}

/// The OpenGL backend's own render target: a GL framebuffer object plus the
/// contract's shared handle state.
// The framebuffer object is written at creation; the release path (the
// next milestones) deletes it.
#[allow(dead_code)]
struct OpenGlRenderTarget {
    inner: Arc<Mutex<RenderTargetInner>>,
    views: Vec<OpenGlTexture2D>,
    depth_view: Option<OpenGlTexture2D>,
    framebuffer: u32,
    released: bool,
}

/// The diagnostic triangle's uniform block: the MVP matrix and the colour.
/// The engine's Mat4 stores column-major (the convention the C++ engine
/// established); the GL `uniformMatrix4fv` call uploads it as-is (the GL
/// reads column-major by default, transpose = false).
#[repr(C)]
struct TriUniform {
    mvp: [[f32; 4]; 4],
    color: [f32; 4],
}

/// The headless render backend.
// The library handle and the diagnostic-triangle bookkeeping fields are
// written at setup; the release path (the next milestones) reads them.
#[allow(dead_code)]
pub struct OpenGlBackend {
    config: OpenGlBackendConfig,
    loader: GlLoader,
    library: *mut std::os::raw::c_void,
    // The EGL surface objects.
    display: *mut std::os::raw::c_void,
    egl_config: *mut std::os::raw::c_void,
    surface: *mut std::os::raw::c_void,
    context: *mut std::os::raw::c_void,
    // The GL identity the describe() reports.
    gl_version: String,
    gl_vendor: String,
    gl_renderer: String,
    // The default framebuffer's colour attachment (the pbuffer's plane).
    // The default framebuffer owns the pbuffer's plane directly.
    // The diagnostic triangle resources.
    tri_mesh: Option<GpuMesh>,
    tri_shader: Option<ShaderProgram>,
    tri_vao: u32,
    tri_vbo: u32,
    tri_program: u32,
    tri_uniform_mvp: i32,
    tri_uniform_color: i32,
    tri_ready: bool,
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
    targets: Vec<OpenGlRenderTarget>,
    textures_2d: Vec<OpenGlTexture2D>,
    readback_queue: Vec<(u32, u32)>,
    last_readback: Vec<ReadbackSample>,
    has_center: bool,
    center: ClearColor,
    center_depth: f32,
    center_stencil: i32,
}

impl OpenGlBackend {
    /// Creates the backend. Throws `RenderError` when the platform's EGL
    /// stack cannot be loaded or the GL 3.3 core context cannot be created.
    pub fn new(config: OpenGlBackendConfig) -> Result<Self, RenderError> {
        if config.width == 0 || config.height == 0 {
            return Err(RenderError::from(
                "OpenGlBackend: width and height must be positive",
            ));
        }
        let library = super::gl_loader::load_library().map_err(RenderError::from)?;
        let mut loader = GlLoader::new();
        let mut missing = String::new();
        if !loader.load(library, &mut missing) {
            return Err(RenderError::from(format!(
                "OpenGlBackend: the loader could not resolve {}",
                missing
            )));
        }
        let clear_depth = config.clear_depth;
        let clear_stencil = config.clear_stencil;
        let mut backend = OpenGlBackend {
            config,
            loader,
            library,
            display: EGL_NO_DISPLAY,
            egl_config: std::ptr::null_mut(),
            surface: EGL_NO_SURFACE,
            context: EGL_NO_CONTEXT,
            gl_version: String::new(),
            gl_vendor: String::new(),
            gl_renderer: String::new(),
            tri_mesh: None,
            tri_shader: None,
            tri_vao: 0,
            tri_vbo: 0,
            tri_program: 0,
            tri_uniform_mvp: -1,
            tri_uniform_color: -1,
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

    /// The EGL surface and the GL 3.3 core context creation. The
    /// surfaceless path runs through the MESA surfaceless platform (the
    /// platform headless one); the context request is the GL 3.3 core
    /// profile, the same profile the C++ engine's backend requests.
    fn initialise(&mut self) -> Result<(), RenderError> {
        if self.shutdown {
            return Err(RenderError::from(
                "OpenGlBackend::initialise: the backend is shut down",
            ));
        }
        // The platform's surfaceless EGL display (the headless path).
        let display = unsafe {
            (self.loader.egl_get_platform_display_ext.unwrap())(
                EGL_PLATFORM_SURFACELESS_MESA,
                EGL_DEFAULT_DISPLAY,
                [EGL_NONE as i32, EGL_NONE as i32].as_ptr(),
            )
        };
        if display.is_null() {
            return Err(RenderError::from(
                "OpenGlBackend: the platform's EGL does not offer a \
                 surfaceless display",
            ));
        }
        let mut major: u32 = 0;
        let mut minor: u32 = 0;
        let ok = unsafe {
            (self.loader.egl_initialize.unwrap())(
                display,
                &mut major,
                &mut minor,
            )
        };
        if ok != EGL_TRUE_U32 {
            return Err(RenderError::from(
                "OpenGlBackend: eglInitialize failed",
            ));
        }
        // The client API is desktop OpenGL (the surfaceless device carries
        // the platform's GL implementation).
        if unsafe { (self.loader.egl_bind_api.unwrap())(EGL_OPENGL_API) }
            != EGL_TRUE_U32
        {
            return Err(RenderError::from(
                "OpenGlBackend: eglBindAPI(EGL_OPENGL_API) failed",
            ));
        }
        // The config request: a pbuffer-capable OpenGL config with the
        // depth/stencil/alpha planes the contract's clear and readback
        // path needs (24/8/8, the same sizes the C++ backend requests).
        let attribs: [i32; 11] = [
            EGL_SURFACE_TYPE as i32,
            EGL_PBUFFER_BIT as i32,
            EGL_RENDERABLE_TYPE as i32,
            EGL_OPENGL_BIT as i32,
            EGL_DEPTH_SIZE as i32,
            24,
            EGL_STENCIL_SIZE as i32,
            8,
            EGL_ALPHA_SIZE as i32,
            8,
            EGL_NONE as i32,
        ];
        let mut egl_config: *mut std::os::raw::c_void = std::ptr::null_mut();
        let mut num_configs: i32 = 0;
        let ok = unsafe {
            (self.loader.egl_choose_config.unwrap())(
                display,
                attribs.as_ptr(),
                &mut egl_config,
                1,
                &mut num_configs,
            )
        };
        if ok != EGL_TRUE_U32 || num_configs == 0 || egl_config.is_null() {
            return Err(RenderError::from(
                "OpenGlBackend: eglChooseConfig found no GL config",
            ));
        }
        // The pbuffer surface: the off-screen frame the backend renders
        // into (the same off-screen rule the Vulkan backend follows).
        let surface_attribs: [i32; 5] = [
            EGL_WIDTH as i32,
            self.config.width as i32,
            EGL_HEIGHT as i32,
            self.config.height as i32,
            EGL_NONE as i32,
        ];
        let surface = unsafe {
            (self.loader.egl_create_pbuffer_surface.unwrap())(
                display,
                egl_config,
                surface_attribs.as_ptr(),
            )
        };
        if surface.is_null() {
            return Err(RenderError::from(
                "OpenGlBackend: eglCreatePbufferSurface failed",
            ));
        }
        // The GL 3.3 core context request (the same profile the C++
        // backend requests).
        let context_attribs: [i32; 9] = [
            EGL_CONTEXT_MAJOR_VERSION as i32,
            3,
            EGL_CONTEXT_MINOR_VERSION as i32,
            3,
            EGL_CONTEXT_OPENGL_PROFILE_MASK as i32,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT as i32,
            EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE as i32,
            EGL_TRUE as i32,
            EGL_NONE as i32,
        ];
        let context = unsafe {
            (self.loader.egl_create_context.unwrap())(
                display,
                egl_config,
                EGL_NO_CONTEXT,
                context_attribs.as_ptr(),
            )
        };
        if context.is_null() {
            return Err(RenderError::from(
                "OpenGlBackend: the GL 3.3 core context creation failed",
            ));
        }
        if unsafe {
            (self.loader.egl_make_current.unwrap())(
                display,
                surface,
                surface,
                context,
            )
        } != EGL_TRUE_U32
        {
            return Err(RenderError::from(
                "OpenGlBackend: eglMakeCurrent failed",
            ));
        }
        self.display = display;
        self.egl_config = egl_config;
        self.surface = surface;
        self.context = context;
        // The GL identity the describe() reports.
        unsafe {
            let version = (self.loader.gl_get_string.unwrap())(GL_VERSION);
            let vendor = (self.loader.gl_get_string.unwrap())(GL_VENDOR);
            let renderer = (self.loader.gl_get_string.unwrap())(GL_RENDERER);
            if !version.is_null() {
                self.gl_version = c_string(version);
            }
            if !vendor.is_null() {
                self.gl_vendor = c_string(vendor);
            }
            if !renderer.is_null() {
                self.gl_renderer = c_string(renderer);
            }
        }
        // The viewport follows the frame size (the same full-frame rule
        // the C++ backend's applyViewport follows).
        unsafe {
            (self.loader.gl_viewport.unwrap())(
                0,
                0,
                self.config.width as i32,
                self.config.height as i32,
            );
        }
        self.frame_open = true;
        Ok(())
    }

    /// The GL pixel format mapping (the contract's enum onto the GL
    /// internal/storage format pair).
    fn to_gl_format(&self, format: PixelFormat) -> (i32, u32) {
        match format {
            PixelFormat::Rgba8 => (GL_RGBA8 as i32, GL_RGBA),
            PixelFormat::Rgba16F => (GL_RGBA16F as i32, GL_RGBA),
            PixelFormat::R8 => (GL_R8 as i32, GL_RED),
            PixelFormat::Depth24Stencil8 => (GL_DEPTH24_STENCIL8 as i32, GL_DEPTH_STENCIL),
        }
    }

    /// The diagnostic triangle's GL resources: the mesh, the shader program
    /// and the pipeline state (the same path the Vulkan backend's
    /// `ensure_triangle_pipeline` records). The resources are created on
    /// the first draw and reused.
    fn ensure_triangle_resources(&mut self) -> bool {
        if self.tri_ready {
            return true;
        }
        // The shader program: the same flat-unlit pair the shader library
        // carries, compiled at runtime by the driver (the OpenGL path's
        // rule: no build-time SPIR-V step).
        let program = unsafe {
            (self.loader.gl_create_program.unwrap())()
        };
        if program == 0 {
            return false;
        }
        let vs = unsafe { (self.loader.gl_create_shader.unwrap())(GL_VERTEX_SHADER) };
        let fs = unsafe {
            (self.loader.gl_create_shader.unwrap())(GL_FRAGMENT_SHADER)
        };
        if vs == 0 || fs == 0 {
            return false;
        }
        let vs_source = [TRIANGLE_VERTEX_SHADER.as_ptr() as *const c_char];
        let vs_len = [TRIANGLE_VERTEX_SHADER.len() as i32];
        unsafe {
            (self.loader.gl_shader_source.unwrap())(
                vs,
                1,
                vs_source.as_ptr(),
                vs_len.as_ptr(),
            );
            (self.loader.gl_compile_shader.unwrap())(vs);
        }
        let fs_source = [TRIANGLE_FRAGMENT_SHADER.as_ptr() as *const c_char];
        let fs_len = [TRIANGLE_FRAGMENT_SHADER.len() as i32];
        unsafe {
            (self.loader.gl_shader_source.unwrap())(
                fs,
                1,
                fs_source.as_ptr(),
                fs_len.as_ptr(),
            );
            (self.loader.gl_compile_shader.unwrap())(fs);
            (self.loader.gl_attach_shader.unwrap())(program, vs);
            (self.loader.gl_attach_shader.unwrap())(program, fs);
            (self.loader.gl_link_program.unwrap())(program);
        }
        let mut linked: i32 = 0;
        unsafe {
            (self.loader.gl_get_programiv.unwrap())(
                program,
                GL_LINK_STATUS,
                &mut linked,
            );
        }
        if linked != GL_TRUE as i32 {
            return false;
        }
        unsafe {
            (self.loader.gl_delete_shader.unwrap())(vs);
            (self.loader.gl_delete_shader.unwrap())(fs);
        }
        // The vertex array and the vertex buffer: the triangle's three
        // positions (the same data the C++ engine's ensureTriangle
        // carries).
        let vertices: [f32; 9] = [
            -0.5, -0.5, 0.0, //
            0.5, -0.5, 0.0, //
            0.0, 0.5, 0.0,
        ];
        let mut vao: u32 = 0;
        let mut vbo: u32 = 0;
        unsafe {
            (self.loader.gl_gen_vertex_arrays.unwrap())(1, &mut vao);
            (self.loader.gl_gen_buffers.unwrap())(1, &mut vbo);
            (self.loader.gl_bind_vertex_array.unwrap())(vao);
            (self.loader.gl_bind_buffer.unwrap())(GL_ARRAY_BUFFER, vbo);
            (self.loader.gl_buffer_data.unwrap())(
                GL_ARRAY_BUFFER,
                (vertices.len() * std::mem::size_of::<f32>()) as isize,
                vertices.as_ptr() as *const c_void,
                GL_STATIC_DRAW,
            );
            (self.loader.gl_vertex_attrib_pointer.unwrap())(
                0,
                3,
                GL_FLOAT,
                GL_FALSE,
                3 * std::mem::size_of::<f32>() as i32,
                std::ptr::null(),
            );
            (self.loader.gl_enable_vertex_attrib_array.unwrap())(0);
            (self.loader.gl_bind_vertex_array.unwrap())(0);
            (self.loader.gl_bind_buffer.unwrap())(GL_ARRAY_BUFFER, 0);
        }
        self.tri_program = program;
        self.tri_vao = vao;
        self.tri_vbo = vbo;
        self.tri_uniform_mvp = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"mvp\0".as_ptr() as *const c_char,
            )
        };
        self.tri_uniform_color = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"color\0".as_ptr() as *const c_char,
            )
        };
        self.tri_ready = true;
        true
    }

    /// The diagnostic triangle draw: the same flat-unlit path the Vulkan
    /// backend's diagnostic draw records, on the default framebuffer only.
    fn draw_triangle(&mut self) {
        if self.shutdown {
            return;
        }
        if !self.ensure_triangle_resources() {
            return;
        }
        // The clear path's diagnostic draw happens on the default
        // framebuffer; the triangle never writes depth.
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
        unsafe {
            // The diagnostic triangle never writes depth (the Vulkan
            // pipeline's depth_test_enable is 0, the same rule).
            (self.loader.gl_disable.unwrap())(GL_DEPTH_TEST);
            (self.loader.gl_depth_mask.unwrap())(GL_FALSE);
            (self.loader.gl_use_program.unwrap())(self.tri_program);
            (self.loader.gl_uniform_matrix_4fv.unwrap())(
                self.tri_uniform_mvp,
                1,
                GL_FALSE,
                uniform.mvp.as_ptr() as *const f32,
            );
            (self.loader.gl_uniform_4f.unwrap())(
                self.tri_uniform_color,
                uniform.color[0],
                uniform.color[1],
                uniform.color[2],
                uniform.color[3],
            );
            (self.loader.gl_bind_vertex_array.unwrap())(self.tri_vao);
            (self.loader.gl_draw_arrays.unwrap())(GL_TRIANGLES, 0, 3);
            (self.loader.gl_bind_vertex_array.unwrap())(0);
            // Restore the contract's depth state after the draw.
            if self.depth_test_enabled {
                (self.loader.gl_enable.unwrap())(GL_DEPTH_TEST);
                (self.loader.gl_depth_func.unwrap())(GL_LESS);
                (self.loader.gl_depth_mask.unwrap())(GL_TRUE);
            }
        }
    }

    /// The readback's one pixel: the colour, the depth and the stencil
    /// values the contract's readPixelInto follows. The GL reads
    /// bottom-up; the contract's readback addresses top-down, so the
    /// backend flips the row (the same rule the C++ engine's readback
    /// follows on the GL path).
    fn readback_pixel(&mut self, x: u32, y: u32) {
        let gl_y = self.config.height.saturating_sub(1 + y);
        let mut pixel: [u8; 4] = [0; 4];
        let mut depth: f32 = 1.0;
        let mut stencil: u8 = 0;
        unsafe {
            (self.loader.gl_read_pixels.unwrap())(
                x as i32,
                gl_y as i32,
                1,
                1,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                pixel.as_mut_ptr() as *mut c_void,
            );
            (self.loader.gl_read_pixels.unwrap())(
                x as i32,
                gl_y as i32,
                1,
                1,
                GL_DEPTH_COMPONENT,
                GL_FLOAT,
                &mut depth as *mut f32 as *mut c_void,
            );
            (self.loader.gl_read_pixels.unwrap())(
                x as i32,
                gl_y as i32,
                1,
                1,
                GL_STENCIL_INDEX,
                GL_UNSIGNED_BYTE,
                &mut stencil as *mut u8 as *mut c_void,
            );
        }
        let colour = ClearColor {
            r: pixel[0],
            g: pixel[1],
            b: pixel[2],
            a: pixel[3],
        };
        self.has_center = true;
        self.center = colour;
        self.center_depth = depth;
        self.center_stencil = stencil as i32;
        self.last_readback.push(ReadbackSample {
            x,
            y,
            color: colour,
            depth,
            stencil: stencil as i32,
        });
    }

    /// The frame's clear: the clear plus the diagnostic draw (the same
    /// frame the Vulkan backend's clear records).
    fn clear_frame(&mut self) {
        let mut mask = 0;
        if self.config.clear_buffers.color {
            mask |= GL_COLOR_BUFFER_BIT;
        }
        if self.config.clear_buffers.depth {
            mask |= GL_DEPTH_BUFFER_BIT;
        }
        if self.config.clear_buffers.stencil {
            mask |= GL_STENCIL_BUFFER_BIT;
        }
        let colour = self.clear_color;
        unsafe {
            (self.loader.gl_clear_color.unwrap())(
                colour.r as f32 / 255.0,
                colour.g as f32 / 255.0,
                colour.b as f32 / 255.0,
                colour.a as f32 / 255.0,
            );
            (self.loader.gl_clear_depth.unwrap())(self.clear_depth as f64);
            (self.loader.gl_clear_stencil.unwrap())(self.clear_stencil);
            (self.loader.gl_clear.unwrap())(mask);
        }
        // The depth state follows the contract's switch (the same rule the
        // C++ backend's applyDepthState follows).
        if self.depth_test_enabled {
            unsafe {
                (self.loader.gl_enable.unwrap())(GL_DEPTH_TEST);
                (self.loader.gl_depth_func.unwrap())(GL_LESS);
                (self.loader.gl_depth_mask.unwrap())(GL_TRUE);
            }
        } else {
            unsafe {
                (self.loader.gl_disable.unwrap())(GL_DEPTH_TEST);
                (self.loader.gl_depth_mask.unwrap())(GL_FALSE);
            }
        }
        if self.config.draw_triangle {
            self.draw_triangle();
        }
    }
}

/// Copies a NUL-terminated C string out of a GL string pointer.
fn c_string(pointer: *const u8) -> String {
    unsafe {
        let mut length = 0;
        while *pointer.add(length) != 0 {
            length += 1;
        }
        let bytes =
            std::slice::from_raw_parts(pointer, length);
        String::from_utf8_lossy(bytes).into_owned()
    }
}

impl RenderBackend for OpenGlBackend {
    fn describe(&self) -> String {
        if self.display.is_null() {
            return "opengl (not initialised)".to_string();
        }
        format!(
            "opengl {}; vendor={}; renderer={}",
            self.gl_version, self.gl_vendor, self.gl_renderer
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
        // The clear opens the frame (the clear plus the diagnostic draw),
        // the same frame the Vulkan backend's clear records.
        self.clear_frame();
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
        // The EGL surface objects release (the context and the surface
        // terminate; the library handle stays loaded for the process).
        unsafe {
            if !self.display.is_null() {
                if !self.context.is_null() {
                    (self.loader.egl_destroy_context.unwrap())(
                        self.display,
                        self.context,
                    );
                    self.context = EGL_NO_CONTEXT;
                }
                if !self.surface.is_null() {
                    (self.loader.egl_destroy_surface.unwrap())(
                        self.display,
                        self.surface,
                    );
                    self.surface = EGL_NO_SURFACE;
                }
                (self.loader.egl_terminate.unwrap())(self.display);
                self.display = EGL_NO_DISPLAY;
            }
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
                "upload_mesh: the vertex data is not a multiple of the stride",
            ));
        }
        for attribute in &layout.attributes {
            let end = attribute.offset
                + attribute.components
                    * match attribute.component_type {
                        crate::render::ComponentType::Float32 => 4,
                        crate::render::ComponentType::Uint8Normalized => 1,
                    };
            if end > layout.stride {
                return Err(RenderError::from(
                    "upload_mesh: an attribute runs past the stride",
                ));
            }
        }
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
        }
    }

    fn submit(&mut self, submission: &crate::render::DrawSubmission) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        for light in &submission.lights {
            if light.validate().is_err() {
                self.violations += 1;
            }
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
            let (internal, storage) = self.to_gl_format(desc.color_formats[i]);
            let mut texture: u32 = 0;
            unsafe {
                (self.loader.gl_gen_textures.unwrap())(1, &mut texture);
                (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, texture);
                (self.loader.gl_tex_image_2d.unwrap())(
                    GL_TEXTURE_2D,
                    0,
                    internal,
                    desc.width as i32,
                    desc.height as i32,
                    0,
                    storage,
                    GL_UNSIGNED_BYTE,
                    std::ptr::null(),
                );
                (self.loader.gl_tex_parameter_i.unwrap())(
                    GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER,
                    GL_NEAREST as i32,
                );
                (self.loader.gl_tex_parameter_i.unwrap())(
                    GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER,
                    GL_NEAREST as i32,
                );
                (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, 0);
            }
            views.push(OpenGlTexture2D {
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
            let mut texture: u32 = 0;
            unsafe {
                (self.loader.gl_gen_textures.unwrap())(1, &mut texture);
                (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, texture);
                (self.loader.gl_tex_image_2d.unwrap())(
                    GL_TEXTURE_2D,
                    0,
                    GL_DEPTH24_STENCIL8 as i32,
                    desc.width as i32,
                    desc.height as i32,
                    0,
                    GL_DEPTH_STENCIL,
                    GL_UNSIGNED_INT_24_8,
                    std::ptr::null(),
                );
                (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, 0);
            }
            Some(OpenGlTexture2D {
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
        // The framebuffer object: the colour and the depth attachments.
        let mut framebuffer: u32 = 0;
        unsafe {
            (self.loader.gl_gen_framebuffers.unwrap())(1, &mut framebuffer);
            (self.loader.gl_bind_framebuffer.unwrap())(GL_FRAMEBUFFER, framebuffer);
            for view in &views {
                (self.loader.gl_framebuffer_texture_2d.unwrap())(
                    GL_FRAMEBUFFER,
                    GL_COLOR_ATTACHMENT0,
                    GL_TEXTURE_2D,
                    view.texture,
                    0,
                );
            }
            if let Some(depth) = &depth_view {
                (self.loader.gl_framebuffer_texture_2d.unwrap())(
                    GL_FRAMEBUFFER,
                    GL_DEPTH_ATTACHMENT,
                    GL_TEXTURE_2D,
                    depth.texture,
                    0,
                );
            }
            let status =
                (self.loader.gl_check_framebuffer_status.unwrap())(GL_FRAMEBUFFER);
            (self.loader.gl_bind_framebuffer.unwrap())(GL_FRAMEBUFFER, 0);
            if status != GL_FRAMEBUFFER_COMPLETE {
                return Err(RenderError::from(
                    "create_render_target: the framebuffer is incomplete",
                ));
            }
        }
        let target = OpenGlRenderTarget {
            inner: Arc::new(Mutex::new(RenderTargetInner {
                width: desc.width,
                height: desc.height,
                color_attachment_count: desc.color_attachment_count,
                depth_attachment: desc.depth_attachment,
                released: false,
            })),
            views,
            depth_view,
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
                // The default framebuffer (the pbuffer surface).
                self.bound_target = None;
                unsafe {
                    (self.loader.gl_bind_framebuffer.unwrap())(GL_FRAMEBUFFER, 0);
                }
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
                            unsafe {
                                (self.loader.gl_bind_framebuffer.unwrap())(
                                    GL_FRAMEBUFFER,
                                    self.targets[index].framebuffer,
                                );
                            }
                        }
                    }
                    None => {
                        self.violations += 1;
                    }
                }
            }
        }
        self.pass_open = true;
    }

    fn end_render_pass(&mut self) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.bound_target = None;
        self.pass_open = false;
        unsafe {
            (self.loader.gl_bind_framebuffer.unwrap())(GL_FRAMEBUFFER, 0);
        }
    }

    fn release_render_target(&mut self, target: &RenderTarget) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        match self
            .targets
            .iter_mut()
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
                } else if let Some(view) = t.views.first() {
                    Some(TextureView {
                        inner: view.inner.clone(),
                    })
                } else {
                    t.depth_view
                        .as_ref()
                        .map(|d| TextureView {
                            inner: d.inner.clone(),
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
                t.views
                    .get(attachment as usize)
                    .map(|v| TextureView {
                        inner: v.inner.clone(),
                    })
            }
            None => None,
        }
    }

    fn bind_texture(&mut self, _view: Option<&TextureView>, _unit: u32) {
        if self.shutdown {
            self.violations += 1;
        }
    }

    fn create_texture_2d(&mut self, desc: &Texture2DDesc) -> Result<TextureView, RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "create_texture_2d: the backend is shut down",
            ));
        }
        if desc.width == 0 || desc.height == 0 {
            return Err(RenderError::from(
                "create_texture_2d: width and height must be positive",
            ));
        }
        if desc.format == PixelFormat::Depth24Stencil8 {
            return Err(RenderError::from(
                "create_texture_2d: the depth/stencil format is not a colour texture",
            ));
        }
        let (internal, storage) = self.to_gl_format(desc.format);
        let mut texture: u32 = 0;
        unsafe {
            (self.loader.gl_gen_textures.unwrap())(1, &mut texture);
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, texture);
            (self.loader.gl_tex_image_2d.unwrap())(
                GL_TEXTURE_2D,
                0,
                internal,
                desc.width as i32,
                desc.height as i32,
                0,
                storage,
                GL_UNSIGNED_BYTE,
                std::ptr::null(),
            );
            (self.loader.gl_tex_parameter_i.unwrap())(
                GL_TEXTURE_2D,
                GL_TEXTURE_MIN_FILTER,
                GL_NEAREST as i32,
            );
            (self.loader.gl_tex_parameter_i.unwrap())(
                GL_TEXTURE_2D,
                GL_TEXTURE_MAG_FILTER,
                GL_NEAREST as i32,
            );
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, 0);
        }
        let texture_obj = OpenGlTexture2D {
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
            inner: texture_obj.inner.clone(),
        };
        self.textures_2d.push(texture_obj);
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
        let index = match self
            .textures_2d
            .iter()
            .position(|x| Arc::ptr_eq(&x.inner, &view.inner))
        {
            Some(i) => i,
            None => {
                self.violations += 1;
                return Err(RenderError::from(
                    "fill_texture_2d: the texture is not owned by this backend",
                ));
            }
        };
        if self.textures_2d[index].released {
            self.violations += 1;
            return Err(RenderError::from(
                "fill_texture_2d: the texture is released",
            ));
        }
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
        let texture = self.textures_2d[index].texture;
        let inner = view.inner.lock().unwrap();
        let (internal, storage) = self.to_gl_format(inner.format);
        let (width, height) = (inner.width, inner.height);
        drop(inner);
        unsafe {
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, texture);
            (self.loader.gl_pixel_store_i.unwrap())(GL_UNPACK_ALIGNMENT, 1);
            (self.loader.gl_tex_image_2d.unwrap())(
                GL_TEXTURE_2D,
                0,
                internal,
                width as i32,
                height as i32,
                0,
                storage,
                GL_UNSIGNED_BYTE,
                data.as_ptr() as *const c_void,
            );
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, 0);
        }
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
            Some(i) => i,
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
        // The readback happens before the present (the same pre-present
        // rule the C++ OpenGL backend follows). The pbuffer surface is
        // not presented: its pixels stay readable.
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
        // The readback path copies the centre pixel from the default
        // framebuffer through glReadPixels (the host-visible copy the
        // contract's readPixelInto follows).
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
