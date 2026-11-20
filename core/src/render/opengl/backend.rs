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
    bytes_per_pixel, CameraParams, ClearBuffer, ClearColor,
    DirectionalLight, DynamicLight, GpuMesh, GpuMeshInner, PixelFormat,
    ReadbackSample, RenderBackend, RenderError, RenderTarget,
    RenderTargetDesc, RenderTargetInner, ShaderProgram, ShaderProgramInner,
    Texture2DDesc, Texture3DDesc, TextureView, TextureViewInner, VertexLayout,
    DEPTH_ATTACHMENT,
};
use super::gl_loader::{GlLoader, *};
use super::shader_library::{TRIANGLE_FRAGMENT_SHADER, TRIANGLE_VERTEX_SHADER};
use super::shader_library::{MESH_FRAGMENT_SHADER, MESH_VERTEX_SHADER};
use super::shader_library::{LIT_FRAGMENT_SHADER, LIT_VERTEX_SHADER};
use super::shader_library::{GBUFFER_FRAGMENT_SHADER, GBUFFER_VERTEX_SHADER};
use super::shader_library::{
    LIGHTING_FRAGMENT_SHADER, LIGHTING_VERTEX_SHADER,
};
use super::shader_library::{
    POST_COMPOSITE_FRAGMENT_SHADER, POST_GBUFFER_FRAGMENT_SHADER,
};
use crate::math::{Mat4, Vec3};
use crate::lightvolume::LightVolume;
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

/// The OpenGL backend's own mesh state: the VAO (the attribute pointers),
/// the vertex and the index buffers, and the counts the draw path needs.
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct OpenGlMeshState {
    vao: u32,
    vertex_buffer: u32,
    index_buffer: u32,
    vertex_count: u32,
    index_count: u32,
}

/// The lights' uniform block (M5): the same layout every backend maps —
/// the sun's direction (xyz + pad), the sun's colour (xyz) and intensity
/// (w), the four point lights' position (xyz) and radius (w), the four
/// point lights' colour (xyz) and intensity (w), the params (x = the
/// live point-light count). The GL path uploads the fields as the lit
/// fragment shader's named uniform arrays (the same layout the other
/// backends' uniform blocks carry).
#[repr(C)]
#[derive(Clone, Copy)]
struct LightsUniform {
    sun_direction: [f32; 4],
    sun_color_intensity: [f32; 4],
    point_position_radius: [[f32; 4]; 4],
    point_color_intensity: [[f32; 4]; 4],
    params: [f32; 4],
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
    // The generic mesh path's shared program (the PNU + camera + sampler
    // shader pair the shader library carries; one program serves every
    // mesh, the way the triangle's program serves the diagnostic draw).
    mesh_program: u32,
    mesh_uniform_view_projection: i32,
    mesh_uniform_model: i32,
    mesh_uniform_sampler: i32,
    mesh_ready: bool,
    // The lit mesh path's program (M5): the same PNU vertex shape, the
    // LIT shader pair, the lights' uniforms driven by name.
    lit_program: u32,
    lit_uniform_view_projection: i32,
    lit_uniform_model: i32,
    lit_uniform_sampler: i32,
    lit_uniform_sun_direction: i32,
    lit_uniform_sun_color_intensity: i32,
    lit_uniform_point_position_radius: i32,
    lit_uniform_point_color_intensity: i32,
    lit_uniform_params: i32,
    lit_ready: bool,
    // The deferred lighting path's programs (M6): the G-buffer's
    // geometry-pass program and the fullscreen lighting-pass program.
    gbuffer_program: u32,
    gbuffer_uniform_view_projection: i32,
    gbuffer_uniform_model: i32,
    gbuffer_uniform_sampler: i32,
    gbuffer_ready: bool,
    lighting_program: u32,
    lighting_uniform_sun_direction: i32,
    lighting_uniform_sun_color_intensity: i32,
    lighting_uniform_point_position_radius: i32,
    lighting_uniform_point_color_intensity: i32,
    lighting_uniform_params: i32,
    lighting_uniform_albedo: i32,
    lighting_uniform_normal: i32,
    lighting_uniform_world: i32,
    lighting_ready: bool,
    // The post-processing chain's programs (M7): the HDR geometry-pass
    // program and the fullscreen composite-pass program.
    post_gbuffer_program: u32,
    post_gbuffer_uniform_view_projection: i32,
    post_gbuffer_uniform_model: i32,
    post_gbuffer_uniform_sampler: i32,
    post_gbuffer_uniform_sun_direction: i32,
    post_gbuffer_uniform_sun_color_intensity: i32,
    post_gbuffer_uniform_point_position_radius: i32,
    post_gbuffer_uniform_point_color_intensity: i32,
    post_gbuffer_uniform_params: i32,
    post_gbuffer_uniform_volume_origin: i32,
    post_gbuffer_uniform_volume_dims: i32,
    post_gbuffer_uniform_volume_params: i32,
    post_gbuffer_uniform_volume_sampler: i32,
    post_gbuffer_ready: bool,
    post_composite_program: u32,
    post_composite_uniform_radiance: i32,
    post_composite_uniform_world: i32,
    post_composite_uniform_params0: i32,
    post_composite_uniform_params1: i32,
    post_composite_uniform_params2: i32,
    post_composite_ready: bool,
    post_composite_vao: u32,
    /// The fullscreen triangle's own VAO (the core profile forbids the
    /// VAO 0 draw; an attribute-less draw still needs a bound VAO).
    lighting_vao: u32,
    // The submission's lights block (the draw's uniform writes read it).
    lit_block: LightsUniform,
    /// The per-mesh state: the VAO (the attribute pointers), the vertex
    /// and the index buffers, the vertex/index counts (the create order
    /// matches the handle's id).
    meshes: Vec<Option<OpenGlMeshState>>,
    /// The sampler unit 0's bound texture (the texture's index in the
    /// textures_2d vector; None keeps the default texture).
    bound_texture: Option<usize>,
    /// The default texture's index (the 1x1 white texel the setup creates).
    default_texture: Option<usize>,
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
    textures_3d: Vec<OpenGlTexture2D>,
    light_volume: Option<OpenGlTexture2D>,
    volume_uniform: [f32; 12],
    next_mesh_id: usize,
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
            mesh_program: 0,
            mesh_uniform_view_projection: -1,
            mesh_uniform_model: -1,
            mesh_uniform_sampler: -1,
            mesh_ready: false,
            lit_program: 0,
            lit_uniform_view_projection: -1,
            lit_uniform_model: -1,
            lit_uniform_sampler: -1,
            lit_uniform_sun_direction: -1,
            lit_uniform_sun_color_intensity: -1,
            lit_uniform_point_position_radius: -1,
            lit_uniform_point_color_intensity: -1,
            lit_uniform_params: -1,
            lit_ready: false,
            gbuffer_program: 0,
            gbuffer_uniform_view_projection: -1,
            gbuffer_uniform_model: -1,
            gbuffer_uniform_sampler: -1,
            gbuffer_ready: false,
            lighting_program: 0,
            lighting_uniform_sun_direction: -1,
            lighting_uniform_sun_color_intensity: -1,
            lighting_uniform_point_position_radius: -1,
            lighting_uniform_point_color_intensity: -1,
            lighting_uniform_params: -1,
            lighting_uniform_albedo: -1,
            lighting_uniform_normal: -1,
            lighting_uniform_world: -1,
            lighting_ready: false,
            post_gbuffer_program: 0,
            post_gbuffer_uniform_view_projection: -1,
            post_gbuffer_uniform_model: -1,
            post_gbuffer_uniform_sampler: -1,
            post_gbuffer_uniform_sun_direction: -1,
            post_gbuffer_uniform_sun_color_intensity: -1,
            post_gbuffer_uniform_point_position_radius: -1,
            post_gbuffer_uniform_point_color_intensity: -1,
            post_gbuffer_uniform_params: -1,
            post_gbuffer_uniform_volume_origin: -1,
            post_gbuffer_uniform_volume_dims: -1,
            post_gbuffer_uniform_volume_params: -1,
            post_gbuffer_uniform_volume_sampler: -1,
            post_gbuffer_ready: false,
            post_composite_program: 0,
            post_composite_uniform_radiance: -1,
            post_composite_uniform_world: -1,
            post_composite_uniform_params0: -1,
            post_composite_uniform_params1: -1,
            post_composite_uniform_params2: -1,
            post_composite_ready: false,
            post_composite_vao: 0,
            lighting_vao: 0,
            lit_block: LightsUniform::none(),
            meshes: Vec::new(),
            bound_texture: None,
            default_texture: None,
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
            volume_uniform: [0.0; 12],
            next_mesh_id: 0,
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
        // The default texture: one 1x1 opaque white texel the mesh path's
        // sampler reads when no texture is bound (the same default the
        // Vulkan path carries).
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
            .position(|t| Arc::ptr_eq(&t.inner, &view.inner))
            .expect("the default texture's slot");
        self.default_texture = Some(index);
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
    /// The mesh path's shared program: the two mesh shaders from the
    /// library, compiled at runtime by the driver (the same lazy ensure
    /// the triangle path follows).
    pub(crate) fn ensure_mesh_resources(&mut self) -> bool {
        if self.mesh_ready {
            return true;
        }
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
        let vs_source = [MESH_VERTEX_SHADER.as_ptr() as *const c_char];
        let vs_len = [MESH_VERTEX_SHADER.len() as i32];
        let fs_source = [MESH_FRAGMENT_SHADER.as_ptr() as *const c_char];
        let fs_len = [MESH_FRAGMENT_SHADER.len() as i32];
        unsafe {
            (self.loader.gl_shader_source.unwrap())(
                vs,
                1,
                vs_source.as_ptr(),
                vs_len.as_ptr(),
            );
            (self.loader.gl_compile_shader.unwrap())(vs);
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
            unsafe {
                (self.loader.gl_delete_shader.unwrap())(vs);
                (self.loader.gl_delete_shader.unwrap())(fs);
                (self.loader.gl_delete_program.unwrap())(program);
            }
            return false;
        }
        unsafe {
            (self.loader.gl_delete_shader.unwrap())(vs);
            (self.loader.gl_delete_shader.unwrap())(fs);
        }
        self.mesh_program = program;
        self.mesh_uniform_view_projection = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"view_projection\0".as_ptr() as *const c_char,
            )
        };
        self.mesh_uniform_model = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"model\0".as_ptr() as *const c_char,
            )
        };
        self.mesh_uniform_sampler = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"surface_texture\0".as_ptr() as *const c_char,
            )
        };
        self.mesh_ready = true;
        true
    }

    /// The lit mesh path's program (M5): the LIT shader pair from the
    /// library, compiled at runtime by the driver (the same lazy ensure
    /// the unlit mesh path follows; the compilation happens outside an
    /// open frame — the upload's ensure, not the draw's).
    pub(crate) fn ensure_lit_program(&mut self) -> bool {
        if self.lit_ready {
            return true;
        }
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
        let vs_source = [LIT_VERTEX_SHADER.as_ptr() as *const c_char];
        let vs_len = [LIT_VERTEX_SHADER.len() as i32];
        let fs_source = [LIT_FRAGMENT_SHADER.as_ptr() as *const c_char];
        let fs_len = [LIT_FRAGMENT_SHADER.len() as i32];
        unsafe {
            (self.loader.gl_shader_source.unwrap())(
                vs,
                1,
                vs_source.as_ptr(),
                vs_len.as_ptr(),
            );
            (self.loader.gl_compile_shader.unwrap())(vs);
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
            unsafe {
                (self.loader.gl_delete_shader.unwrap())(vs);
                (self.loader.gl_delete_shader.unwrap())(fs);
                (self.loader.gl_delete_program.unwrap())(program);
            }
            return false;
        }
        unsafe {
            (self.loader.gl_delete_shader.unwrap())(vs);
            (self.loader.gl_delete_shader.unwrap())(fs);
        }
        self.lit_program = program;
        self.lit_uniform_view_projection = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"view_projection\0".as_ptr() as *const c_char,
            )
        };
        self.lit_uniform_model = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"model\0".as_ptr() as *const c_char,
            )
        };
        self.lit_uniform_sampler = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"surface_texture\0".as_ptr() as *const c_char,
            )
        };
        self.lit_uniform_sun_direction = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"sun_direction\0".as_ptr() as *const c_char,
            )
        };
        self.lit_uniform_sun_color_intensity = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"sun_color_intensity\0".as_ptr() as *const c_char,
            )
        };
        self.lit_uniform_point_position_radius = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"point_position_radius\0".as_ptr() as *const c_char,
            )
        };
        self.lit_uniform_point_color_intensity = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"point_color_intensity\0".as_ptr() as *const c_char,
            )
        };
        self.lit_uniform_params = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"params\0".as_ptr() as *const c_char,
            )
        };
        self.lit_ready = true;
        true
    }

    /// The deferred geometry pass's program (M6): the G-buffer shader
    /// pair (the same PNU vertex shape, the three-attachment fragment
    /// shader), compiled at runtime (the same lazy ensure the lit path
    /// follows).
    pub(crate) fn ensure_gbuffer_program(&mut self) -> bool {
        if self.gbuffer_ready {
            return true;
        }
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
        let vs_source = [GBUFFER_VERTEX_SHADER.as_ptr() as *const c_char];
        let vs_len = [GBUFFER_VERTEX_SHADER.len() as i32];
        let fs_source = [GBUFFER_FRAGMENT_SHADER.as_ptr() as *const c_char];
        let fs_len = [GBUFFER_FRAGMENT_SHADER.len() as i32];
        unsafe {
            (self.loader.gl_shader_source.unwrap())(
                vs,
                1,
                vs_source.as_ptr(),
                vs_len.as_ptr(),
            );
            (self.loader.gl_compile_shader.unwrap())(vs);
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
            unsafe {
                (self.loader.gl_delete_shader.unwrap())(vs);
                (self.loader.gl_delete_shader.unwrap())(fs);
                (self.loader.gl_delete_program.unwrap())(program);
            }
            return false;
        }
        unsafe {
            (self.loader.gl_delete_shader.unwrap())(vs);
            (self.loader.gl_delete_shader.unwrap())(fs);
        }
        self.gbuffer_program = program;
        self.gbuffer_uniform_view_projection = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"view_projection\0".as_ptr() as *const c_char,
            )
        };
        self.gbuffer_uniform_model = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"model\0".as_ptr() as *const c_char,
            )
        };
        self.gbuffer_uniform_sampler = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"surface_texture\0".as_ptr() as *const c_char,
            )
        };
        self.gbuffer_ready = true;
        true
    }

    /// The deferred lighting pass's program (M6): the fullscreen
    /// triangle's vertex shader and the G-buffer-sampling fragment
    /// shader (the same Blinn-Phong the forward path carries), compiled
    /// at runtime (the same lazy ensure).
    pub(crate) fn ensure_lighting_program(&mut self) -> bool {
        if self.lighting_ready {
            return true;
        }
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
        let vs_source = [LIGHTING_VERTEX_SHADER.as_ptr() as *const c_char];
        let vs_len = [LIGHTING_VERTEX_SHADER.len() as i32];
        let fs_source = [LIGHTING_FRAGMENT_SHADER.as_ptr() as *const c_char];
        let fs_len = [LIGHTING_FRAGMENT_SHADER.len() as i32];
        unsafe {
            (self.loader.gl_shader_source.unwrap())(
                vs,
                1,
                vs_source.as_ptr(),
                vs_len.as_ptr(),
            );
            (self.loader.gl_compile_shader.unwrap())(vs);
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
            unsafe {
                (self.loader.gl_delete_shader.unwrap())(vs);
                (self.loader.gl_delete_shader.unwrap())(fs);
                (self.loader.gl_delete_program.unwrap())(program);
            }
            return false;
        }
        unsafe {
            (self.loader.gl_delete_shader.unwrap())(vs);
            (self.loader.gl_delete_shader.unwrap())(fs);
        }
        self.lighting_program = program;
        self.lighting_uniform_sun_direction = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"sun_direction\0".as_ptr() as *const c_char,
            )
        };
        self.lighting_uniform_sun_color_intensity = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"sun_color_intensity\0".as_ptr() as *const c_char,
            )
        };
        self.lighting_uniform_point_position_radius = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"point_position_radius\0".as_ptr() as *const c_char,
            )
        };
        self.lighting_uniform_point_color_intensity = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"point_color_intensity\0".as_ptr() as *const c_char,
            )
        };
        self.lighting_uniform_params = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"params\0".as_ptr() as *const c_char,
            )
        };
        self.lighting_uniform_albedo = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"gbuffer_albedo\0".as_ptr() as *const c_char,
            )
        };
        self.lighting_uniform_normal = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"gbuffer_normal\0".as_ptr() as *const c_char,
            )
        };
        self.lighting_uniform_world = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"gbuffer_world\0".as_ptr() as *const c_char,
            )
        };
        // The fullscreen triangle's own VAO: the core profile forbids
        // the VAO 0 draw (the attribute-less draw still needs a bound
        // VAO — the same rule the mesh path's VAO follows).
        let mut vao: u32 = 0;
        unsafe {
            (self.loader.gl_gen_vertex_arrays.unwrap())(1, &mut vao);
        }
        self.lighting_vao = vao;
        self.lighting_ready = true;
        true
    }
    /// The post chain's geometry pass's program (M7): the G-buffer
    /// shader pair (the same PNU vertex shape, the two-attachment HDR
    /// fragment shader), compiled at runtime (the same lazy ensure the
    /// lit path follows).
    pub(crate) fn ensure_post_gbuffer_program(&mut self) -> bool {
        if self.post_gbuffer_ready {
            return true;
        }
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
        let vs_source = [GBUFFER_VERTEX_SHADER.as_ptr() as *const c_char];
        let vs_len = [GBUFFER_VERTEX_SHADER.len() as i32];
        let fs_source = [POST_GBUFFER_FRAGMENT_SHADER.as_ptr() as *const c_char];
        let fs_len = [POST_GBUFFER_FRAGMENT_SHADER.len() as i32];
        unsafe {
            (self.loader.gl_shader_source.unwrap())(
                vs,
                1,
                vs_source.as_ptr(),
                vs_len.as_ptr(),
            );
            (self.loader.gl_compile_shader.unwrap())(vs);
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
            unsafe {
                (self.loader.gl_delete_shader.unwrap())(vs);
                (self.loader.gl_delete_shader.unwrap())(fs);
                (self.loader.gl_delete_program.unwrap())(program);
            }
            return false;
        }
        unsafe {
            (self.loader.gl_delete_shader.unwrap())(vs);
            (self.loader.gl_delete_shader.unwrap())(fs);
        }
        self.post_gbuffer_program = program;
        self.post_gbuffer_uniform_view_projection = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"view_projection\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_uniform_model = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"model\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_uniform_sampler = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"surface_texture\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_uniform_sun_direction = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"sun_direction\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_uniform_sun_color_intensity = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"sun_color_intensity\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_uniform_point_position_radius = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"point_position_radius\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_uniform_point_color_intensity = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"point_color_intensity\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_uniform_params = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"params\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_uniform_volume_origin = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"volume_origin\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_uniform_volume_dims = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"volume_dims\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_uniform_volume_params = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"volume_params\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_uniform_volume_sampler = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"volume_texture\0".as_ptr() as *const c_char,
            )
        };
        self.post_gbuffer_ready = true;
        true
    }
    /// The post chain's composite pass's program (M7): the fullscreen
    /// shader pair (the same gl_VertexID triangle the lighting path
    /// carries, the composite fragment shader).
    pub(crate) fn ensure_post_composite_program(&mut self) -> bool {
        if self.post_composite_ready {
            return true;
        }
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
        let vs_source = [LIGHTING_VERTEX_SHADER.as_ptr() as *const c_char];
        let vs_len = [LIGHTING_VERTEX_SHADER.len() as i32];
        let fs_source = [POST_COMPOSITE_FRAGMENT_SHADER.as_ptr() as *const c_char];
        let fs_len = [POST_COMPOSITE_FRAGMENT_SHADER.len() as i32];
        unsafe {
            (self.loader.gl_shader_source.unwrap())(
                vs,
                1,
                vs_source.as_ptr(),
                vs_len.as_ptr(),
            );
            (self.loader.gl_compile_shader.unwrap())(vs);
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
            unsafe {
                (self.loader.gl_delete_shader.unwrap())(vs);
                (self.loader.gl_delete_shader.unwrap())(fs);
                (self.loader.gl_delete_program.unwrap())(program);
            }
            return false;
        }
        unsafe {
            (self.loader.gl_delete_shader.unwrap())(vs);
            (self.loader.gl_delete_shader.unwrap())(fs);
        }
        self.post_composite_program = program;
        self.post_composite_uniform_radiance = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"radiance_texture\0".as_ptr() as *const c_char,
            )
        };
        self.post_composite_uniform_world = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"world_texture\0".as_ptr() as *const c_char,
            )
        };
        self.post_composite_uniform_params0 = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"params0\0".as_ptr() as *const c_char,
            )
        };
        self.post_composite_uniform_params1 = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"params1\0".as_ptr() as *const c_char,
            )
        };
        self.post_composite_uniform_params2 = unsafe {
            (self.loader.gl_get_uniform_location.unwrap())(
                program,
                b"params2\0".as_ptr() as *const c_char,
            )
        };
        // The fullscreen triangle's own VAO (the core profile forbids
        // the VAO 0 draw — the same rule the lighting path follows).
        let mut vao: u32 = 0;
        unsafe {
            (self.loader.gl_gen_vertex_arrays.unwrap())(1, &mut vao);
        }
        self.post_composite_vao = vao;
        self.post_composite_ready = true;
        true
    }

    /// The generic draw path: the mesh's VAO, the camera uniforms (the
    /// set_camera state's view-projection and the submission's model) and
    /// the indexed draw. A draw with no open frame counts one violation
    /// (the same rule the Vulkan path follows).
    fn draw_mesh_by_id(&mut self, mesh_id: usize, model: Mat4) {
        self.draw_mesh_inner(mesh_id, model, false);
    }

    /// The lit draw (M5): the same path, the lit program and the
    /// lights' uniforms uploaded (the submit path carries the lights).
    fn draw_mesh_lit_by_id(&mut self, mesh_id: usize, model: Mat4) {
        self.draw_mesh_inner(mesh_id, model, true);
    }

    /// The mesh draw's shared body: the unlit path binds the mesh
    /// program; the lit path binds the lit program and uploads the
    /// lights' uniforms (the same Blinn-Phong state the Vulkan path's
    /// block carries).
    fn draw_mesh_inner(&mut self, mesh_id: usize, model: Mat4, lit: bool) {
        if !self.frame_open {
            self.violations += 1;
            return;
        }
        let (vao, index_count) = match self.meshes.get(mesh_id) {
            Some(Some(state)) => (state.vao, state.index_count),
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
        // The camera uniforms: the set_camera state's view-projection
        // (the same matrices the camera path builds) and the model matrix
        // the submission carries.
        let camera = self.camera;
        let aspect = self.config.width as f32 / self.config.height as f32;
        let view = Mat4::look_at(camera.eye, camera.target, camera.up);
        let projection =
            Mat4::perspective(camera.fov_y_rad, aspect, camera.near, camera.far);
        let view_projection = projection.multiply(view);
        let texture = self.bound_texture
            .and_then(|i| self.textures_2d.get(i))
            .filter(|t| !t.released)
            .map(|t| t.texture)
            .unwrap_or(
                self.default_texture
                    .and_then(|i| self.textures_2d.get(i))
                    .filter(|t| !t.released)
                    .map(|t| t.texture)
                    .unwrap_or(0),
            );
        unsafe {
            (self.loader.gl_use_program.unwrap())(if lit {
                self.lit_program
            } else {
                self.mesh_program
            });
            (self.loader.gl_uniform_matrix_4fv.unwrap())(
                if lit {
                    self.lit_uniform_view_projection
                } else {
                    self.mesh_uniform_view_projection
                },
                1,
                GL_FALSE,
                view_projection.m.as_ptr() as *const f32,
            );
            (self.loader.gl_uniform_matrix_4fv.unwrap())(
                if lit {
                    self.lit_uniform_model
                } else {
                    self.mesh_uniform_model
                },
                1,
                GL_FALSE,
                model.m.as_ptr() as *const f32,
            );
            (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0);
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, texture);
            (self.loader.gl_uniform_1i.unwrap())(
                if lit {
                    self.lit_uniform_sampler
                } else {
                    self.mesh_uniform_sampler
                },
                0,
            );
            if lit {
                // The lights' uniforms: the block's fields, the arrays
                // via the 4fv path (the same values the Vulkan path's
                // block carries).
                let block = self.lit_block;
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.lit_uniform_sun_direction,
                    1,
                    block.sun_direction.as_ptr(),
                );
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.lit_uniform_sun_color_intensity,
                    1,
                    block.sun_color_intensity.as_ptr(),
                );
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.lit_uniform_point_position_radius,
                    4,
                    block.point_position_radius.as_ptr() as *const f32,
                );
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.lit_uniform_point_color_intensity,
                    4,
                    block.point_color_intensity.as_ptr() as *const f32,
                );
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.lit_uniform_params,
                    1,
                    block.params.as_ptr(),
                );
            }
            (self.loader.gl_bind_vertex_array.unwrap())(vao);
            (self.loader.gl_draw_elements.unwrap())(
                GL_TRIANGLES,
                index_count as i32,
                GL_UNSIGNED_INT,
                std::ptr::null(),
            );
            (self.loader.gl_bind_vertex_array.unwrap())(0);
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
        // The mesh's GPU buffers: one VAO (the attribute pointers), one
        // vertex buffer (the PNU bytes) and one index buffer (the u32s).
        // The generic path's first layout is the PNU the mesh module's
        // pnu_layout() builds; the pointers come from the layout's
        // attributes (position at 0, normal at 12, texcoord0 at 24).
        let mesh_id = mesh.inner.lock().unwrap().id;
        let mut vao: u32 = 0;
        let mut vbo: u32 = 0;
        let ibo: u32;
        unsafe {
            (self.loader.gl_gen_vertex_arrays.unwrap())(1, &mut vao);
            (self.loader.gl_gen_buffers.unwrap())(2, &mut vbo);
            ibo = vbo.wrapping_add(1);
            (self.loader.gl_bind_vertex_array.unwrap())(vao);
            (self.loader.gl_bind_buffer.unwrap())(GL_ARRAY_BUFFER, vbo);
            (self.loader.gl_buffer_data.unwrap())(
                GL_ARRAY_BUFFER,
                vertices.len() as isize,
                vertices.as_ptr() as *const c_void,
                GL_STATIC_DRAW,
            );
            // The attribute pointers: one per layout attribute, the
            // semantic mapping onto the shader's locations (position 0,
            // normal 1, texcoord0 2).
            for attribute in &layout.attributes {
                let location = match attribute.semantic {
                    crate::render::VertexSemantic::Position => 0,
                    crate::render::VertexSemantic::Normal => 1,
                    crate::render::VertexSemantic::TexCoord0 => 2,
                    crate::render::VertexSemantic::Color => 3,
                };
                let (size, gl_type) = match attribute.component_type {
                    crate::render::ComponentType::Float32 => (
                        attribute.components as i32,
                        GL_FLOAT,
                    ),
                    crate::render::ComponentType::Uint8Normalized => (
                        attribute.components as i32,
                        GL_UNSIGNED_BYTE,
                    ),
                };
                (self.loader.gl_vertex_attrib_pointer.unwrap())(
                    location,
                    size,
                    gl_type,
                    GL_FALSE,
                    layout.stride as i32,
                    attribute.offset as *const c_void,
                );
                (self.loader.gl_enable_vertex_attrib_array.unwrap())(location);
            }
            (self.loader.gl_bind_buffer.unwrap())(
                GL_ELEMENT_ARRAY_BUFFER,
                ibo,
            );
            (self.loader.gl_buffer_data.unwrap())(
                GL_ELEMENT_ARRAY_BUFFER,
                (indices.len() * std::mem::size_of::<u32>()) as isize,
                indices.as_ptr() as *const c_void,
                GL_STATIC_DRAW,
            );
            (self.loader.gl_bind_vertex_array.unwrap())(0);
            (self.loader.gl_bind_buffer.unwrap())(GL_ARRAY_BUFFER, 0);
            (self.loader.gl_bind_buffer.unwrap())(GL_ELEMENT_ARRAY_BUFFER, 0);
        }
        // The mesh path's program: the same lazy ensure the triangle path
        // follows (one program serves every mesh).
        if !self.ensure_mesh_resources() {
            return Err(RenderError::from(
                "upload_mesh: the mesh's shader program failed to compile",
            ));
        }
        let state = OpenGlMeshState {
            vao,
            vertex_buffer: vbo,
            index_buffer: ibo,
            vertex_count: (vertices.len() as u32) / layout.stride,
            index_count: indices.len() as u32,
        };
        if self.meshes.len() <= mesh_id {
            self.meshes.resize(mesh_id + 1, None);
        }
        self.meshes[mesh_id] = Some(state);
        Ok(())
    }
    fn draw(&mut self, mesh: &GpuMesh, _program: &ShaderProgram) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        let mesh_id = mesh.inner.lock().unwrap().id;
        self.draw_mesh_by_id(mesh_id, Mat4::IDENTITY);
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
        let mesh_id = submission.mesh.inner.lock().unwrap().id;
        let has_lights = submission.sun.is_some() || !submission.lights.is_empty();
        if has_lights {
            // The lit path (M5): the lights' block from the submission's
            // sun and point lights (the sun absent means intensity zero),
            // the lit program draws.
            let default_sun = DirectionalLight {
                direction: Vec3::new(0.0, -1.0, 0.0),
                color: Vec3::new(0.0, 0.0, 0.0),
                intensity: 0.0,
            };
            let sun = submission.sun.as_ref().unwrap_or(&default_sun);
            self.lit_block = LightsUniform::from_lights(sun, &submission.lights);
            if self.ensure_lit_program() {
                self.draw_mesh_lit_by_id(mesh_id, submission.model);
                return;
            }
            // The lit path's failure falls back to the unlit draw (the
            // render keeps going).
        }
        self.draw_mesh_by_id(mesh_id, submission.model);
    }

    /// The deferred lighting pass (M6): the two internal steps the
    /// contract's submit_lighting_pass carries. Step one (the geometry
    /// pass) binds the G-buffer's framebuffer, uploads the camera
    /// uniforms and draws the mesh into the three colour attachments.
    /// Step two (the lighting pass) binds the default framebuffer,
    /// uploads the lights' uniforms and the three G-buffer textures, and
    /// draws the fullscreen triangle (the three vertices the vertex ID
    /// generates, no VAO). The G-buffer's textures read through the
    /// sampler units 0-2.
    fn submit_lighting_pass(
        &mut self,
        submission: &crate::render::DrawSubmission,
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
        if !self.ensure_gbuffer_program() || !self.ensure_lighting_program() {
            return;
        }
        if !self.frame_open {
            self.violations += 1;
            return;
        }
        let default_sun = DirectionalLight {
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::new(0.0, 0.0, 0.0),
            intensity: 0.0,
        };
        let sun = submission.sun.as_ref().unwrap_or(&default_sun);
        let block = LightsUniform::from_lights(sun, &submission.lights);
        let camera = self.camera;
        let aspect = self.config.width as f32 / self.config.height as f32;
        let view = Mat4::look_at(camera.eye, camera.target, camera.up);
        let projection =
            Mat4::perspective(camera.fov_y_rad, aspect, camera.near, camera.far);
        let view_projection = projection.multiply(view);
        let model = submission.model;
        let texture = self
            .bound_texture
            .and_then(|i| self.textures_2d.get(i))
            .filter(|t| !t.released)
            .map(|t| t.texture)
            .unwrap_or(
                self.default_texture
                    .and_then(|i| self.textures_2d.get(i))
                    .filter(|t| !t.released)
                    .map(|t| t.texture)
                    .unwrap_or(0),
            );
        let gbuffer_framebuffer = self.targets[target_index].framebuffer;
        let gbuffer_textures: Vec<u32> = self.targets[target_index]
            .views
            .iter()
            .take(3)
            .map(|v| v.texture)
            .collect();
        if gbuffer_textures.len() != 3 {
            self.violations += 1;
            return;
        }
        let depth_test_enabled = self.depth_test_enabled;
        let clear_color = self.clear_color;
        let clear_depth = self.clear_depth;
        let clear_stencil = self.clear_stencil;
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
        // ---- Step one: the geometry pass ----
        unsafe {
            (self.loader.gl_bind_framebuffer.unwrap())(
                GL_FRAMEBUFFER,
                gbuffer_framebuffer,
            );
            let colour_attachments = [
                GL_COLOR_ATTACHMENT0,
                GL_COLOR_ATTACHMENT1,
                GL_COLOR_ATTACHMENT2,
            ];
            (self.loader.gl_draw_buffers.unwrap())(
                3,
                colour_attachments.as_ptr(),
            );
            (self.loader.gl_clear_color.unwrap())(
                clear_color.r as f32 / 255.0,
                clear_color.g as f32 / 255.0,
                clear_color.b as f32 / 255.0,
                clear_color.a as f32 / 255.0,
            );
            (self.loader.gl_clear_depth.unwrap())(clear_depth as f64);
            (self.loader.gl_clear_stencil.unwrap())(clear_stencil);
            (self.loader.gl_clear.unwrap())(mask);
            if depth_test_enabled {
                (self.loader.gl_enable.unwrap())(GL_DEPTH_TEST);
                (self.loader.gl_depth_func.unwrap())(GL_LESS);
                (self.loader.gl_depth_mask.unwrap())(GL_TRUE);
            } else {
                (self.loader.gl_disable.unwrap())(GL_DEPTH_TEST);
                (self.loader.gl_depth_mask.unwrap())(GL_FALSE);
            }
        }
        {
            let (vao, index_count) = match self.meshes.get(mesh_id) {
                Some(Some(state)) => (state.vao, state.index_count),
                _ => {
                    self.violations += 1;
                    return;
                }
            };
            unsafe {
                (self.loader.gl_use_program.unwrap())(self.gbuffer_program);
                (self.loader.gl_uniform_matrix_4fv.unwrap())(
                    self.gbuffer_uniform_view_projection,
                    1,
                    GL_FALSE,
                    view_projection.m.as_ptr() as *const f32,
                );
                (self.loader.gl_uniform_matrix_4fv.unwrap())(
                    self.gbuffer_uniform_model,
                    1,
                    GL_FALSE,
                    model.m.as_ptr() as *const f32,
                );
                (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0);
                (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, texture);
                (self.loader.gl_uniform_1i.unwrap())(
                    self.gbuffer_uniform_sampler,
                    0,
                );
                (self.loader.gl_bind_vertex_array.unwrap())(vao);
                (self.loader.gl_draw_elements.unwrap())(
                    GL_TRIANGLES,
                    index_count as i32,
                    GL_UNSIGNED_INT,
                    std::ptr::null(),
                );
                (self.loader.gl_bind_vertex_array.unwrap())(0);
            }
        }
        // ---- Step two: the lighting pass ----
        unsafe {
            (self.loader.gl_bind_framebuffer.unwrap())(GL_FRAMEBUFFER, 0);
            (self.loader.gl_draw_buffer.unwrap())(GL_COLOR_ATTACHMENT0);
            (self.loader.gl_clear_color.unwrap())(
                clear_color.r as f32 / 255.0,
                clear_color.g as f32 / 255.0,
                clear_color.b as f32 / 255.0,
                clear_color.a as f32 / 255.0,
            );
            (self.loader.gl_clear_depth.unwrap())(clear_depth as f64);
            (self.loader.gl_clear_stencil.unwrap())(clear_stencil);
            (self.loader.gl_clear.unwrap())(mask);
            // The depth test off: the fullscreen triangle covers the
            // frame and the default framebuffer's depth stays at the
            // clear value (the depth lives in the G-buffer's depth
            // attachment during the geometry pass).
            (self.loader.gl_disable.unwrap())(GL_DEPTH_TEST);
            (self.loader.gl_depth_mask.unwrap())(GL_FALSE);
            (self.loader.gl_use_program.unwrap())(self.lighting_program);
            (self.loader.gl_uniform_4fv.unwrap())(
                self.lighting_uniform_sun_direction,
                1,
                block.sun_direction.as_ptr(),
            );
            (self.loader.gl_uniform_4fv.unwrap())(
                self.lighting_uniform_sun_color_intensity,
                1,
                block.sun_color_intensity.as_ptr(),
            );
            (self.loader.gl_uniform_4fv.unwrap())(
                self.lighting_uniform_point_position_radius,
                4,
                block.point_position_radius.as_ptr() as *const f32,
            );
            (self.loader.gl_uniform_4fv.unwrap())(
                self.lighting_uniform_point_color_intensity,
                4,
                block.point_color_intensity.as_ptr() as *const f32,
            );
            (self.loader.gl_uniform_4fv.unwrap())(
                self.lighting_uniform_params,
                1,
                block.params.as_ptr(),
            );
            (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0);
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, gbuffer_textures[0]);
            (self.loader.gl_uniform_1i.unwrap())(self.lighting_uniform_albedo, 0);
            (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0 + 1);
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, gbuffer_textures[1]);
            (self.loader.gl_uniform_1i.unwrap())(self.lighting_uniform_normal, 1);
            (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0 + 2);
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, gbuffer_textures[2]);
            (self.loader.gl_uniform_1i.unwrap())(self.lighting_uniform_world, 2);
            (self.loader.gl_bind_vertex_array.unwrap())(self.lighting_vao);
            (self.loader.gl_draw_arrays.unwrap())(GL_TRIANGLES, 0, 3);
            (self.loader.gl_bind_vertex_array.unwrap())(0);
            (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0);
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, texture);
            if depth_test_enabled {
                (self.loader.gl_enable.unwrap())(GL_DEPTH_TEST);
                (self.loader.gl_depth_func.unwrap())(GL_LESS);
                (self.loader.gl_depth_mask.unwrap())(GL_TRUE);
            }
        }
    }

    /// The post-processing chain (M7): the HDR geometry pass writes the
    /// shaded radiance and the world position into the target's two
    /// Rgba16F attachments, the composite pass samples both and applies
    /// the enabled effects into the default framebuffer (the same two-
    /// pass flow the Vulkan path carries).
    fn submit_post_chain(
        &mut self,
        submission: &crate::render::DrawSubmission,
        hdr_target: &RenderTarget,
        config: &crate::render::PostChainConfig,
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
        if self.meshes.get(mesh_id).map(|m| m.is_none()).unwrap_or(true) {
            self.violations += 1;
            return;
        }
        if !self.ensure_post_gbuffer_program()
            || !self.ensure_post_composite_program()
        {
            return;
        }
        if !self.frame_open {
            self.violations += 1;
            return;
        }
        let default_sun = DirectionalLight {
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::new(0.0, 0.0, 0.0),
            intensity: 0.0,
        };
        let sun = submission.sun.as_ref().unwrap_or(&default_sun);
        let block = LightsUniform::from_lights(sun, &submission.lights);
        let camera = self.camera;
        let aspect = self.config.width as f32 / self.config.height as f32;
        let view = Mat4::look_at(camera.eye, camera.target, camera.up);
        let projection =
            Mat4::perspective(camera.fov_y_rad, aspect, camera.near, camera.far);
        let view_projection = projection.multiply(view);
        let model = submission.model;
        let texture = self
            .bound_texture
            .and_then(|i| self.textures_2d.get(i))
            .filter(|t| !t.released)
            .map(|t| t.texture)
            .unwrap_or(
                self.default_texture
                    .and_then(|i| self.textures_2d.get(i))
                    .filter(|t| !t.released)
                    .map(|t| t.texture)
                    .unwrap_or(0),
            );
        let post_framebuffer = self.targets[target_index].framebuffer;
        let post_textures: Vec<u32> = self.targets[target_index]
            .views
            .iter()
            .take(2)
            .map(|v| v.texture)
            .collect();
        if post_textures.len() != 2 {
            self.violations += 1;
            return;
        }
        let depth_test_enabled = self.depth_test_enabled;
        let clear_color = self.clear_color;
        let clear_depth = self.clear_depth;
        let clear_stencil = self.clear_stencil;
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
        // ---- Step one: the HDR geometry pass ----
        unsafe {
            (self.loader.gl_bind_framebuffer.unwrap())(
                GL_FRAMEBUFFER,
                post_framebuffer,
            );
            let colour_attachments = [
                GL_COLOR_ATTACHMENT0,
                GL_COLOR_ATTACHMENT1,
            ];
            (self.loader.gl_draw_buffers.unwrap())(
                2,
                colour_attachments.as_ptr(),
            );
            (self.loader.gl_clear_color.unwrap())(
                clear_color.r as f32 / 255.0,
                clear_color.g as f32 / 255.0,
                clear_color.b as f32 / 255.0,
                clear_color.a as f32 / 255.0,
            );
            (self.loader.gl_clear_depth.unwrap())(clear_depth as f64);
            (self.loader.gl_clear_stencil.unwrap())(clear_stencil);
            (self.loader.gl_clear.unwrap())(mask);
            if depth_test_enabled {
                (self.loader.gl_enable.unwrap())(GL_DEPTH_TEST);
                (self.loader.gl_depth_func.unwrap())(GL_LESS);
                (self.loader.gl_depth_mask.unwrap())(GL_TRUE);
            } else {
                (self.loader.gl_disable.unwrap())(GL_DEPTH_TEST);
                (self.loader.gl_depth_mask.unwrap())(GL_FALSE);
            }
        }
        {
            let (vao, index_count) = match self.meshes.get(mesh_id) {
                Some(Some(state)) => (state.vao, state.index_count),
                _ => {
                    self.violations += 1;
                    return;
                }
            };
            unsafe {
                (self.loader.gl_use_program.unwrap())(self.post_gbuffer_program);
                (self.loader.gl_uniform_matrix_4fv.unwrap())(
                    self.post_gbuffer_uniform_view_projection,
                    1,
                    GL_FALSE,
                    view_projection.m.as_ptr() as *const f32,
                );
                (self.loader.gl_uniform_matrix_4fv.unwrap())(
                    self.post_gbuffer_uniform_model,
                    1,
                    GL_FALSE,
                    model.m.as_ptr() as *const f32,
                );
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.post_gbuffer_uniform_sun_direction,
                    1,
                    block.sun_direction.as_ptr(),
                );
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.post_gbuffer_uniform_sun_color_intensity,
                    1,
                    block.sun_color_intensity.as_ptr(),
                );
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.post_gbuffer_uniform_point_position_radius,
                    4,
                    block.point_position_radius.as_ptr() as *const f32,
                );
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.post_gbuffer_uniform_point_color_intensity,
                    4,
                    block.point_color_intensity.as_ptr() as *const f32,
                );
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.post_gbuffer_uniform_params,
                    1,
                    block.params.as_ptr(),
                );
                // The volume's uniforms: the origin, the dims and the
                // params (the cell size, the indirect intensity and the
                // toggle) the POST_GBUFFER fragment's volume_sample
                // reads. The toggle's violation when the config asks for
                // the indirect light with no volume set.
                let volume_params = [
                    self.volume_uniform[8],
                    config.indirect_intensity,
                    if config.indirect_light { 1.0f32 } else { 0.0 },
                    0.0,
                ];
                if config.indirect_light {
                    match self.light_volume.as_ref() {
                        Some(_) => {}
                        None => {
                            self.violations += 1;
                        }
                    }
                }
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.post_gbuffer_uniform_volume_origin,
                    1,
                    self.volume_uniform[0..4].as_ptr(),
                );
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.post_gbuffer_uniform_volume_dims,
                    1,
                    self.volume_uniform[4..8].as_ptr(),
                );
                (self.loader.gl_uniform_4fv.unwrap())(
                    self.post_gbuffer_uniform_volume_params,
                    1,
                    volume_params.as_ptr(),
                );
                (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0);
                (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, texture);
                (self.loader.gl_uniform_1i.unwrap())(
                    self.post_gbuffer_uniform_sampler,
                    0,
                );
                // The volume's sampler: the 3D texture at unit 1 (the
                // POST_GBUFFER fragment's sampler3D declaration).
                let volume_texture = self
                    .light_volume
                    .as_ref()
                    .map(|t| t.texture)
                    .unwrap_or(0);
                (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0 + 1);
                (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_3D, volume_texture);
                (self.loader.gl_uniform_1i.unwrap())(
                    self.post_gbuffer_uniform_volume_sampler,
                    1,
                );
                (self.loader.gl_bind_vertex_array.unwrap())(vao);
                (self.loader.gl_draw_elements.unwrap())(
                    GL_TRIANGLES,
                    index_count as i32,
                    GL_UNSIGNED_INT,
                    std::ptr::null(),
                );
                (self.loader.gl_bind_vertex_array.unwrap())(0);
                // The sampler unit 0's restore (the composite pass's
                // radiance binding follows the same unit).
                (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0);
            }
        }
        // ---- Step two: the composite pass ----
        unsafe {
            (self.loader.gl_bind_framebuffer.unwrap())(GL_FRAMEBUFFER, 0);
            (self.loader.gl_draw_buffer.unwrap())(GL_COLOR_ATTACHMENT0);
            (self.loader.gl_clear_color.unwrap())(
                clear_color.r as f32 / 255.0,
                clear_color.g as f32 / 255.0,
                clear_color.b as f32 / 255.0,
                clear_color.a as f32 / 255.0,
            );
            (self.loader.gl_clear_depth.unwrap())(clear_depth as f64);
            (self.loader.gl_clear_stencil.unwrap())(clear_stencil);
            (self.loader.gl_clear.unwrap())(mask);
            // The depth test off: the fullscreen triangle covers the
            // frame (the default framebuffer's depth stays at the clear
            // value — the same rule the lighting pass follows).
            (self.loader.gl_disable.unwrap())(GL_DEPTH_TEST);
            (self.loader.gl_depth_mask.unwrap())(GL_FALSE);
            (self.loader.gl_use_program.unwrap())(self.post_composite_program);
            (self.loader.gl_uniform_4fv.unwrap())(
                self.post_composite_uniform_params0,
                1,
                [
                    if config.tonemapping { 1.0f32 } else { 0.0 },
                    if config.bloom { 1.0 } else { 0.0 },
                    if config.depth_of_field { 1.0 } else { 0.0 },
                    if config.ssao { 1.0 } else { 0.0 },
                ]
                .as_ptr(),
            );
            (self.loader.gl_uniform_4fv.unwrap())(
                self.post_composite_uniform_params1,
                1,
                [
                    config.bloom_threshold,
                    config.bloom_knee,
                    config.bloom_intensity,
                    config.ssao_radius,
                ]
                .as_ptr(),
            );
            (self.loader.gl_uniform_4fv.unwrap())(
                self.post_composite_uniform_params2,
                1,
                [
                    config.focal_distance,
                    config.focus_near_range,
                    config.focus_far_range,
                    if config.anti_aliasing { 1.0 } else { 0.0 },
                ]
                .as_ptr(),
            );
            (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0);
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, post_textures[0]);
            (self.loader.gl_uniform_1i.unwrap())(
                self.post_composite_uniform_radiance,
                0,
            );
            (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0 + 1);
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, post_textures[1]);
            (self.loader.gl_uniform_1i.unwrap())(
                self.post_composite_uniform_world,
                1,
            );
            (self.loader.gl_bind_vertex_array.unwrap())(self.post_composite_vao);
            (self.loader.gl_draw_arrays.unwrap())(GL_TRIANGLES, 0, 3);
            (self.loader.gl_bind_vertex_array.unwrap())(0);
            (self.loader.gl_active_texture.unwrap())(GL_TEXTURE0);
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_2D, texture);
            if depth_test_enabled {
                (self.loader.gl_enable.unwrap())(GL_DEPTH_TEST);
                (self.loader.gl_depth_func.unwrap())(GL_LESS);
                (self.loader.gl_depth_mask.unwrap())(GL_TRUE);
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
            // The MRT attachments: each view lands on its own
            // GL_COLOR_ATTACHMENTn (the bug the first deferred run
            // caught: every view landing on GL_COLOR_ATTACHMENT0 left
            // the geometry pass writing all three outputs into one
            // attachment).
            let colour_attachments = [
                GL_COLOR_ATTACHMENT0,
                GL_COLOR_ATTACHMENT1,
                GL_COLOR_ATTACHMENT2,
                GL_COLOR_ATTACHMENT3,
            ];
            for (i, view) in views.iter().enumerate() {
                (self.loader.gl_framebuffer_texture_2d.unwrap())(
                    GL_FRAMEBUFFER,
                    colour_attachments[i],
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
            // The draw-buffer list: the fragment shader's three outputs
            // (the draw call writes into the bound attachment list; the
            // default framebuffer's list restores at the end).
            if desc.color_attachment_count > 1 {
                (self.loader.gl_draw_buffers.unwrap())(
                    desc.color_attachment_count as i32,
                    colour_attachments.as_ptr(),
                );
            }
            (self.loader.gl_draw_buffer.unwrap())(GL_COLOR_ATTACHMENT0);
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
            return;
        }
        // The first sampler unit: the mesh path's draw binds the texture
        // (the later units record without the draw's use).
        if _unit != 0 {
            return;
        }
        match _view {
            None => {
                self.bound_texture = None;
            }
            Some(v) => {
                self.bound_texture = self
                    .textures_2d
                    .iter()
                    .position(|t| Arc::ptr_eq(&t.inner, &v.inner));
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

    fn create_texture_3d(&mut self, desc: &Texture3DDesc) -> Result<TextureView, RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "create_texture_3d: the backend is shut down",
            ));
        }
        if desc.width == 0 || desc.height == 0 || desc.depth == 0 {
            return Err(RenderError::from(
                "create_texture_3d: width, height and depth must be positive",
            ));
        }
        if desc.format == PixelFormat::Depth24Stencil8 {
            return Err(RenderError::from(
                "create_texture_3d: the depth/stencil format is not a colour texture",
            ));
        }
        let (internal, storage) = self.to_gl_format(desc.format);
        let mut texture: u32 = 0;
        unsafe {
            (self.loader.gl_gen_textures.unwrap())(1, &mut texture);
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_3D, texture);
            // The image's type follows the format: the Rgba16F texture's
            // half-float pixels (the unsigned-byte type would be an
            // invalid type/format pair for the RGBA16F internal format).
            let upload_type = match desc.format {
                PixelFormat::Rgba16F => GL_HALF_FLOAT,
                _ => GL_UNSIGNED_BYTE,
            };
            (self.loader.gl_tex_image_3d.unwrap())(
                GL_TEXTURE_3D,
                0,
                internal,
                desc.width as i32,
                desc.height as i32,
                desc.depth as i32,
                0,
                storage,
                upload_type,
                std::ptr::null(),
            );
            (self.loader.gl_tex_parameter_i.unwrap())(
                GL_TEXTURE_3D,
                GL_TEXTURE_MIN_FILTER,
                GL_NEAREST as i32,
            );
            (self.loader.gl_tex_parameter_i.unwrap())(
                GL_TEXTURE_3D,
                GL_TEXTURE_MAG_FILTER,
                GL_NEAREST as i32,
            );
            (self.loader.gl_tex_parameter_i.unwrap())(
                GL_TEXTURE_3D,
                GL_TEXTURE_WRAP_R,
                GL_REPEAT as i32,
            );
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_3D, 0);
        }
        let texture_obj = OpenGlTexture2D {
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
            inner: texture_obj.inner.clone(),
        };
        self.textures_3d.push(texture_obj);
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

    fn fill_texture_3d(&mut self, view: &TextureView, data: &[u8]) -> Result<(), RenderError> {
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
        let format = inner.format;
        let (internal, storage) = self.to_gl_format(format);
        let (width, height, depth) = (inner.width, inner.height, inner.depth);
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "fill_texture_3d: the byte count is {} but the texture is {}",
                data.len(),
                expected
            )));
        }
        unsafe {
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_3D, self.textures_3d[index].texture);
            (self.loader.gl_pixel_store_i.unwrap())(GL_UNPACK_ALIGNMENT, 1);
            // The upload's type follows the format: the Rgba16F pack's
            // binary16 half floats (the same data the Vulkan path's
            // staging buffer carries). The unsigned-byte type would be
            // an invalid type/format pair for the RGBA16F internal
            // format (the driver would reject the upload).
            let upload_type = match format {
                PixelFormat::Rgba16F => GL_HALF_FLOAT,
                _ => GL_UNSIGNED_BYTE,
            };
            (self.loader.gl_tex_image_3d.unwrap())(
                GL_TEXTURE_3D,
                0,
                internal,
                width as i32,
                height as i32,
                depth as i32,
                0,
                storage,
                upload_type,
                data.as_ptr() as *const c_void,
            );
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_3D, 0);
        }
        Ok(())
    }

    fn read_texture_3d(&mut self, view: &TextureView, data: &mut [u8]) -> Result<bool, RenderError> {
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
        let expected = (inner.width as usize)
            * (inner.height as usize)
            * (inner.depth as usize)
            * bytes_per_pixel(inner.format) as usize;
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "read_texture_3d: the byte count is {} but the texture is {}",
                data.len(),
                expected
            )));
        }
        unsafe {
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_3D, self.textures_3d[index].texture);
            (self.loader.gl_pixel_store_i.unwrap())(GL_UNPACK_ALIGNMENT, 1);
            // The Rgba16F texture's readback type: the pack's binary16
            // half floats (the same type the upload carries; the float
            // type would return 4x the byte count the contract carries).
            let read_type = if view.inner.lock().unwrap().format
                == PixelFormat::Rgba16F
            {
                GL_HALF_FLOAT
            } else {
                GL_UNSIGNED_BYTE
            };
            (self.loader.gl_get_tex_image.unwrap())(
                GL_TEXTURE_3D,
                0,
                GL_RGBA,
                read_type,
                data.as_mut_ptr() as *mut c_void,
            );
            (self.loader.gl_bind_texture.unwrap())(GL_TEXTURE_3D, 0);
        }
        Ok(true)
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
        self.light_volume = Some(OpenGlTexture2D {
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
