//! The OpenGL backend's loader: the engine's own loader, own code.
//!
//! RULE (lesson 3): no third-party binding. The platform's libEGL.so.1 is
//! dlopen()ed at runtime and the GL 3.3 core entry points resolve through
//! eglGetProcAddress (the EGL 1.5 core rule: entry points of the core
//! versions resolve through eglGetProcAddress on every EGL implementation).
//! Every constant and function-pointer type is declared here; the
//! platform-neutral contract never sees an EGL or GL type.

use std::os::raw::{c_char, c_void};

pub type ResolveFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;

pub const GL_TRUE: u8 = 1;
pub const GL_FALSE: u8 = 0;
pub const GL_UNPACK_ALIGNMENT: u32 = 0x0CF5;
pub const GL_PACK_ALIGNMENT: u32 = 0x0D05;

// -- GL constants (the values the published GL headers carry) --------------
pub const GL_VERSION: u32 = 0x1F02;
pub const GL_VENDOR: u32 = 0x1F00;
pub const GL_RENDERER: u32 = 0x1F01;
pub const GL_SHADING_LANGUAGE_VERSION: u32 = 0x8B8C;
pub const GL_COLOR: u32 = 0x1800;
pub const GL_DEPTH_COMPONENT: u32 = 0x1902;
pub const GL_STENCIL_INDEX: u32 = 0x1901;
pub const GL_UNSIGNED_BYTE: u32 = 0x1401;
pub const GL_UNSIGNED_INT: u32 = 0x1405;
pub const GL_FLOAT: u32 = 0x1406;
pub const GL_DEPTH_TEST: u32 = 0x0B71;
pub const GL_LESS: u32 = 0x0201;
pub const GL_RGBA: u32 = 0x1908;
pub const GL_RGBA8: u32 = 0x8058;
pub const GL_DEPTH_COMPONENT24: u32 = 0x81A6;
pub const GL_DEPTH24_STENCIL8: u32 = 0x88F0;
pub const GL_DEPTH_STENCIL: u32 = 0x84F9;
pub const GL_UNSIGNED_INT_24_8: u32 = 0x84FA;
pub const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
pub const GL_DEPTH_ATTACHMENT: u32 = 0x8D00;
pub const GL_STENCIL_ATTACHMENT: u32 = 0x8D20;
pub const GL_FRAMEBUFFER: u32 = 0x8D40;
pub const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;
pub const GL_TEXTURE_2D: u32 = 0x0DE1;
pub const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
pub const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
pub const GL_NEAREST: u32 = 0x2600;
pub const GL_LINEAR: u32 = 0x2601;
pub const GL_REPEAT: u32 = 0x2901;
pub const GL_TEXTURE_WRAP_S: u32 = 0x2802;
pub const GL_TEXTURE_WRAP_T: u32 = 0x2803;
pub const GL_RGB: u32 = 0x1907;
pub const GL_RED: u32 = 0x1903;
pub const GL_R8: u32 = 0x8229;
pub const GL_RGBA16F: u32 = 0x881A;
pub const GL_HALF_FLOAT: u32 = 0x140B;
pub const GL_TRIANGLE_LIST: u32 = 0;
pub const GL_TRIANGLES: u32 = 0x0004;
pub const GL_FLOAT_VEC4: u32 = 0x8B52;
pub const GL_ARRAY_BUFFER: u32 = 0x8892;
pub const GL_ELEMENT_ARRAY_BUFFER: u32 = 0x8893;
pub const GL_TEXTURE0: u32 = 0x84C0;
pub const GL_STATIC_DRAW: u32 = 0x88E4;
pub const GL_STREAM_DRAW: u32 = 0x88E0;
pub const GL_VERTEX_SHADER: u32 = 0x8B31;
pub const GL_FRAGMENT_SHADER: u32 = 0x8B30;
pub const GL_COMPILE_STATUS: u32 = 0x8B81;
pub const GL_LINK_STATUS: u32 = 0x8B82;
pub const GL_NO_ERROR: u32 = 0;
pub const GL_VERTEX_ARRAY_BINDING: u32 = 0x85B5;
pub const GL_COLOR_BUFFER_BIT: u32 = 0x4000;
pub const GL_DEPTH_BUFFER_BIT: u32 = 0x0100;
pub const GL_STENCIL_BUFFER_BIT: u32 = 0x0400;

// -- EGL constants (the values the published EGL headers carry) ------------
pub const EGL_SUCCESS: u32 = 0x3000;
pub const EGL_NOT_INITIALIZED: u32 = 0x3001;
pub const EGL_NO_SURFACE: *mut c_void = std::ptr::null_mut();
pub const EGL_NO_CONTEXT: *mut c_void = std::ptr::null_mut();
pub const EGL_NO_DISPLAY: *mut c_void = std::ptr::null_mut();
pub const EGL_DEFAULT_DISPLAY: *mut c_void = std::ptr::null_mut();
pub const EGL_SURFACE_TYPE: u32 = 0x3033;
pub const EGL_PBUFFER_BIT: u32 = 0x0001;
pub const EGL_RENDERABLE_TYPE: u32 = 0x3040;
pub const EGL_OPENGL_BIT: u32 = 0x0008;
pub const EGL_NONE: u32 = 0x3038;
pub const EGL_WIDTH: u32 = 0x3057;
pub const EGL_HEIGHT: u32 = 0x3056;
pub const EGL_OPENGL_API: u32 = 0x30A2;
pub const EGL_CONTEXT_MAJOR_VERSION: u32 = 0x3098;
pub const EGL_CONTEXT_MINOR_VERSION: u32 = 0x30FB;
pub const EGL_CONTEXT_OPENGL_PROFILE_MASK: u32 = 0x30FD;
pub const EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT: u32 = 0x0001;
pub const EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE: u32 = 0x31B1;
pub const EGL_PLATFORM_SURFACELESS_MESA: u32 = 0x31DD;
pub const EGL_PLATFORM_DEVICE_EXT: u32 = 0x313F;
pub const EGL_CLIENT_APIS: u32 = 0x308D;
pub const EGL_TRUE: u32 = 1;
pub const EGL_FALSE: u32 = 0;
pub const EGL_RED_SIZE: u32 = 0x3024;
pub const EGL_GREEN_SIZE: u32 = 0x3023;
pub const EGL_BLUE_SIZE: u32 = 0x3022;
pub const EGL_ALPHA_SIZE: u32 = 0x3021;
pub const EGL_DEPTH_SIZE: u32 = 0x3025;
pub const EGL_STENCIL_SIZE: u32 = 0x3026;

pub const EGL_MAX_CONFIG_ATTRIBS: usize = 32;

/// The EGL_SUCCESS return code, aliased for the calls the backend makes
/// through the surface API (which returns EGLBoolean).
pub const EGL_TRUE_U32: u32 = EGL_TRUE;

// -- Function pointer types (the published C signatures) -------------------
pub type PfnGlGetString =
    unsafe extern "C" fn(name: u32) -> *const u8;
pub type PfnGlGetError = unsafe extern "C" fn() -> u32;
pub type PfnGlGetIntegerv = unsafe extern "C" fn(pname: u32, params: *mut i32);
pub type PfnGlGetFloatv = unsafe extern "C" fn(pname: u32, params: *mut f32);
pub type PfnGlClearColor =
    unsafe extern "C" fn(r: f32, g: f32, b: f32, a: f32);
pub type PfnGlClearDepth = unsafe extern "C" fn(depth: f64);
pub type PfnGlClearStencil = unsafe extern "C" fn(s: i32);
pub type PfnGlClear = unsafe extern "C" fn(mask: u32);
pub type PfnGlViewport =
    unsafe extern "C" fn(x: i32, y: i32, w: i32, h: i32);
pub type PfnGlReadPixels = unsafe extern "C" fn(
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    format: u32,
    typ: u32,
    data: *mut c_void,
);
pub type PfnGlEnable = unsafe extern "C" fn(cap: u32);
pub type PfnGlDisable = unsafe extern "C" fn(cap: u32);
pub type PfnGlDepthFunc = unsafe extern "C" fn(func: u32);
pub type PfnGlDepthMask = unsafe extern "C" fn(flag: u8);
pub type PfnGlGenVertexArrays = unsafe extern "C" fn(n: i32, arrays: *mut u32);
pub type PfnGlBindVertexArray = unsafe extern "C" fn(array: u32);
pub type PfnGlDeleteVertexArrays =
    unsafe extern "C" fn(n: i32, arrays: *const u32);
pub type PfnGlGenBuffers = unsafe extern "C" fn(n: i32, buffers: *mut u32);
pub type PfnGlBindBuffer = unsafe extern "C" fn(target: u32, buffer: u32);
pub type PfnGlDeleteBuffers =
    unsafe extern "C" fn(n: i32, buffers: *const u32);
pub type PfnGlBufferData = unsafe extern "C" fn(
    target: u32,
    size: isize,
    data: *const c_void,
    usage: u32,
);
pub type PfnGlCreateShader = unsafe extern "C" fn(kind: u32) -> u32;
pub type PfnGlShaderSource = unsafe extern "C" fn(
    shader: u32,
    count: i32,
    source: *const *const c_char,
    length: *const i32,
);
pub type PfnGlCompileShader = unsafe extern "C" fn(shader: u32);
pub type PfnGlGetShaderiv =
    unsafe extern "C" fn(shader: u32, pname: u32, params: *mut i32);
pub type PfnGlGetShaderInfoLog = unsafe extern "C" fn(
    shader: u32,
    max_length: i32,
    length: *mut i32,
    log: *mut c_char,
);
pub type PfnGlDeleteShader = unsafe extern "C" fn(shader: u32);
pub type PfnGlCreateProgram = unsafe extern "C" fn() -> u32;
pub type PfnGlAttachShader =
    unsafe extern "C" fn(program: u32, shader: u32);
pub type PfnGlLinkProgram = unsafe extern "C" fn(program: u32);
pub type PfnGlGetProgramiv =
    unsafe extern "C" fn(program: u32, pname: u32, params: *mut i32);
pub type PfnGlGetProgramInfoLog = unsafe extern "C" fn(
    program: u32,
    max_length: i32,
    length: *mut i32,
    log: *mut c_char,
);
pub type PfnGlDeleteProgram = unsafe extern "C" fn(program: u32);
pub type PfnGlGetUniformLocation =
    unsafe extern "C" fn(program: u32, name: *const c_char) -> i32;
pub type PfnGlUniform1f = unsafe extern "C" fn(location: i32, v: f32);
pub type PfnGlUniform1i = unsafe extern "C" fn(location: i32, v: i32);
pub type PfnGlUniform4f =
    unsafe extern "C" fn(location: i32, a: f32, b: f32, c: f32, d: f32);
pub type PfnGlUniform4fv =
    unsafe extern "C" fn(location: i32, count: i32, value: *const f32);
pub type PfnGlUniformMatrix4fv = unsafe extern "C" fn(
    location: i32,
    count: i32,
    transpose: u8,
    value: *const f32,
);
pub type PfnGlVertexAttribPointer = unsafe extern "C" fn(
    index: u32,
    size: i32,
    typ: u32,
    normalized: u8,
    stride: i32,
    pointer: *const c_void,
);
pub type PfnGlEnableVertexAttribArray = unsafe extern "C" fn(index: u32);
pub type PfnGlDrawArrays =
    unsafe extern "C" fn(mode: u32, first: i32, count: i32);
pub type PfnGlDrawElements = unsafe extern "C" fn(
    mode: u32,
    count: i32,
    typ: u32,
    indices: *const c_void,
);
pub type PfnGlGenFramebuffers =
    unsafe extern "C" fn(n: i32, framebuffers: *mut u32);
pub type PfnGlBindFramebuffer =
    unsafe extern "C" fn(target: u32, framebuffer: u32);
pub type PfnGlDeleteFramebuffers =
    unsafe extern "C" fn(n: i32, framebuffers: *const u32);
pub type PfnGlFramebufferTexture2D = unsafe extern "C" fn(
    target: u32,
    attachment: u32,
    textarget: u32,
    texture: u32,
    level: i32,
);
pub type PfnGlCheckFramebufferStatus = unsafe extern "C" fn(target: u32) -> u32;
pub type PfnGlGenTextures = unsafe extern "C" fn(n: i32, textures: *mut u32);
pub type PfnGlBindTexture = unsafe extern "C" fn(target: u32, texture: u32);
pub type PfnGlDeleteTextures =
    unsafe extern "C" fn(n: i32, textures: *const u32);
pub type PfnGlTexImage2D = unsafe extern "C" fn(
    target: u32,
    level: i32,
    internal_format: i32,
    w: i32,
    h: i32,
    border: i32,
    format: u32,
    typ: u32,
    data: *const c_void,
);
pub type PfnGlPixelStorei = unsafe extern "C" fn(pname: u32, param: i32);
pub type PfnGlTexParameteri =
    unsafe extern "C" fn(target: u32, pname: u32, param: i32);
pub type PfnGlActiveTexture = unsafe extern "C" fn(unit: u32);
pub type PfnGlUseProgram = unsafe extern "C" fn(program: u32);
pub type PfnGlDrawBuffer = unsafe extern "C" fn(mode: u32);
pub type PfnGlReadBuffer = unsafe extern "C" fn(mode: u32);

// -- EGL function pointer types (the published C signatures) ---------------
pub type PfnEglGetError = unsafe extern "C" fn() -> u32;
pub type PfnEglInitialize = unsafe extern "C" fn(
    display: *mut c_void,
    major: *mut u32,
    minor: *mut u32,
) -> u32;
pub type PfnEglTerminate = unsafe extern "C" fn(display: *mut c_void);
pub type PfnEglChooseConfig =
    unsafe extern "C" fn(display: *mut c_void, attribs: *const i32,
                         configs: *mut *mut c_void, num: i32,
                         out: *mut i32) -> u32;
pub type PfnEglCreatePbufferSurface = unsafe extern "C" fn(
    display: *mut c_void,
    config: *mut c_void,
    attribs: *const i32,
) -> *mut c_void;
pub type PfnEglDestroySurface =
    unsafe extern "C" fn(display: *mut c_void, surface: *mut c_void) -> u32;
pub type PfnEglCreateContext = unsafe extern "C" fn(
    display: *mut c_void,
    config: *mut c_void,
    share: *mut c_void,
    attribs: *const i32,
) -> *mut c_void;
pub type PfnEglDestroyContext =
    unsafe extern "C" fn(display: *mut c_void, context: *mut c_void) -> u32;
pub type PfnEglMakeCurrent = unsafe extern "C" fn(
    display: *mut c_void,
    draw: *mut c_void,
    read: *mut c_void,
    context: *mut c_void,
) -> u32;
pub type PfnEglGetProcAddress =
    unsafe extern "C" fn(name: *const c_char) -> *mut c_void;
pub type PfnEglQueryString =
    unsafe extern "C" fn(display: *mut c_void, name: u32) -> *const c_char;
pub type PfnEglGetPlatformDisplayEXT = unsafe extern "C" fn(
    platform: u32,
    native_display: *mut c_void,
    attribs: *const i32,
) -> *mut c_void;
pub type PfnEglBindApi = unsafe extern "C" fn(api: u32) -> u32;
pub type PfnEglSwapBuffers =
    unsafe extern "C" fn(display: *mut c_void, surface: *mut c_void) -> u32;

/// The GL/EGL entry points the backend calls. Every field is a raw
/// resolved function pointer (lesson 3: the engine's own loader).
pub struct GlLoader {
    // EGL surface
    pub egl_get_error: Option<PfnEglGetError>,
    pub egl_initialize: Option<PfnEglInitialize>,
    pub egl_terminate: Option<PfnEglTerminate>,
    pub egl_choose_config: Option<PfnEglChooseConfig>,
    pub egl_create_pbuffer_surface: Option<PfnEglCreatePbufferSurface>,
    pub egl_destroy_surface: Option<PfnEglDestroySurface>,
    pub egl_create_context: Option<PfnEglCreateContext>,
    pub egl_destroy_context: Option<PfnEglDestroyContext>,
    pub egl_make_current: Option<PfnEglMakeCurrent>,
    pub egl_get_proc_address: Option<PfnEglGetProcAddress>,
    pub egl_query_string: Option<PfnEglQueryString>,
    pub egl_get_platform_display_ext: Option<PfnEglGetPlatformDisplayEXT>,
    pub egl_bind_api: Option<PfnEglBindApi>,
    pub egl_swap_buffers: Option<PfnEglSwapBuffers>,
    // GL core
    pub gl_get_string: Option<PfnGlGetString>,
    pub gl_get_error: Option<PfnGlGetError>,
    pub gl_get_integerv: Option<PfnGlGetIntegerv>,
    pub gl_clear_color: Option<PfnGlClearColor>,
    pub gl_clear_depth: Option<PfnGlClearDepth>,
    pub gl_clear_stencil: Option<PfnGlClearStencil>,
    pub gl_clear: Option<PfnGlClear>,
    pub gl_viewport: Option<PfnGlViewport>,
    pub gl_read_pixels: Option<PfnGlReadPixels>,
    pub gl_enable: Option<PfnGlEnable>,
    pub gl_disable: Option<PfnGlDisable>,
    pub gl_depth_func: Option<PfnGlDepthFunc>,
    pub gl_depth_mask: Option<PfnGlDepthMask>,
    pub gl_gen_vertex_arrays: Option<PfnGlGenVertexArrays>,
    pub gl_bind_vertex_array: Option<PfnGlBindVertexArray>,
    pub gl_delete_vertex_arrays: Option<PfnGlDeleteVertexArrays>,
    pub gl_gen_buffers: Option<PfnGlGenBuffers>,
    pub gl_bind_buffer: Option<PfnGlBindBuffer>,
    pub gl_delete_buffers: Option<PfnGlDeleteBuffers>,
    pub gl_buffer_data: Option<PfnGlBufferData>,
    pub gl_create_shader: Option<PfnGlCreateShader>,
    pub gl_shader_source: Option<PfnGlShaderSource>,
    pub gl_compile_shader: Option<PfnGlCompileShader>,
    pub gl_get_shaderiv: Option<PfnGlGetShaderiv>,
    pub gl_get_shader_info_log: Option<PfnGlGetShaderInfoLog>,
    pub gl_delete_shader: Option<PfnGlDeleteShader>,
    pub gl_create_program: Option<PfnGlCreateProgram>,
    pub gl_attach_shader: Option<PfnGlAttachShader>,
    pub gl_link_program: Option<PfnGlLinkProgram>,
    pub gl_get_programiv: Option<PfnGlGetProgramiv>,
    pub gl_get_program_info_log: Option<PfnGlGetProgramInfoLog>,
    pub gl_delete_program: Option<PfnGlDeleteProgram>,
    pub gl_get_uniform_location: Option<PfnGlGetUniformLocation>,
    pub gl_uniform_1f: Option<PfnGlUniform1f>,
    pub gl_uniform_1i: Option<PfnGlUniform1i>,
    pub gl_uniform_4f: Option<PfnGlUniform4f>,
    pub gl_uniform_4fv: Option<PfnGlUniform4fv>,
    pub gl_uniform_matrix_4fv: Option<PfnGlUniformMatrix4fv>,
    pub gl_vertex_attrib_pointer: Option<PfnGlVertexAttribPointer>,
    pub gl_enable_vertex_attrib_array: Option<PfnGlEnableVertexAttribArray>,
    pub gl_draw_arrays: Option<PfnGlDrawArrays>,
    pub gl_draw_elements: Option<PfnGlDrawElements>,
    pub gl_gen_framebuffers: Option<PfnGlGenFramebuffers>,
    pub gl_bind_framebuffer: Option<PfnGlBindFramebuffer>,
    pub gl_delete_framebuffers: Option<PfnGlDeleteFramebuffers>,
    pub gl_framebuffer_texture_2d: Option<PfnGlFramebufferTexture2D>,
    pub gl_check_framebuffer_status: Option<PfnGlCheckFramebufferStatus>,
    pub gl_gen_textures: Option<PfnGlGenTextures>,
    pub gl_bind_texture: Option<PfnGlBindTexture>,
    pub gl_delete_textures: Option<PfnGlDeleteTextures>,
    pub gl_tex_image_2d: Option<PfnGlTexImage2D>,
    pub gl_pixel_store_i: Option<PfnGlPixelStorei>,
    pub gl_tex_parameter_i: Option<PfnGlTexParameteri>,
    pub gl_active_texture: Option<PfnGlActiveTexture>,
    pub gl_use_program: Option<PfnGlUseProgram>,
    pub gl_draw_buffer: Option<PfnGlDrawBuffer>,
    pub gl_read_buffer: Option<PfnGlReadBuffer>,
}

impl GlLoader {
    pub fn new() -> Self {
        GlLoader {
            egl_get_error: None,
            egl_initialize: None,
            egl_terminate: None,
            egl_choose_config: None,
            egl_create_pbuffer_surface: None,
            egl_destroy_surface: None,
            egl_create_context: None,
            egl_destroy_context: None,
            egl_make_current: None,
            egl_get_proc_address: None,
            egl_query_string: None,
            egl_get_platform_display_ext: None,
            egl_bind_api: None,
            egl_swap_buffers: None,
            gl_get_string: None,
            gl_get_error: None,
            gl_get_integerv: ::std::option::Option::None,
            gl_clear_color: None,
            gl_clear_depth: None,
            gl_clear_stencil: None,
            gl_clear: None,
            gl_viewport: None,
            gl_read_pixels: None,
            gl_enable: None,
            gl_disable: None,
            gl_depth_func: None,
            gl_depth_mask: None,
            gl_gen_vertex_arrays: None,
            gl_bind_vertex_array: None,
            gl_delete_vertex_arrays: None,
            gl_gen_buffers: None,
            gl_bind_buffer: None,
            gl_delete_buffers: None,
            gl_buffer_data: None,
            gl_create_shader: None,
            gl_shader_source: None,
            gl_compile_shader: None,
            gl_get_shaderiv: None,
            gl_get_shader_info_log: None,
            gl_delete_shader: None,
            gl_create_program: None,
            gl_attach_shader: None,
            gl_link_program: None,
            gl_get_programiv: None,
            gl_get_program_info_log: None,
            gl_delete_program: None,
            gl_get_uniform_location: None,
            gl_uniform_1f: None,
            gl_uniform_1i: None,
            gl_uniform_4f: None,
            gl_uniform_4fv: None,
            gl_uniform_matrix_4fv: None,
            gl_vertex_attrib_pointer: None,
            gl_enable_vertex_attrib_array: None,
            gl_draw_arrays: None,
            gl_draw_elements: None,
            gl_gen_framebuffers: None,
            gl_bind_framebuffer: None,
            gl_delete_framebuffers: None,
            gl_framebuffer_texture_2d: None,
            gl_check_framebuffer_status: None,
            gl_gen_textures: None,
            gl_bind_texture: None,
            gl_delete_textures: None,
            gl_tex_image_2d: None,
            gl_pixel_store_i: None,
            gl_tex_parameter_i: None,
            gl_active_texture: None,
            gl_use_program: None,
            gl_draw_buffer: None,
            gl_read_buffer: None,
        }
    }
}

impl Default for GlLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// dlopen()s the platform's EGL library (libEGL.so.1) and resolves the
/// EGL surface through eglGetProcAddress; then resolves the GL core.
///
/// The EGL surfaceless path is a POSIX shape (libEGL.so.1 is a
/// dlopen()ed library): on Windows the load fails cleanly with the
/// platform's own message and the backend reports it (the build-only
/// status the validation matrix records; a WGL path would be its own
/// milestone). The loader's other functions stay platform-neutral, so
/// the Windows build links the module unchanged.
#[cfg(not(windows))]
pub fn load_library() -> Result<*mut c_void, String> {
    unsafe {
        let handle = libc::dlopen(
            b"libEGL.so.1\0".as_ptr() as *const c_char,
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        );
        if handle.is_null() {
            return Err(format!(
                "OpenGlBackend: dlopen libEGL.so.1 failed: {}",
                dl_error_string()
            ));
        }
        Ok(handle)
    }
}

#[cfg(windows)]
pub fn load_library() -> Result<*mut c_void, String> {
    Err(
        "OpenGlBackend: the EGL surfaceless path does not exist on \
         Windows (the build-only status; the WGL path is its own \
         milestone)"
            .to_string(),
    )
}

#[cfg(not(windows))]
pub unsafe fn get_library_proc(
    library: *mut c_void,
    name: &str,
) -> *mut c_void {
    let name_c = std::ffi::CString::new(name).unwrap();
    libc::dlsym(library, name_c.as_ptr())
}

#[cfg(windows)]
pub unsafe fn get_library_proc(
    _library: *mut c_void,
    _name: &str,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[cfg(not(windows))]
unsafe fn dl_error_string() -> String {
    let err = libc::dlerror();
    if err.is_null() {
        "unknown error".to_string()
    } else {
        std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned()
    }
}


impl GlLoader {
    /// Resolves the EGL surface through the library handle, then the GL
    /// core through eglGetProcAddress (the EGL core rule: GL entry points
    /// resolve through eglGetProcAddress on every EGL implementation).
    /// `missing` carries the first name the loader could not resolve.
    pub fn load(
        &mut self,
        library: *mut c_void,
        missing: &mut String,
    ) -> bool {
        // The EGL surface resolves through dlsym on the library handle
        // (the get_library_proc call carries the platform's gate: on
        // Windows the load failed before this runs and the function is
        // the null stub).
        macro_rules! load_egl {
            ($field:ident, $name:literal) => {
                if self.$field.is_none() {
                    unsafe {
                        let symbol = get_library_proc(
                            library,
                            $name,
                        );
                        if symbol.is_null() {
                            *missing = $name.to_string();
                            return false;
                        }
                        self.$field = Some(
                            std::mem::transmute::<*mut c_void, _>(symbol),
                        );
                    }
                }
            };
        }
        // The GL core resolves through eglGetProcAddress.
        macro_rules! load_gl {
            ($field:ident, $name:literal) => {
                if self.$field.is_none() {
                    unsafe {
                        let symbol = match self.egl_get_proc_address {
                            Some(proc_addr) => proc_addr(
                                concat!($name, "\0").as_ptr() as *const c_char,
                            ),
                            None => std::ptr::null_mut(),
                        };
                        if symbol.is_null() {
                            *missing = $name.to_string();
                            return false;
                        }
                        self.$field = Some(
                            std::mem::transmute::<*mut c_void, _>(symbol),
                        );
                    }
                }
            };
        }
        load_egl!(egl_get_error, "eglGetError");
        load_egl!(egl_initialize, "eglInitialize");
        load_egl!(egl_terminate, "eglTerminate");
        load_egl!(egl_choose_config, "eglChooseConfig");
        load_egl!(egl_create_pbuffer_surface, "eglCreatePbufferSurface");
        load_egl!(egl_destroy_surface, "eglDestroySurface");
        load_egl!(egl_create_context, "eglCreateContext");
        load_egl!(egl_destroy_context, "eglDestroyContext");
        load_egl!(egl_make_current, "eglMakeCurrent");
        load_egl!(egl_get_proc_address, "eglGetProcAddress");
        load_egl!(egl_query_string, "eglQueryString");
        load_egl!(egl_bind_api, "eglBindAPI");
        load_egl!(egl_swap_buffers, "eglSwapBuffers");
        // eglGetPlatformDisplayEXT is an extension entry point: the EGL
        // dispatcher (libglvnd) does not export it through dlsym on every
        // platform, so the loader falls back to eglGetProcAddress (the
        // extension resolution rule).
        if self.egl_get_platform_display_ext.is_none() {
            unsafe {
                let symbol = self.egl_get_proc_address.unwrap()(
                    b"eglGetPlatformDisplayEXT\0".as_ptr() as *const c_char,
                );
                if symbol.is_null() {
                    *missing = "eglGetPlatformDisplayEXT".to_string();
                    return false;
                }
                self.egl_get_platform_display_ext = Some(
                    std::mem::transmute::<*mut c_void, _>(symbol),
                );
            }
        }
        load_gl!(gl_get_string, "glGetString");
        load_gl!(gl_get_error, "glGetError");
        load_gl!(gl_get_integerv, "glGetIntegerv");
        load_gl!(gl_clear_color, "glClearColor");
        load_gl!(gl_clear_depth, "glClearDepth");
        load_gl!(gl_clear_stencil, "glClearStencil");
        load_gl!(gl_clear, "glClear");
        load_gl!(gl_viewport, "glViewport");
        load_gl!(gl_read_pixels, "glReadPixels");
        load_gl!(gl_enable, "glEnable");
        load_gl!(gl_disable, "glDisable");
        load_gl!(gl_depth_func, "glDepthFunc");
        load_gl!(gl_depth_mask, "glDepthMask");
        load_gl!(gl_gen_vertex_arrays, "glGenVertexArrays");
        load_gl!(gl_bind_vertex_array, "glBindVertexArray");
        load_gl!(gl_delete_vertex_arrays, "glDeleteVertexArrays");
        load_gl!(gl_gen_buffers, "glGenBuffers");
        load_gl!(gl_bind_buffer, "glBindBuffer");
        load_gl!(gl_delete_buffers, "glDeleteBuffers");
        load_gl!(gl_buffer_data, "glBufferData");
        load_gl!(gl_create_shader, "glCreateShader");
        load_gl!(gl_shader_source, "glShaderSource");
        load_gl!(gl_compile_shader, "glCompileShader");
        load_gl!(gl_get_shaderiv, "glGetShaderiv");
        load_gl!(gl_get_shader_info_log, "glGetShaderInfoLog");
        load_gl!(gl_delete_shader, "glDeleteShader");
        load_gl!(gl_create_program, "glCreateProgram");
        load_gl!(gl_attach_shader, "glAttachShader");
        load_gl!(gl_link_program, "glLinkProgram");
        load_gl!(gl_get_programiv, "glGetProgramiv");
        load_gl!(gl_get_program_info_log, "glGetProgramInfoLog");
        load_gl!(gl_delete_program, "glDeleteProgram");
        load_gl!(gl_get_uniform_location, "glGetUniformLocation");
        load_gl!(gl_uniform_1f, "glUniform1f");
        load_gl!(gl_uniform_1i, "glUniform1i");
        load_gl!(gl_uniform_4f, "glUniform4f");
        load_gl!(gl_uniform_4fv, "glUniform4fv");
        load_gl!(gl_uniform_matrix_4fv, "glUniformMatrix4fv");
        load_gl!(gl_vertex_attrib_pointer, "glVertexAttribPointer");
        load_gl!(gl_enable_vertex_attrib_array, "glEnableVertexAttribArray");
        load_gl!(gl_draw_arrays, "glDrawArrays");
        load_gl!(gl_draw_elements, "glDrawElements");
        load_gl!(gl_gen_framebuffers, "glGenFramebuffers");
        load_gl!(gl_bind_framebuffer, "glBindFramebuffer");
        load_gl!(gl_delete_framebuffers, "glDeleteFramebuffers");
        load_gl!(gl_framebuffer_texture_2d, "glFramebufferTexture2D");
        load_gl!(gl_check_framebuffer_status, "glCheckFramebufferStatus");
        load_gl!(gl_gen_textures, "glGenTextures");
        load_gl!(gl_bind_texture, "glBindTexture");
        load_gl!(gl_delete_textures, "glDeleteTextures");
        load_gl!(gl_tex_image_2d, "glTexImage2D");
        load_gl!(gl_pixel_store_i, "glPixelStorei");
        load_gl!(gl_tex_parameter_i, "glTexParameteri");
        load_gl!(gl_active_texture, "glActiveTexture");
        load_gl!(gl_use_program, "glUseProgram");
        load_gl!(gl_draw_buffer, "glDrawBuffer");
        load_gl!(gl_read_buffer, "glReadBuffer");
        true
    }
}
