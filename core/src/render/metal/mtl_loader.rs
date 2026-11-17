//! The Metal backend's loader: the engine's own loader, own code.
//!
//! RULE (lesson 3): every constant, structure and function signature the
//! engine uses is declared here, so no Apple SDK header and no third-party
//! framework join the tree, and the platform-neutral contract never
//! includes it.
//!
//! Metal has no C entry points to resolve: it is an Objective-C framework.
//! The loader therefore declares the Objective-C runtime's own C entry
//! points (objc_getClass, sel_registerName, objc_msgSend) and talks to the
//! framework through them, with every Metal class handle an opaque pointer
//! and every call a runtime message send. The enum values are the
//! published API's values (the M19 lesson: the values the device receives
//! must be the API's own, never a local guess).
//!
//! On a non-Apple platform the loader still compiles (the declarations are
//! plain C), but the framework's dlopen fails and `load()` reports that
//! cleanly: the mapping unit tests run on Linux, the hardware path needs a
//! Mac (the M20 recorded scope).

use std::os::raw::{c_char, c_void};

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

// Metal handles are opaque Objective-C objects. Every handle is a plain
// pointer; the runtime never dereferences one through this module.
pub type MtlDeviceHandle = *mut c_void;
pub type MtlCommandQueueHandle = *mut c_void;
pub type MtlCommandBufferHandle = *mut c_void;
pub type MtlRenderCommandEncoderHandle = *mut c_void;
pub type MTLLibraryHandle = *mut c_void;
pub type MtlFunctionHandle = *mut c_void;
pub type MtlRenderPipelineStateHandle = *mut c_void;
pub type MtlDepthStencilStateHandle = *mut c_void;
pub type MtlSamplerStateHandle = *mut c_void;
pub type MtlBufferHandle = *mut c_void;
pub type MtlTextureHandle = *mut c_void;
pub type MtlClassHandle = *mut c_void;
// NSUInteger is the framework's own index/size/count type: 64-bit on every
// architecture Metal supports.
pub type NsUInteger = u64;

// ---------------------------------------------------------------------------
// Published enum values (the published API's own values)
// ---------------------------------------------------------------------------

pub const MTLPX_INVALID: NsUInteger = 0;
pub const MTLPX_R8_UNORM: NsUInteger = 10;
pub const MTLPX_RGBA8_UNORM: NsUInteger = 70;
pub const MTLPX_BGRA8_UNORM: NsUInteger = 80;
pub const MTLPX_RGBA16_FLOAT: NsUInteger = 115;
pub const MTLPX_DEPTH32_FLOAT: NsUInteger = 252;
pub const MTLPX_STENCIL8: NsUInteger = 253;
pub const MTLPX_DEPTH24_UNORM_STENCIL8: NsUInteger = 255;
pub const MTLPX_DEPTH32_FLOAT_STENCIL8: NsUInteger = 260;

pub const MTLCF_NEVER: NsUInteger = 0;
pub const MTLCF_LESS: NsUInteger = 1;
pub const MTLCF_EQUAL: NsUInteger = 2;
pub const MTLCF_LESS_EQUAL: NsUInteger = 3;
pub const MTLCF_GREATER: NsUInteger = 4;
pub const MTLCF_NOT_EQUAL: NsUInteger = 5;
pub const MTLCF_GREATER_EQUAL: NsUInteger = 6;
pub const MTLCF_ALWAYS: NsUInteger = 7;

pub const MTLLA_DONT_CARE: NsUInteger = 0;
pub const MTLLA_LOAD: NsUInteger = 1;
pub const MTLLA_CLEAR: NsUInteger = 2;

pub const MTLSA_DONT_CARE: NsUInteger = 0;
pub const MTLSA_STORE: NsUInteger = 1;

pub const MTLPT_TRIANGLE: NsUInteger = 3;

pub const MTLIT_UINT32: NsUInteger = 1;

pub const MTLVF_FLOAT3: NsUInteger = 30;
pub const MTLVF_FLOAT2: NsUInteger = 29;
pub const MTLSF_CONSTANT: NsUInteger = 0;
pub const MTLSF_PER_VERTEX: NsUInteger = 1;

pub const MTLSM_SHARED: NsUInteger = 0;
pub const MTLSM_MANAGED: NsUInteger = 1;
pub const MTLSM_PRIVATE: NsUInteger = 2;

pub const MTLCU_SHADER_READ: NsUInteger = 1;
pub const MTLCU_RENDER_TARGET: NsUInteger = 4;

pub const MTLLV_2_0: NsUInteger = 2 << 8;

// The published resource options: the CPU cache mode at bits 0-3, the
// storage mode at bits 4-7 and the hazard tracking mode at bits 8-9
// (MTLResource.h's published shift macros). The backend builds the value
// from a storage mode plus the default cache mode.
pub const MTLRO_SHARED: NsUInteger = MTLSM_SHARED << 4;
pub const MTLRO_PRIVATE: NsUInteger = MTLSM_PRIVATE << 4;
// The published blit options: the combined depth/stencil formats'
// buffer copies name the plane the copy carries through the
// copy(...options:) variant (the plain copy form is the API's
// forbidden form for these formats — the hardware run's zero
// depth/stencil readback caught it).
pub const MTLBLOPT_DEPTH_FROM_DEPTH_STENCIL: NsUInteger = 1;
pub const MTLBLOPT_STENCIL_FROM_DEPTH_STENCIL: NsUInteger = 2;
// The published sampler filter states (MTLSamplerMinMagFilter's own
// values: nearest 0, linear 1) and the sampler address modes
// (MTLSamplerAddressMode: clamp to edge 0). The mesh path's sampler is
// NEAREST with the clamp-to-edge address mode.
pub const MTLSMF_NEAREST: NsUInteger = 0;
pub const MTLSMF_LINEAR: NsUInteger = 1;
pub const MTLSAM_CLAMP_TO_EDGE: NsUInteger = 0;
// The published winding rule (MTLWinding: counter-clockwise 0) and the
// cull mode (MTLCullMode: none 0). The mesh path's first step draws
// both faces (no culling, the same forgiving rule the Vulkan path's
// first step records).
pub const MTLWINDING_COUNTER_CLOCKWISE: NsUInteger = 0;
pub const MTLCULL_NONE: NsUInteger = 0;

// ---------------------------------------------------------------------------
// Fixed-size structures (the published C layouts)
// ---------------------------------------------------------------------------

/// The published struct: four 64-bit doubles (clear colours are doubles on
/// the Metal ABI).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MtlClearColor {
    pub red: f64,
    pub green: f64,
    pub blue: f64,
    pub alpha: f64,
}

/// The published struct: six 64-bit doubles (origin, size, depthRange).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MtlViewport {
    pub origin_x: f64,
    pub origin_y: f64,
    pub width: f64,
    pub height: f64,
    pub znear: f64,
    pub zfar: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MtlOrigin {
    pub x: NsUInteger,
    pub y: NsUInteger,
    pub z: NsUInteger,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MtlSize {
    pub width: NsUInteger,
    pub height: NsUInteger,
    pub depth: NsUInteger,
}

/// The published texture coordinate struct the blit's copy region takes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MtlRegion {
    pub origin: MtlOrigin,
    pub size: MtlSize,
}

// ---------------------------------------------------------------------------
// The Objective-C runtime's entry points (the platform's public runtime ABI)
// ---------------------------------------------------------------------------

pub type FnObjcGetClass =
    unsafe extern "C" fn(name: *const c_char) -> *mut c_void;
pub type FnSelRegisterName =
    unsafe extern "C" fn(name: *const c_char) -> *mut c_void;
// objc_msgSend's real signature is the ABI's own variadic trampoline: a
// call through it casts to the target method's signature. The loader
// stores the raw pointer and the call sites cast (the documented pattern
// for calling Objective-C from C code without headers).
pub type FnObjcMsgSend = unsafe extern "C" fn(*mut c_void, *mut c_void);

/// The Metal loader: resolves the Objective-C runtime and the Metal
/// framework's entry points at runtime (dlopen/dlsym; no link-time
/// dependency, the same rule the OpenGL and the Vulkan loaders record).
/// On a platform without the framework, `load()` reports false and
/// `missing` names the first symbol that could not be resolved.
pub struct MtlLoader {
    pub objc_get_class: Option<FnObjcGetClass>,
    pub sel_register_name: Option<FnSelRegisterName>,
    pub objc_msg_send: Option<FnObjcMsgSend>,
    // The one published C function Metal exposes.
    pub mtl_create_system_default_device: Option<
        unsafe extern "C" fn() -> MtlDeviceHandle,
    >,
    // The class handles the backend allocates descriptor objects from
    // (null until load() resolved them).
    pub cls_mtl_texture_descriptor: MtlClassHandle,
    pub cls_mtl_render_pipeline_descriptor: MtlClassHandle,
    pub cls_mtl_depth_stencil_descriptor: MtlClassHandle,
    pub cls_mtl_vertex_descriptor: MtlClassHandle,
    pub cls_mtl_compile_options: MtlClassHandle,
    pub cls_mtl_render_pass_descriptor: MtlClassHandle,
    pub cls_mtl_sampler_descriptor: MtlClassHandle,
    pub loaded: bool,
}

impl MtlLoader {
    pub fn new() -> Self {
        MtlLoader {
            objc_get_class: None,
            sel_register_name: None,
            objc_msg_send: None,
            mtl_create_system_default_device: None,
            cls_mtl_texture_descriptor: std::ptr::null_mut(),
            cls_mtl_render_pipeline_descriptor: std::ptr::null_mut(),
            cls_mtl_depth_stencil_descriptor: std::ptr::null_mut(),
            cls_mtl_vertex_descriptor: std::ptr::null_mut(),
            cls_mtl_compile_options: std::ptr::null_mut(),
            cls_mtl_render_pass_descriptor: std::ptr::null_mut(),
            cls_mtl_sampler_descriptor: std::ptr::null_mut(),
            loaded: false,
        }
    }

    /// The dlopen path attempt. On Apple platforms the framework loads;
    /// elsewhere it reports false with a clear missing-symbol message.
    /// The Windows build links the module but the load fails cleanly
    /// before any dlopen (no libobjc, no Metal.framework there; the
    /// build-only status the validation matrix records).
    #[cfg(not(windows))]
    pub fn load(&mut self, missing: &mut String) -> bool {
        if self.loaded {
            return true;
        }
        // The Objective-C runtime: on Apple platforms libobjc is part of
        // the system and already loaded into every process, so dlopen of
        // the well-known path succeeds trivially; the dlsym calls resolve
        // the three entry points the backend sends messages through.
        unsafe {
            let runtime = libc::dlopen(
                b"/usr/lib/libobjc.dylib\0".as_ptr() as *const c_char,
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            );
            if runtime.is_null() {
                *missing = "Objective-C runtime (dlopen)".to_string();
                return false;
            }
            macro_rules! resolve {
                ($field:ident, $name:literal) => {
                    let symbol = libc::dlsym(
                        runtime,
                        concat!($name, "\0").as_ptr() as *const c_char,
                    );
                    if symbol.is_null() {
                        *missing = $name.to_string();
                        return false;
                    }
                    self.$field = Some(
                        std::mem::transmute::<*mut c_void, _>(symbol),
                    );
                };
            }
            resolve!(objc_get_class, "objc_getClass");
            resolve!(sel_register_name, "sel_registerName");
            resolve!(objc_msg_send, "objc_msgSend");
            // The Metal framework itself. On Apple platforms the
            // framework is present; on Linux (and any non-Apple host) the
            // dlopen fails with the platform's own error, and the
            // message names it.
            let framework = libc::dlopen(
                b"/System/Library/Frameworks/Metal.framework/Metal\0"
                    .as_ptr() as *const c_char,
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            );
            if framework.is_null() {
                *missing =
                    "Metal.framework (dlopen: no Metal on this platform)"
                        .to_string();
                return false;
            }
            let create_device = libc::dlsym(
                framework,
                b"MTLCreateSystemDefaultDevice\0".as_ptr() as *const c_char,
            );
            if create_device.is_null() {
                *missing = "MTLCreateSystemDefaultDevice".to_string();
                return false;
            }
            self.mtl_create_system_default_device = Some(
                std::mem::transmute::<*mut c_void, _>(create_device),
            );
            // The framework's classes the backend allocates descriptor
            // objects from (a class not present means a framework too old
            // for the backend; the missing-symbol message names it).
            macro_rules! class {
                ($field:ident, $name:literal) => {
                    let handle = self.objc_get_class.unwrap()(
                        concat!($name, "\0").as_ptr() as *const c_char,
                    );
                    if handle.is_null() {
                        *missing = $name.to_string();
                        return false;
                    }
                    self.$field = handle;
                };
            }
            class!(cls_mtl_texture_descriptor, "MTLTextureDescriptor");
            class!(
                cls_mtl_render_pipeline_descriptor,
                "MTLRenderPipelineDescriptor"
            );
            class!(
                cls_mtl_depth_stencil_descriptor,
                "MTLDepthStencilDescriptor"
            );
            class!(cls_mtl_vertex_descriptor, "MTLVertexDescriptor");
            class!(cls_mtl_compile_options, "MTLCompileOptions");
            class!(cls_mtl_render_pass_descriptor, "MTLRenderPassDescriptor");
            class!(cls_mtl_sampler_descriptor, "MTLSamplerDescriptor");
        }
        self.loaded = true;
        true
    }

    #[cfg(windows)]
    pub fn load(&mut self, missing: &mut String) -> bool {
        *missing = "the Objective-C runtime (no Metal on Windows)".to_string();
        false
    }

    /// Registers a selector by name (the backend's sel_registerName
    /// wrapper; the selector string is interned once).
    pub fn sel(&self, name: &str) -> *mut c_void {
        let name = std::ffi::CString::new(name).unwrap();
        unsafe { self.sel_register_name.unwrap()(name.as_ptr()) }
    }

    /// The message-send helpers: each is a thin transmute of objc_msgSend
    /// to the method's real signature, so the call sites read like the
    /// API's own names.

    pub unsafe fn send0(&self, self_: *mut c_void, sel: *mut c_void)
        -> *mut c_void {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void)
            -> *mut c_void = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel)
    }

    pub unsafe fn send_ret_usize(&self, self_: *mut c_void, sel: *mut c_void)
        -> NsUInteger {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void) -> NsUInteger =
            std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel)
    }

    pub unsafe fn send_ret_double(&self, self_: *mut c_void, sel: *mut c_void)
        -> f64 {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void) -> f64 =
            std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel)
    }

    pub unsafe fn send_ret_bool(&self, self_: *mut c_void, sel: *mut c_void)
        -> bool {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool =
            std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel)
    }

    pub unsafe fn send_ret_ptr(&self, self_: *mut c_void, sel: *mut c_void)
        -> *mut c_void {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void)
            -> *mut c_void = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel)
    }

    /// The message-send form `id objc_msgSend(id self, SEL op, id arg)`:
    /// one object argument, one object return (the device's
    /// newTextureWithDescriptor:, newFunctionWithName:, and friends).
    pub unsafe fn send_ptr_arg(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg: *mut c_void,
    ) -> *mut c_void {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> *mut c_void = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg)
    }

    /// The message-send form `id objc_msgSend(id self, SEL op, id arg1,
    /// id arg2, id arg3)`: three object arguments, one object return
    /// (the device's newLibraryWithSource:options:error:, whose third
    /// argument is the NSError** out-slot the caller passes by pointer).
    pub unsafe fn send_ptr_arg3(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg1: *mut c_void,
        arg2: *mut c_void,
        arg3: *mut c_void,
    ) -> *mut c_void {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> *mut c_void = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg1, arg2, arg3)
    }

    /// The message-send form `id objc_msgSend(id self, SEL op, id arg1,
    /// void* arg2)`: one object argument and one pointer (the device's
    /// newRenderPipelineStateWithDescriptor:error:, whose second
    /// argument is the NSError** out-slot the caller passes by
    /// pointer — the API's only documented synchronous pipeline
    /// creation; the plain descriptor-only form does not exist in the
    /// protocol's headers).
    pub unsafe fn send_ptr_arg2(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg1: *mut c_void,
        arg2: *mut c_void,
    ) -> *mut c_void {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> *mut c_void = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg1, arg2)
    }

    pub unsafe fn send_void(&self, self_: *mut c_void, sel: *mut c_void) {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void) =
            std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel)
    }

    pub unsafe fn send_void_ptr(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg: *mut c_void,
    ) {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) =
            std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg)
    }

    pub unsafe fn send_void_ul(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg: NsUInteger,
    ) {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            NsUInteger,
        ) = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg);
    }

    /// The message-send form `void objc_msgSend(id self, SEL op, BOOL
    /// arg)`: one BOOL (the depth/stencil descriptor's
    /// setDepthWriteEnabled:, the Objective-C BOOL is one byte).
    pub unsafe fn send_void_bool(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg: bool,
    ) {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void, u8) =
            std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, if arg { 1 } else { 0 });
    }

    /// The message-send form `void objc_msgSend(id self, SEL op, double
    /// arg)`: one double argument (the depth attachment's clear depth).
    pub unsafe fn send_void_double(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg: f64,
    ) {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void, f64) =
            std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg);
    }

    /// The message-send form `void objc_msgSend(id self, SEL op,
    /// MTLClearColor arg)`: the clear colour's four-double struct. The
    /// struct is a homogeneous floating-point aggregate, so the arm64
    /// ABI passes it in the vector registers (v0..v3) the C ABI's own
    /// rule gives it.
    pub unsafe fn send_void_clear_color(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg: MtlClearColor,
    ) {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            MtlClearColor,
        ) = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg);
    }

    /// The message-send form `void objc_msgSend(id self, SEL op, id arg,
    /// NSUInteger arg2, NSUInteger arg3)`: one object and two integers
    /// (the buffer bindings' setVertexBuffer:offset:atIndex: and
    /// setFragmentBuffer:offset:atIndex:).
    pub unsafe fn send_void_ptr_ul2(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg1: *mut c_void,
        arg2: NsUInteger,
        arg3: NsUInteger,
    ) {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            NsUInteger,
            NsUInteger,
        ) = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg1, arg2, arg3);
    }

    pub unsafe fn send_void_ul2(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg1: NsUInteger,
        arg2: NsUInteger,
    ) {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            NsUInteger,
            NsUInteger,
        ) = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg1, arg2);
    }

    /// The message-send form `void objc_msgSend(id self, SEL op,
    /// NSUInteger arg1, NSUInteger arg2, NSUInteger arg3)`: three
    /// integers (the draw's drawPrimitives:vertexStart:vertexCount:).
    pub unsafe fn send_void_ul3(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg1: NsUInteger,
        arg2: NsUInteger,
        arg3: NsUInteger,
    ) {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            NsUInteger,
            NsUInteger,
            NsUInteger,
        ) = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg1, arg2, arg3);
    }

    pub unsafe fn send_ul_ret_ptr(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg: NsUInteger,
    ) -> *mut c_void {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            NsUInteger,
        ) -> *mut c_void = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg)
    }

    /// The message-send form `void objc_msgSend(id self, SEL op, id arg1,
    /// NSUInteger arg2)`: one object and one integer (the fragment
    /// stage's texture and sampler bindings'
    /// setFragmentTexture:atIndex: and setFragmentSamplerState:atIndex:).
    pub unsafe fn send_void_ptr_ul(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg1: *mut c_void,
        arg2: NsUInteger,
    ) {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            NsUInteger,
        ) = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg1, arg2);
    }

    /// The message-send form `void objc_msgSend(id self, SEL op,
    /// NSUInteger a1, NSUInteger a2, NSUInteger a3, id a4, NSUInteger
    /// a5)`: the indexed draw's drawIndexedPrimitives:indexCount:
    /// indexType:indexBuffer:indexBufferOffset: (primitive type, index
    /// count, index type, index buffer, byte offset).
    pub unsafe fn send_void_ul3_ptr_ul(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg1: NsUInteger,
        arg2: NsUInteger,
        arg3: NsUInteger,
        arg4: *mut c_void,
        arg5: NsUInteger,
    ) {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            NsUInteger,
            NsUInteger,
            NsUInteger,
            *mut c_void,
            NsUInteger,
        ) = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg1, arg2, arg3, arg4, arg5);
    }

    pub unsafe fn send_ul2_ret_ptr(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg1: NsUInteger,
        arg2: NsUInteger,
    ) -> *mut c_void {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            NsUInteger,
            NsUInteger,
        ) -> *mut c_void = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg1, arg2)
    }

    pub unsafe fn send_ul3_ret_ptr(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg1: NsUInteger,
        arg2: NsUInteger,
        arg3: NsUInteger,
    ) -> *mut c_void {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            NsUInteger,
            NsUInteger,
            NsUInteger,
        ) -> *mut c_void = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg1, arg2, arg3)
    }

    pub unsafe fn send_ul4_ret_ptr(
        &self,
        self_: *mut c_void,
        sel: *mut c_void,
        arg1: NsUInteger,
        arg2: NsUInteger,
        arg3: NsUInteger,
        arg4: NsUInteger,
    ) -> *mut c_void {
        let f: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            NsUInteger,
            NsUInteger,
            NsUInteger,
            NsUInteger,
        ) -> *mut c_void = std::mem::transmute(self.objc_msg_send.unwrap());
        f(self_, sel, arg1, arg2, arg3, arg4)
    }

    /// The alloc/init pair: allocates a new descriptor object from a
    /// class. The runtime's alloc class method and the init instance
    /// method both send one selector.
    pub unsafe fn alloc_init(&self, class: MtlClassHandle) -> *mut c_void {
        let alloc_sel = self.sel("alloc");
        let init_sel = self.sel("init");
        let alloc = self.send0(class, alloc_sel);
        self.send0(alloc, init_sel)
    }

    /// The device's name: the NSString the published property carries,
    /// read through UTF8String.
    pub unsafe fn device_name(&self, device: MtlDeviceHandle) -> String {
        let name_sel = self.sel("name");
        let utf8_sel = self.sel("UTF8String");
        let name_obj = self.send0(device, name_sel);
        if name_obj.is_null() {
            return String::new();
        }
        let utf8 = self.send_ret_ptr(name_obj, utf8_sel) as *const c_char;
        if utf8.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(utf8).to_string_lossy().into_owned()
    }
}
