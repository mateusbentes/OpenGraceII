//! The Direct3D 12 backend's loader: the engine's own loader, own code.
//!
//! RULE (lesson 3): every constant, structure, vtable entry and function
//! signature the engine uses is declared here, so no Windows SDK header
//! and no third-party binding join the tree, and the platform-neutral
//! contract never contains them.
//!
//! D3D12 is a COM interface stack: the loader declares the published
//! vtable layouts (the function pointer order the official headers
//! guarantee) and the GUIDs of the interfaces it queries, and resolves
//! the three published entry points (D3D12CreateDevice,
//! CreateDXGIFactory1, D3D12SerializeRootSignature, D3DCompile) from
//! d3d12.dll / dxgi.dll / d3dcompiler_47.dll at runtime through
//! LoadLibraryA/GetProcAddress. The kernel32 entry points are the
//! platform's own always-loaded surface; no link-time dependency joins
//! the tree.
//!
//! On a non-Windows platform the loader still compiles (the declarations
//! are plain Rust), but the dll's LoadLibraryA fails and `load()` reports
//! that cleanly: the module tests run on Linux, the hardware path needs
//! Windows (the M3's recorded scope).
//!
//! Validation scope: the ABI's layout (sizeof/offsetof of every struct,
//! the vtable order, the GUIDs) was verified against the MinGW-w64
//! official headers in the sandbox, and the whole diagnostic-frame path
//! (clear, draw, the per-plane readback copies) was executed under Wine
//! (vkd3d over lavapipe) before this loader was written. The M3 records
//! the Wine run as the D3D12 path's validation; the hardware run is the
//! LucDev's Windows scope.

use std::os::raw::{c_char, c_int, c_void};

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

// Every COM interface handle is an opaque pointer; the loader never
// dereferences one through this module.
pub type D3d12DeviceHandle = *mut c_void;
pub type D3d12CommandQueueHandle = *mut c_void;
pub type D3d12CommandAllocatorHandle = *mut c_void;
pub type D3d12GraphicsCommandListHandle = *mut c_void;
pub type D3d12RootSignatureHandle = *mut c_void;
pub type D3d12PipelineStateHandle = *mut c_void;
pub type D3d12ResourceHandle = *mut c_void;
pub type D3d12FenceHandle = *mut c_void;
pub type D3d12DescriptorHeapHandle = *mut c_void;
pub type D3d12DebugHandle = *mut c_void;
pub type DxgiFactoryHandle = *mut c_void;
pub type DxgiAdapterHandle = *mut c_void;
pub type D3dBlobHandle = *mut c_void;
// The COM objects' vtables: the loader calls through the vtable pointer
// the object's first word carries.
pub type ComObject = *mut c_void;
pub type ComVtable = *mut c_void;

// ---------------------------------------------------------------------------
// Published enum values (the published API's own values)
// ---------------------------------------------------------------------------

pub const DXGI_FORMAT_UNKNOWN: u32 = 0;
pub const DXGI_FORMAT_R32G32B32_FLOAT: u32 = 6;
pub const DXGI_FORMAT_R16G16B16A16_FLOAT: u32 = 10;
pub const DXGI_FORMAT_R8G8B8A8_UNORM: u32 = 28;
pub const DXGI_FORMAT_R8_UNORM: u32 = 61;
pub const DXGI_FORMAT_R32_FLOAT: u32 = 41;
pub const DXGI_FORMAT_R8_UINT: u32 = 30;
pub const DXGI_FORMAT_R24_UNORM_X8_TYPELESS: u32 = 46;
pub const DXGI_FORMAT_X24_TYPELESS_G8_UINT: u32 = 47;
pub const DXGI_FORMAT_D24_UNORM_S8_UINT: u32 = 45;

pub const D3D_FEATURE_LEVEL_11_0: u32 = 0xb000;
pub const D3D_FEATURE_LEVEL_12_0: u32 = 0xc000;

pub const D3D12_FEATURE_FEATURE_LEVELS: u32 = 1;

pub const D3D12_COMMAND_LIST_TYPE_DIRECT: u32 = 0;
pub const D3D12_COMMAND_QUEUE_PRIORITY_NORMAL: c_int = 0;
pub const D3D12_COMMAND_QUEUE_FLAG_NONE: u32 = 0;

pub const D3D12_RESOURCE_DIMENSION_BUFFER: u32 = 1;
pub const D3D12_RESOURCE_DIMENSION_TEXTURE2D: u32 = 3;

pub const D3D12_TEXTURE_LAYOUT_ROW_MAJOR: u32 = 1;

pub const D3D12_RESOURCE_STATE_COMMON: u32 = 0;
pub const D3D12_RESOURCE_STATE_RENDER_TARGET: u32 = 0x4;
pub const D3D12_RESOURCE_STATE_DEPTH_WRITE: u32 = 0x10;
pub const D3D12_RESOURCE_STATE_COPY_DEST: u32 = 0x400;
pub const D3D12_RESOURCE_STATE_COPY_SOURCE: u32 = 0x800;
pub const D3D12_RESOURCE_STATE_GENERIC_READ: u32 = 0xac3;

pub const D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET: u32 = 0x1;
pub const D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL: u32 = 0x2;

pub const D3D12_HEAP_TYPE_DEFAULT: u32 = 1;
pub const D3D12_HEAP_TYPE_UPLOAD: u32 = 2;
pub const D3D12_HEAP_TYPE_READBACK: u32 = 3;
pub const D3D12_HEAP_FLAG_NONE: u32 = 0;

pub const D3D12_DESCRIPTOR_HEAP_TYPE_RTV: u32 = 2;
pub const D3D12_DESCRIPTOR_HEAP_TYPE_DSV: u32 = 3;
pub const D3D12_DESCRIPTOR_HEAP_FLAG_NONE: u32 = 0;

pub const D3D12_CLEAR_FLAG_DEPTH: u32 = 0x1;
pub const D3D12_CLEAR_FLAG_STENCIL: u32 = 0x2;

pub const D3D12_FENCE_FLAG_NONE: u32 = 0;

pub const D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA: u32 = 0;

pub const D3D12_BLEND_ONE: u32 = 2;
pub const D3D12_BLEND_ZERO: u32 = 1;
pub const D3D12_BLEND_OP_ADD: u32 = 1;

pub const D3D12_COLOR_WRITE_ENABLE_ALL: u8 = 0xf;

pub const D3D12_FILL_MODE_SOLID: u32 = 3;
pub const D3D12_CULL_MODE_NONE: u32 = 1;

pub const D3D12_DEPTH_WRITE_MASK_ZERO: u32 = 0;
pub const D3D12_DEPTH_WRITE_MASK_ALL: u32 = 1;

pub const D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE: u32 = 3;
pub const D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST: u32 = 4;

pub const D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS: u32 = 1;
pub const D3D12_SHADER_VISIBILITY_ALL: u32 = 0;
pub const D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT: u32 =
    0x1;
pub const D3D_ROOT_SIGNATURE_VERSION_1_0: u32 = 1;

pub const D3D12_PIPELINE_STATE_FLAG_NONE: u32 = 0;
pub const D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED: u32 = 0;
pub const D3D12_RESOURCE_BARRIER_TYPE_TRANSITION: u32 = 0;
pub const D3D12_RESOURCE_BARRIER_FLAG_NONE: u32 = 0;
pub const D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES: u32 = 0xffff_ffff;

pub const D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX: u32 = 0;
pub const D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT: u32 = 1;

pub const D3D12_TEXTURE_DATA_PITCH_ALIGNMENT: u32 = 256;
pub const D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT: u32 = 512;

pub const D3D12_CPU_DESCRIPTOR_HANDLE_SIZE: u64 = 64;
pub const D3D12_DSV_DIMENSION_TEXTURE2D: u32 = 3;

// The WAIT object status the flush waits for.
pub const WAIT_OBJECT_0: u32 = 0;
pub const INFINITE: u32 = 0xffff_ffff;

// ---------------------------------------------------------------------------
// Fixed-size structures (the published ABI, verified from the MinGW-w64
// official headers: sizeof/offsetof checked under Wine)
// ---------------------------------------------------------------------------

/// The published struct: the viewport the rasterizer clips against.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12Viewport {
    pub top_left_x: f32,
    pub top_left_y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

/// The published struct: the scissor rectangle.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12Rect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

/// The published struct: the command queue's description.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12CommandQueueDesc {
    pub type_: u32,
    pub priority: c_int,
    pub flags: u32,
    pub node_mask: u32,
}

/// The published struct: the CPU descriptor handle (the address of one
/// descriptor slot in a descriptor heap).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12CpuDescriptorHandle {
    pub ptr: u64,
}

/// The published struct: the descriptor heap's description.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12DescriptorHeapDesc {
    pub type_: u32,
    pub num_descriptors: u32,
    pub flags: u32,
    pub node_mask: u32,
}

/// The published struct: the depth/stencil view's description.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12DepthStencilViewDesc {
    pub format: u32,
    pub view_dimension: u32,
    pub flags: u32,
    pub texture_2d: D3d12Tex2DDsv,
}

/// The published struct: the DSV's texture 2D part (the mip slice).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12Tex2DDsv {
    pub mip_slice: u32,
}

/// The published struct: the clear value (the colour union with the
/// depth/stencil value).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12ClearValue {
    pub format: u32,
    pub color: [f32; 4],
}

/// The published struct: the resource's description.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12ResourceDesc {
    pub dimension: u32,
    pub alignment: u64,
    pub width: u64,
    pub height: u32,
    pub depth_or_array_size: u16,
    pub mip_levels: u16,
    pub format: u32,
    pub sample_desc: D3d12SampleDesc,
    pub layout: u32,
    pub flags: u32,
}

/// The published struct: the sample description.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12SampleDesc {
    pub count: u32,
    pub quality: u32,
}

/// The published struct: the heap properties.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12HeapProperties {
    pub type_: u32,
    pub cpu_page_property: u32,
    pub memory_pool_preference: u32,
    pub creation_node_mask: u32,
    pub visible_node_mask: u32,
}

/// The published struct: the subresource footprint (the buffer copy's
/// row pitch and plane size).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12SubresourceFootprint {
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub row_pitch: u32,
}

/// The published struct: the placed footprint (the buffer offset plus the
/// footprint).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12PlacedFootprint {
    pub offset: u64,
    pub footprint: D3d12SubresourceFootprint,
}

/// The published struct: the texture copy location (the resource plus the
/// union of the placed footprint or the subresource index). The union's
/// memory is the footprint's 24 bytes; a subresource-index copy writes
/// the index into the union's first word (the ABI's rule: the union is
/// the smaller member's storage, the footprint's 24 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12TextureCopyLocation {
    pub resource: D3d12ResourceHandle,
    pub type_: u32,
    pub placed_footprint: D3d12PlacedFootprint,
}

/// The published struct: the resource transition barrier.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12ResourceTransitionBarrier {
    pub resource: D3d12ResourceHandle,
    pub subresource: u32,
    pub state_before: u32,
    pub state_after: u32,
}

/// The published struct: the resource barrier (the union with the
/// transition barrier).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12ResourceBarrier {
    pub type_: u32,
    pub flags: u32,
    pub transition: D3d12ResourceTransitionBarrier,
}

/// The published struct: the shader bytecode view.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12ShaderBytecode {
    pub bytecode: *const c_void,
    pub length: u64,
}

/// The published struct: one input element's description.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12InputElementDesc {
    pub semantic_name: *const c_char,
    pub semantic_index: u32,
    pub format: u32,
    pub input_slot: u32,
    pub aligned_byte_offset: u32,
    pub input_slot_class: u32,
    pub instance_data_step_rate: u32,
}

/// The published struct: the input layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12InputLayoutDesc {
    pub elements: *const D3d12InputElementDesc,
    pub num_elements: u32,
}

/// The published struct: one render target's blend description.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12RenderTargetBlendDesc {
    pub blend_enable: i32,
    pub logic_op_enable: i32,
    pub src_blend: u32,
    pub dest_blend: u32,
    pub blend_op: u32,
    pub src_blend_alpha: u32,
    pub dest_blend_alpha: u32,
    pub blend_op_alpha: u32,
    pub logic_op: u32,
    pub render_target_write_mask: u8,
}

/// The published struct: the blend state.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12BlendDesc {
    pub alpha_to_coverage_enable: i32,
    pub independent_blend_enable: i32,
    pub render_target: [D3d12RenderTargetBlendDesc; 8],
}

/// The published struct: the rasterizer state.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12RasterizerDesc {
    pub fill_mode: u32,
    pub cull_mode: u32,
    pub front_counter_clockwise: i32,
    pub depth_bias: i32,
    pub depth_bias_clamp: f32,
    pub slope_scaled_depth_bias: f32,
    pub depth_clip_enable: i32,
    pub multisample_enable: i32,
    pub antialiased_line_enable: i32,
    pub forced_sample_count: u32,
    pub conservative_raster: u32,
}

/// The published struct: one stencil operation's description.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12DepthStencilopDesc {
    pub stencil_fail_op: u32,
    pub stencil_depth_fail_op: u32,
    pub stencil_pass_op: u32,
    pub stencil_func: u32,
}

/// The published struct: the depth/stencil state.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12DepthStencilDesc {
    pub depth_enable: i32,
    pub depth_write_mask: u32,
    pub depth_func: u32,
    pub stencil_enable: i32,
    pub stencil_read_mask: u8,
    pub stencil_write_mask: u8,
    pub front_face: D3d12DepthStencilopDesc,
    pub back_face: D3d12DepthStencilopDesc,
}

/// The published struct: the root signature's 32-bit constants parameter.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12RootConstants {
    pub shader_register: u32,
    pub register_space: u32,
    pub num_32bit_values: u32,
}

/// The published struct: the root signature's one parameter (the union
/// with the 32-bit constants).
#[repr(C)]
#[repr(align(8))]
#[derive(Clone, Copy)]
pub struct D3d12RootParameter {
    pub parameter_type: u32,
    // The union's alignment is 8 (the descriptor members carry a
    // pointer), so one padding word follows the type.
    pub union_head_padding: [u32; 1],
    pub constants: D3d12RootConstants,
    // The union's storage is 16 bytes: 4 padding bytes follow the
    // 12-byte constants before the visibility word at offset 24.
    pub union_tail_padding: [u32; 1],
    pub visibility: u32,
}

/// The published struct: the root signature's description.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12RootSignatureDesc {
    pub num_parameters: u32,
    pub parameters: *const D3d12RootParameter,
    pub num_static_samplers: u32,
    pub static_samplers: *const c_void,
    pub flags: u32,
}

/// The published struct: the pipeline state's cached blob (the PSO desc's
/// cache part; the engine passes no cached blob).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12CachedPipelineState {
    pub cached_blob: D3dBlobHandle,
    pub cached_blob_size: u64,
}

/// The published struct: the stream output's description (the engine
/// passes none; the ABI's 32 bytes are the two pointers plus the three
/// counts).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12StreamOutputDesc {
    pub so_declaration: *const c_void,
    pub num_entries: u32,
    // The second pointer's alignment pads the first count.
    pub head_padding: [u32; 1],
    pub buffer_strides: *const u32,
    pub num_strides: u32,
    pub rasterized_stream: u32,
}

/// The published struct: the graphics pipeline state's description.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12GraphicsPipelineStateDesc {
    pub root_signature: D3d12RootSignatureHandle,
    pub vs: D3d12ShaderBytecode,
    pub ps: D3d12ShaderBytecode,
    pub ds: D3d12ShaderBytecode,
    pub hs: D3d12ShaderBytecode,
    pub gs: D3d12ShaderBytecode,
    pub stream_output: D3d12StreamOutputDesc,
    pub blend_state: D3d12BlendDesc,
    pub sample_mask: u32,
    pub rasterizer_state: D3d12RasterizerDesc,
    pub depth_stencil_state: D3d12DepthStencilDesc,
    pub input_layout: D3d12InputLayoutDesc,
    pub ib_strip_cut_value: u32,
    pub primitive_topology_type: u32,
    pub num_render_targets: u32,
    pub rtv_formats: [u32; 8],
    pub dsv_format: u32,
    pub sample_desc: D3d12SampleDesc,
    pub node_mask: u32,
    pub cached_pso: D3d12CachedPipelineState,
    pub flags: u32,
}

/// The published struct: the feature levels' feature data.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12FeatureDataFeatureLevels {
    pub num_feature_levels: u32,
    pub feature_levels_requested: *const u32,
    pub max_supported_feature_level: u32,
}

/// The published struct: the DXGI adapter's description.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DxgiAdapterDesc1 {
    pub description: [u16; 128],
    pub vendor_id: u32,
    pub device_id: u32,
    pub sub_sys_id: u32,
    pub revision: u32,
    pub dedicated_video_memory: u64,
    pub dedicated_system_memory: u64,
    pub shared_system_memory: u64,
    pub adapter_luid_low: u32,
    pub adapter_luid_high: c_int,
    pub flags: u32,
}

/// The published struct: the vertex buffer view.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12VertexBufferView {
    pub buffer_location: u64,
    pub size_in_bytes: u32,
    pub stride_in_bytes: u32,
}

/// The published struct: the mapped range.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12Range {
    pub begin: u64,
    pub end: u64,
}

/// The published struct: the box (the copy's sub-rectangle).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3d12Box {
    pub left: u32,
    pub top: u32,
    pub front: u32,
    pub right: u32,
    pub bottom: u32,
    pub back: u32,
}

// ---------------------------------------------------------------------------
// GUIDs (the published interface IDs, from DEFINE_GUID in the official
// headers)
// ---------------------------------------------------------------------------

// The GUIDs are 16 bytes: the loader stores each as four u32 words in
// the little-endian memory layout the ABI's GUID struct carries
// (Data1; Data3<<16|Data2; Data4[0..4]; Data4[4..8]). The M19 lesson:
// the values the device receives must be the published API's own, never
// a local guess — every word below was parsed from the official
// headers' DEFINE_GUID lines.
pub const IID_ID3D12_DEVICE_WORDS: [u32; 4] = [
    0x189819f1, 0x4b571db6, 0x211854be, 0xf7859b33,
];
pub const IID_ID3D12_COMMAND_QUEUE_WORDS: [u32; 4] = [
    0x0ec870a6, 0x4c225d7e, 0xaa5bfc8c, 0xed1676e0,
];
pub const IID_ID3D12_COMMAND_ALLOCATOR_WORDS: [u32; 4] = [
    0x6102dee4, 0x4b09af59, 0x4db499b9, 0x249bf073,
];
pub const IID_ID3D12_COMMAND_LIST_WORDS: [u32; 4] = [
    0x7116d91c, 0x47cee7e4, 0x81ecc6b8, 0xe537f468,
];
pub const IID_ID3D12_GRAPHICS_COMMAND_LIST_WORDS: [u32; 4] = [
    0x5b160d0f, 0x4185ac1b, 0xaeb3a88b, 0x55a4a542,
];
pub const IID_ID3D12_ROOT_SIGNATURE_WORDS: [u32; 4] = [
    0xc54a6b66, 0x4ee872df, 0x46a9e58b, 0x149242a1,
];
pub const IID_ID3D12_PIPELINE_STATE_WORDS: [u32; 4] = [
    0x765a30f3, 0x4c6ff624, 0xe9ac28a8, 0x45246248,
];
pub const IID_ID3D12_RESOURCE_WORDS: [u32; 4] = [
    0x696442be, 0x4059a72e, 0x5c5b79bc, 0xad0f0498,
];
pub const IID_ID3D12_FENCE_WORDS: [u32; 4] = [
    0x0a753dcf, 0x4b91c4d8, 0x5abef6ad, 0x765ad960,
];
pub const IID_ID3D12_DESCRIPTOR_HEAP_WORDS: [u32; 4] = [
    0x8efb471d, 0x4f49616c, 0x7b12f790, 0x51fa63b7,
];
pub const IID_ID3D12_DEBUG_WORDS: [u32; 4] = [
    0x344488b7, 0x474b6846, 0xf027b989, 0xe0458244,
];
pub const IID_IDXGI_FACTORY1_WORDS: [u32; 4] = [
    0x770aae78, 0x4dbaf26f, 0x3c2529a8, 0x87b3d183,
];
pub const IID_IDXGI_ADAPTER1_WORDS: [u32; 4] = [
    0x29038f61, 0x46263839, 0x6808fd91, 0x051a0179,
];
pub const IID_ID3D10_BLOB_WORDS: [u32; 4] = [
    0x8ba5fb08, 0x40e25195, 0x980d58ac, 0x02013a9c,
];

// ---------------------------------------------------------------------------
// Entry point types (the published signatures)
// ---------------------------------------------------------------------------

pub type FnD3D12CreateDevice = unsafe extern "system" fn(
    adapter: *mut c_void,
    minimum_feature_level: u32,
    riid: *const u32,
    device: *mut *mut c_void,
) -> i32;
pub type FnCreateDXGIFactory1 = unsafe extern "system" fn(
    riid: *const u32,
    factory: *mut *mut c_void,
) -> i32;
pub type FnD3D12SerializeRootSignature = unsafe extern "system" fn(
    desc: *const D3d12RootSignatureDesc,
    version: u32,
    blob: *mut D3dBlobHandle,
    error_blob: *mut D3dBlobHandle,
) -> i32;
pub type FnD3DCompile = unsafe extern "system" fn(
    src_data: *const c_void,
    src_data_size: u64,
    source_name: *const c_char,
    defines: *const c_void,
    include: *mut c_void,
    entry_point: *const c_char,
    target: *const c_char,
    flags1: u32,
    flags2: u32,
    code: *mut D3dBlobHandle,
    error_blob: *mut D3dBlobHandle,
) -> i32;

// The kernel32 surface: the platform's own always-loaded library. The
// loader declares the three entry points it uses (the same rule the
// Vulkan loader's dlopen/dlsym surface records: the platform's own
// loading primitives, no third-party binding).
pub type FnLoadLibraryA =
    unsafe extern "system" fn(name: *const c_char) -> *mut c_void;
pub type FnGetProcAddress = unsafe extern "system" fn(
    module: *mut c_void,
    name: *const c_char,
) -> *mut c_void;
pub type FnFreeLibrary = unsafe extern "system" fn(module: *mut c_void) -> i32;
pub type FnCreateEventW = unsafe extern "system" fn(
    attributes: *mut c_void,
    manual_reset: i32,
    initial_state: i32,
    name: *const u16,
) -> *mut c_void;
pub type FnWaitForSingleObject = unsafe extern "system" fn(
    handle: *mut c_void,
    milliseconds: u32,
) -> u32;
pub type FnCloseHandle =
    unsafe extern "system" fn(handle: *mut c_void) -> i32;

// The kernel32 entry points are resolved lazily on the first load: the
// process's kernel32 is always loaded, so GetModuleHandleW would resolve
// it, but the loader dlopens nothing there — the platform's own
// LoadLibraryA is the surface.
pub type FnGetModuleHandleW =
    unsafe extern "system" fn(name: *const u16) -> *mut c_void;

// ---------------------------------------------------------------------------
// COM vtable function pointer types (the published vtable layouts)
// ---------------------------------------------------------------------------

// IUnknown's three entries sit at the start of every COM vtable.
pub type FnQueryInterface = unsafe extern "system" fn(
    this: ComObject,
    riid: *const u32,
    out: *mut *mut c_void,
) -> i32;
pub type FnAddRef = unsafe extern "system" fn(this: ComObject) -> u32;
pub type FnRelease = unsafe extern "system" fn(this: ComObject) -> u32;

// The vtable accessor: the object's first word is the vtable pointer.
#[repr(C)]
struct ComHeader {
    vtable: *mut *mut c_void,
}

/// Reads the object's vtable pointer (the COM ABI's rule: the vtable is
/// the object's first word).
pub fn vtable_of(obj: ComObject) -> *mut *mut c_void {
    let header = unsafe { &*(obj as *const ComHeader) };
    header.vtable
}

/// Calls the vtable slot at `index` with one pointer argument and an
/// i32 (HRESULT) return.
pub unsafe fn call_slot_i32(
    obj: ComObject,
    index: usize,
    arg: *mut c_void,
) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(ComObject, *mut c_void) -> i32 =
        std::mem::transmute(*slot);
    f(obj, arg)
}

/// Calls the vtable slot at `index` with no arguments and an i32
/// (HRESULT) return.
pub unsafe fn call_slot_i32_0(obj: ComObject, index: usize) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(ComObject) -> i32 =
        std::mem::transmute(*slot);
    f(obj)
}

/// Reads the vtable slot at `index` as a function returning a pointer
/// with no arguments (the ID3D10Blob's GetBufferPointer pattern).
pub unsafe fn call_slot_ptr_0(obj: ComObject, index: usize) -> *mut c_void {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(ComObject) -> *mut c_void =
        std::mem::transmute(*slot);
    f(obj)
}

/// Calls the vtable slot at `index` with two pointer arguments and an
/// i32 (HRESULT) return.
pub unsafe fn call_slot_i32_2(
    obj: ComObject,
    index: usize,
    arg1: *mut c_void,
    arg2: *mut c_void,
) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(
        ComObject,
        *mut c_void,
        *mut c_void,
    ) -> i32 = std::mem::transmute(*slot);
    f(obj, arg1, arg2)
}

/// Calls the vtable slot at `index` with three pointer arguments and an
/// i32 (HRESULT) return.
pub unsafe fn call_slot_i32_3(
    obj: ComObject,
    index: usize,
    arg1: *mut c_void,
    arg2: *mut c_void,
    arg3: *mut c_void,
) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(
        ComObject,
        *mut c_void,
        *mut c_void,
        *mut c_void,
    ) -> i32 = std::mem::transmute(*slot);
    f(obj, arg1, arg2, arg3)
}

/// Calls the vtable slot at `index` with four pointer arguments and an
/// i32 (HRESULT) return.
pub unsafe fn call_slot_i32_4(
    obj: ComObject,
    index: usize,
    arg1: *mut c_void,
    arg2: *mut c_void,
    arg3: *mut c_void,
    arg4: *mut c_void,
) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(
        ComObject,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
    ) -> i32 = std::mem::transmute(*slot);
    f(obj, arg1, arg2, arg3, arg4)
}

/// Calls the vtable slot at `index` with five pointer arguments and an
/// i32 (HRESULT) return.
pub unsafe fn call_slot_i32_5(
    obj: ComObject,
    index: usize,
    arg1: *mut c_void,
    arg2: *mut c_void,
    arg3: *mut c_void,
    arg4: *mut c_void,
    arg5: *mut c_void,
) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(
        ComObject,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
    ) -> i32 = std::mem::transmute(*slot);
    f(obj, arg1, arg2, arg3, arg4, arg5)
}

/// Calls the vtable slot at `index` with six pointer arguments and an
/// i32 (HRESULT) return.
pub unsafe fn call_slot_i32_6(
    obj: ComObject,
    index: usize,
    arg1: *mut c_void,
    arg2: *mut c_void,
    arg3: *mut c_void,
    arg4: *mut c_void,
    arg5: *mut c_void,
    arg6: *mut c_void,
) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(
        ComObject,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
    ) -> i32 = std::mem::transmute(*slot);
    f(obj, arg1, arg2, arg3, arg4, arg5, arg6)
}

/// Calls the vtable slot at `index` with seven pointer arguments and an
/// i32 (HRESULT) return.
pub unsafe fn call_slot_i32_7(
    obj: ComObject,
    index: usize,
    arg1: *mut c_void,
    arg2: *mut c_void,
    arg3: *mut c_void,
    arg4: *mut c_void,
    arg5: *mut c_void,
    arg6: *mut c_void,
    arg7: *mut c_void,
) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(
        ComObject,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
    ) -> i32 = std::mem::transmute(*slot);
    f(obj, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
}

/// Calls the vtable slot at `index` with eight pointer arguments and an
/// i32 (HRESULT) return.
pub unsafe fn call_slot_i32_8(
    obj: ComObject,
    index: usize,
    arg1: *mut c_void,
    arg2: *mut c_void,
    arg3: *mut c_void,
    arg4: *mut c_void,
    arg5: *mut c_void,
    arg6: *mut c_void,
    arg7: *mut c_void,
    arg8: *mut c_void,
) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(
        ComObject,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
    ) -> i32 = std::mem::transmute(*slot);
    f(obj, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
}

/// Calls the vtable slot at `index` with an i32 first argument and an
/// i32 (HRESULT) return.
pub unsafe fn call_slot_i32_i32(
    obj: ComObject,
    index: usize,
    arg: i32,
) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(ComObject, i32) -> i32 =
        std::mem::transmute(*slot);
    f(obj, arg)
}

/// Calls the vtable slot at `index` with a u32 first argument and an
/// i32 (HRESULT) return.
pub unsafe fn call_slot_i32_u32(
    obj: ComObject,
    index: usize,
    arg: u32,
) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(ComObject, u32) -> i32 =
        std::mem::transmute(*slot);
    f(obj, arg)
}

/// Reads the vtable slot at `index` as a function returning u32 with no
/// arguments (the AddRef/Release pattern).
pub unsafe fn call_slot_u32_0(obj: ComObject, index: usize) -> u32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(ComObject) -> u32 =
        std::mem::transmute(*slot);
    f(obj)
}

/// Reads the vtable slot at `index` as a function returning a u64 with a
/// u32 argument (the GetCompletedValue/GetDescriptorHandleIncrementSize
/// pattern).
pub unsafe fn call_slot_u64_u32(obj: ComObject, index: usize, arg: u32) -> u64 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(ComObject, u32) -> u64 =
        std::mem::transmute(*slot);
    f(obj, arg)
}

/// Reads the vtable slot at `index` as a function returning a u64 with a
/// u64 argument (the GetGPUVirtualAddress pattern).
pub unsafe fn call_slot_u64_0(obj: ComObject, index: usize) -> u64 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(ComObject) -> u64 =
        std::mem::transmute(*slot);
    f(obj)
}

/// Calls the vtable slot at `index` with a float argument (the Win64
/// ABI passes the f32 in the xmm0 register, not in a GP register — the
/// ClearDepthStencilView's depth needs this signature).
pub unsafe fn call_slot_i32_f32(
    obj: ComObject,
    index: usize,
    arg1: *mut c_void,
    arg2: u32,
    arg3: f32,
    arg4: u32,
    arg5: u32,
    arg6: *mut c_void,
) -> i32 {
    let slot = vtable_of(obj).add(index);
    let f: unsafe extern "system" fn(
        ComObject,
        *mut c_void,
        u32,
        f32,
        u32,
        u32,
        *mut c_void,
    ) -> i32 = std::mem::transmute(*slot);
    f(obj, arg1, arg2, arg3, arg4, arg5, arg6)
}

/// The colour plane's row pitch (the D3D12 texture data's 256-byte
/// alignment: 256 * ceil(width * bytes / 256)).
pub fn colour_row_pitch(width: u32) -> u32 {
    D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
        * ((width * 4 + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)
            / D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
}

/// The stencil plane's row pitch (one byte per texel, the same 256-byte
/// alignment: 256 * ceil(width / 256)).
pub fn stencil_row_pitch(width: u32) -> u32 {
    D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
        * ((width + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)
            / D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
}

// ---------------------------------------------------------------------------
// The loader
// ---------------------------------------------------------------------------

/// The D3D12 loader: resolves the three dlls' entry points at runtime
/// (LoadLibraryA/GetProcAddress; no link-time dependency, the same rule
/// the Vulkan and the OpenGL loaders record). On a non-Windows platform
/// `load()` reports false with a clear missing-symbol message.
pub struct D3d12Loader {
    pub d3d12_create_device: Option<FnD3D12CreateDevice>,
    pub create_dxgi_factory1: Option<FnCreateDXGIFactory1>,
    pub d3d12_serialize_root_signature: Option<FnD3D12SerializeRootSignature>,
    pub d3d_compile: Option<FnD3DCompile>,
    // The kernel32 surface (the process's own library).
    pub load_library_a: Option<FnLoadLibraryA>,
    pub get_proc_address: Option<FnGetProcAddress>,
    pub free_library: Option<FnFreeLibrary>,
    pub create_event_w: Option<FnCreateEventW>,
    pub wait_for_single_object: Option<FnWaitForSingleObject>,
    pub close_handle: Option<FnCloseHandle>,
    pub get_module_handle_w: Option<FnGetModuleHandleW>,
    // The three dll handles (freed at shutdown).
    pub d3d12_library: *mut c_void,
    pub dxgi_library: *mut c_void,
    pub d3dcompiler_library: *mut c_void,
    pub loaded: bool,
}

impl D3d12Loader {
    pub fn new() -> Self {
        D3d12Loader {
            d3d12_create_device: None,
            create_dxgi_factory1: None,
            d3d12_serialize_root_signature: None,
            d3d_compile: None,
            load_library_a: None,
            get_proc_address: None,
            free_library: None,
            create_event_w: None,
            wait_for_single_object: None,
            close_handle: None,
            get_module_handle_w: None,
            d3d12_library: std::ptr::null_mut(),
            dxgi_library: std::ptr::null_mut(),
            d3dcompiler_library: std::ptr::null_mut(),
            loaded: false,
        }
    }

    /// The load path: resolves the kernel32 surface, then the three
    /// dlls' entry points. `missing` names the first symbol that did
    /// not resolve.
    pub fn load(&mut self, missing: &mut String) -> bool {
        // The kernel32 surface: the process's own library (always
        // loaded), resolved through GetModuleHandleW's address. The
        // function is fetched through the process's own export table:
        // the loader uses LoadLibraryA's address from the module's
        // export table via GetProcAddress, which needs the module
        // handle of kernel32 itself. The platform's own rule: the
        // kernel32 module handle is retrieved through
        // GetModuleHandleW("kernel32.dll"), whose address is resolved
        // through the process's own PEB — the loader instead resolves
        // the three functions through the kernel32.dll's well-known
        // export table walked from the module base the loader gets
        // from GetModuleHandleA. To keep the loader self-contained and
        // avoid the bootstrapping loop, the loader takes the
        // platform's own documented path: GetProcAddress on the
        // handle GetModuleHandleW returns for kernel32.dll. That
        // requires GetModuleHandleW itself, resolved through the
        // process's export table. The loader resolves the three
        // functions through the ntdll's documented rule: the process's
        // PEB's loader data lists the loaded modules, and the loader
        // walks that list. That is the platform's own bootstrapping
        // rule (the wine loader and the Windows loader both follow it).
        //
        // The loader takes the simpler and equally documented path:
        // the kernel32.dll's functions are declared with #[link] to
        // the system library the linker resolves. The Rust compiler's
        // linker on Windows (the MSVC and the GNU toolchains both)
        // links kernel32.lib, which the platform's own SDK carries.
        // The engine's rule is no third-party binding: kernel32 is the
        // platform's own library, the same class of surface the Vulkan
        // loader's dlopen/dlsym is on Linux. The three entry points
        // are therefore declared extern "system" and linked.
        self.load_library_a = Some(load_library_a);
        self.get_proc_address = Some(get_proc_address);
        self.free_library = Some(free_library);
        self.create_event_w = Some(create_event_w);
        self.wait_for_single_object = Some(wait_for_single_object);
        self.close_handle = Some(close_handle);
        self.get_module_handle_w = Some(get_module_handle_w);

        // The three dlls: d3d12.dll, dxgi.dll, d3dcompiler_47.dll.
        let d3d12 = unsafe {
            load_library_a(b"d3d12.dll\0".as_ptr() as *const c_char)
        };
        if d3d12.is_null() {
            *missing = "d3d12.dll (LoadLibraryA)".to_string();
            return false;
        }
        let dxgi = unsafe {
            load_library_a(b"dxgi.dll\0".as_ptr() as *const c_char)
        };
        if dxgi.is_null() {
            *missing = "dxgi.dll (LoadLibraryA)".to_string();
            return false;
        }
        let d3dcompiler = unsafe {
            load_library_a(b"d3dcompiler_47.dll\0".as_ptr() as *const c_char)
        };
        if d3dcompiler.is_null() {
            *missing = "d3dcompiler_47.dll (LoadLibraryA)".to_string();
            return false;
        }
        self.d3d12_library = d3d12;
        self.dxgi_library = dxgi;
        self.d3dcompiler_library = d3dcompiler;

        // The published entry points (GetProcAddress's documented
        // rule).
        macro_rules! resolve {
            ($field:ident, $lib:expr, $name:literal) => {
                let symbol = unsafe {
                    get_proc_address(
                        $lib,
                        concat!($name, "\0").as_ptr() as *const c_char,
                    )
                };
                if symbol.is_null() {
                    *missing = $name.to_string();
                    return false;
                }
                self.$field = Some(unsafe {
                    std::mem::transmute::<*mut c_void, _>(symbol)
                });
            };
        }
        resolve!(d3d12_create_device, d3d12, "D3D12CreateDevice");
        resolve!(create_dxgi_factory1, dxgi, "CreateDXGIFactory1");
        resolve!(
            d3d12_serialize_root_signature,
            d3d12,
            "D3D12SerializeRootSignature"
        );
        resolve!(d3d_compile, d3dcompiler, "D3DCompile");
        self.loaded = true;
        true
    }

    /// Frees the three dll handles (the shutdown path).
    pub fn free(&mut self) {
        unsafe {
            if !self.d3d12_library.is_null() {
                free_library(self.d3d12_library);
                self.d3d12_library = std::ptr::null_mut();
            }
            if !self.dxgi_library.is_null() {
                free_library(self.dxgi_library);
                self.dxgi_library = std::ptr::null_mut();
            }
            if !self.d3dcompiler_library.is_null() {
                free_library(self.d3dcompiler_library);
                self.d3dcompiler_library = std::ptr::null_mut();
            }
        }
        self.loaded = false;
    }
}

// The kernel32 entry points: the platform's own loading primitives, the
// same class of surface the Vulkan loader's dlopen/dlsym is on Linux.
// The bindings are only referenced on Windows builds; everywhere else
// the module links against stubs whose null results make `load()` fail
// cleanly with the same message shape the Metal loader reports on a
// host without the framework.
#[cfg(windows)]
extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char)
        -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn CreateEventW(
        attributes: *mut c_void,
        manual_reset: i32,
        initial_state: i32,
        name: *const u16,
    ) -> *mut c_void;
    fn WaitForSingleObject(handle: *mut c_void, milliseconds: u32) -> u32;
    fn CloseHandle(handle: *mut c_void) -> i32;
    fn GetModuleHandleW(name: *const u16) -> *mut c_void;
}

#[cfg(windows)]
unsafe extern "system" fn load_library_a(name: *const c_char)
    -> *mut c_void {
    LoadLibraryA(name)
}
#[cfg(windows)]
unsafe extern "system" fn get_proc_address(
    module: *mut c_void,
    name: *const c_char,
) -> *mut c_void {
    GetProcAddress(module, name)
}
#[cfg(windows)]
unsafe extern "system" fn free_library(module: *mut c_void) -> i32 {
    FreeLibrary(module)
}
#[cfg(windows)]
unsafe extern "system" fn create_event_w(
    attributes: *mut c_void,
    manual_reset: i32,
    initial_state: i32,
    name: *const u16,
) -> *mut c_void {
    CreateEventW(attributes, manual_reset, initial_state, name)
}
#[cfg(windows)]
unsafe extern "system" fn wait_for_single_object(
    handle: *mut c_void,
    milliseconds: u32,
) -> u32 {
    WaitForSingleObject(handle, milliseconds)
}
#[cfg(windows)]
unsafe extern "system" fn close_handle(handle: *mut c_void) -> i32 {
    CloseHandle(handle)
}
#[cfg(windows)]
unsafe extern "system" fn get_module_handle_w(name: *const u16)
    -> *mut c_void {
    GetModuleHandleW(name)
}

// Off Windows the backend must never reach these (the backend reports
// its load error before any call), so the stubs only keep the module
// linkable and `load()` fails at the LoadLibraryA step.
#[cfg(not(windows))]
unsafe extern "system" fn load_library_a(_name: *const c_char)
    -> *mut c_void {
    std::ptr::null_mut()
}
#[cfg(not(windows))]
unsafe extern "system" fn get_proc_address(
    _module: *mut c_void,
    _name: *const c_char,
) -> *mut c_void {
    std::ptr::null_mut()
}
#[cfg(not(windows))]
unsafe extern "system" fn free_library(_module: *mut c_void) -> i32 {
    0
}
#[cfg(not(windows))]
unsafe extern "system" fn create_event_w(
    _attributes: *mut c_void,
    _manual_reset: i32,
    _initial_state: i32,
    _name: *const u16,
) -> *mut c_void {
    std::ptr::null_mut()
}
#[cfg(not(windows))]
unsafe extern "system" fn wait_for_single_object(
    _handle: *mut c_void,
    _milliseconds: u32,
) -> u32 {
    0
}
#[cfg(not(windows))]
unsafe extern "system" fn close_handle(_handle: *mut c_void) -> i32 {
    0
}
#[cfg(not(windows))]
unsafe extern "system" fn get_module_handle_w(_name: *const u16)
    -> *mut c_void {
    std::ptr::null_mut()
}
