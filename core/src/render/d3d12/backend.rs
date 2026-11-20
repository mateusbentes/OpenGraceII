//! The Direct3D 12 render backend: the render contract's Windows
//! implementation, on the platform's Direct3D 12 runtime.
//!
//! RULE (lesson 3): the engine's own loader, own code, no Windows SDK
//! headers and no third-party binding. The `D3d12Loader` resolves the
//! three dlls' published entry points (d3d12.dll, dxgi.dll,
//! d3dcompiler_47.dll) through LoadLibraryA/GetProcAddress at runtime and
//! calls every COM interface through the vtable pointer the object's
//! first word carries.
//!
//! The backend is headless, the same rule the Vulkan, the OpenGL and the
//! Metal backends follow: the frame target is an off-screen colour
//! texture plus a depth/stencil texture, no swap chain and no window.
//! The frame flow matches the other backends: `clear()` opens the frame
//! (the colour and the depth/stencil clears are recorded into the
//! command list), the diagnostic triangle draws into the frame, and
//! `swap_buffers()` closes, executes and waits for the fence before the
//! readback maps the staging buffer with the three-plane layout the Wine
//! probes validated (colour at offset 0, the depth plane's words at
//! offset COLOUR_BYTES, the stencil plane's bytes at offset
//! COLOUR_BYTES + DS_WORDS).
//!
//! The D3D12-specific rules the Wine probes validated before this module
//! was written: a freshly created command list must be closed once
//! before the first Reset pair; the fence signal the wait sees is the
//! queue's (the CPU-side Signal completes immediately), and the fence
//! value grows each frame; the depth and the stencil planes of a D24S8
//! texture are separate subresources, so the readback copies each plane
//! with its own CopyTextureRegion call (the same one-plane-per-copy rule
//! the VUID 09103 fix records on the Vulkan side); the depth plane reads
//! as one 32-bit word per pixel with the depth in the top three bytes,
//! the stencil plane reads as one byte per texel with its own 256-byte
//! row pitch; the second frame's barriers bring the resources back from
//! COPY_SOURCE into the render states before the clears (the two-frame
//! probe validated that path).
//!
//! Validation scope: the module compiles on any platform (the loader
//! declares its own types, so no Windows SDK is needed) and the hardware
//! run needs Windows — the M3 records the Wine runs as the D3D12 path's
//! validation; nothing here claims the hardware run happened. The
//! loader's failure path is the one behaviour a non-Windows host can
//! exercise: `new()` reports the missing dll in the `RenderError`
//! message.

use super::super::{
    bytes_per_pixel, CameraParams, ClearBuffer, ClearColor, DirectionalLight, DrawSubmission,
    DynamicLight, GpuMesh, GpuMeshInner, PixelFormat, ReadbackSample, RenderBackend, RenderError,
    RenderTarget, RenderTargetDesc, RenderTargetInner, ShaderProgram, ShaderProgramInner,
    Texture2DDesc, Texture3DDesc, TextureView, TextureViewInner, VertexLayout, DEPTH_ATTACHMENT,
};
use super::d3d12_loader::{
    self as d3d, call_slot_i32_0, call_slot_i32_2, call_slot_i32_3, call_slot_i32_4,
    call_slot_i32_5, call_slot_i32_6, call_slot_i32_8, call_slot_i32_f32, call_slot_i32_u32,
    call_slot_ptr_0, call_slot_u64_0, call_slot_void_ptr_u64, vtable_of, ComObject, D3d12BlendDesc,
    D3d12CachedPipelineState, D3d12ClearValue, D3d12CommandQueueDesc, D3d12CpuDescriptorHandle,
    D3d12DepthStencilDesc, D3d12DepthStencilViewDesc, D3d12DepthStencilopDesc,
    D3d12DescriptorHeapDesc, D3d12DeviceHandle, D3d12FeatureDataFeatureLevels,
    D3d12GraphicsCommandListHandle, D3d12GraphicsPipelineStateDesc, D3d12HeapProperties,
    D3d12InputElementDesc, D3d12InputLayoutDesc, D3d12Loader, D3d12PlacedFootprint, D3d12Range,
    D3d12RasterizerDesc, D3d12Rect, D3d12RenderTargetBlendDesc, D3d12ResourceBarrier,
    D3d12ResourceDesc, D3d12ResourceTransitionBarrier, D3d12RootConstants, D3d12RootParameter,
    D3d12RootSignatureDesc, D3d12SampleDesc, D3d12ShaderBytecode, D3d12StreamOutputDesc,
    D3d12SubresourceFootprint, D3d12Tex2DDsv, D3d12TextureCopyLocation, D3d12VertexBufferView,
    D3d12Viewport, D3dBlobHandle, DxgiAdapterDesc1, DxgiAdapterHandle, DxgiFactoryHandle,
    D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, D3D12_BLEND_ZERO, D3D12_CLEAR_FLAG_DEPTH,
    D3D12_CLEAR_FLAG_STENCIL, D3D12_COLOR_WRITE_ENABLE_ALL, D3D12_COMMAND_LIST_TYPE_DIRECT,
    D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, D3D12_COMPARISON_FUNC_ALWAYS, D3D12_COMPARISON_FUNC_LESS,
    D3D12_CULL_MODE_NONE, D3D12_DEPTH_WRITE_MASK_ALL, D3D12_DEPTH_WRITE_MASK_ZERO,
    D3D12_DESCRIPTOR_HEAP_FLAG_NONE, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
    D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_FEATURE_FEATURE_LEVELS,
    D3D12_FENCE_FLAG_NONE, D3D12_FILL_MODE_SOLID, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    D3D12_RESOURCE_DIMENSION_TEXTURE3D,
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET,
    D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, D3D12_SHADER_VISIBILITY_ALL,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D_FEATURE_LEVEL_11_0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT,
    DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_UINT,
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_UNKNOWN,
    IID_ID3D12_COMMAND_ALLOCATOR_WORDS, IID_ID3D12_COMMAND_QUEUE_WORDS,
    IID_ID3D12_DESCRIPTOR_HEAP_WORDS, IID_ID3D12_DEVICE_WORDS, IID_ID3D12_FENCE_WORDS,
    IID_ID3D12_GRAPHICS_COMMAND_LIST_WORDS, IID_ID3D12_PIPELINE_STATE_WORDS,
    IID_ID3D12_RESOURCE_WORDS, IID_ID3D12_ROOT_SIGNATURE_WORDS, IID_IDXGI_FACTORY1_WORDS,
    IID_IDXGI_ADAPTER1_WORDS, IID_IDXGI_FACTORY4_WORDS,
};
use crate::math::{Mat4, Vec3};
use crate::lightvolume::LightVolume;
use std::os::raw::{c_char, c_void};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// The vtable slots the backend calls (0-based, from the MinGW-w64 official
// headers: every slot below was parsed from the vtable order, never
// guessed — the same M19 rule the Vulkan loader records)
// ---------------------------------------------------------------------------

// ID3D12Device (the slot the header's vtable order carries).
const DEVICE_CREATE_COMMAND_QUEUE: usize = 8;
const DEVICE_CREATE_COMMAND_ALLOCATOR: usize = 9;
const DEVICE_CREATE_GRAPHICS_PIPELINE_STATE: usize = 10;
const DEVICE_CREATE_COMMAND_LIST: usize = 12;
const DEVICE_CHECK_FEATURE_SUPPORT: usize = 13;
const DEVICE_CREATE_DESCRIPTOR_HEAP: usize = 14;
const DEVICE_CREATE_ROOT_SIGNATURE: usize = 16;
const DEVICE_CREATE_RENDER_TARGET_VIEW: usize = 20;
const DEVICE_CREATE_DEPTH_STENCIL_VIEW: usize = 21;
const DEVICE_CREATE_CONSTANT_BUFFER_VIEW: usize = 17;
const DEVICE_CREATE_SHADER_RESOURCE_VIEW: usize = 18;
const DEVICE_GET_DESCRIPTOR_HANDLE_INCREMENT_SIZE: usize = 15;
const DEVICE_CREATE_COMMITTED_RESOURCE: usize = 27;
const DEVICE_CREATE_FENCE: usize = 36;
// ID3D12CommandQueue.
const QUEUE_EXECUTE_COMMAND_LISTS: usize = 10;
const QUEUE_SIGNAL: usize = 14;
// ID3D12CommandAllocator.
const ALLOCATOR_RESET: usize = 8;
// ID3D12GraphicsCommandList.
const COMMAND_LIST_CLOSE: usize = 9;
const COMMAND_LIST_RESET: usize = 10;
const COMMAND_LIST_DRAW_INSTANCED: usize = 12;
const COMMAND_LIST_COPY_TEXTURE_REGION: usize = 16;
const COMMAND_LIST_IASET_PRIMITIVE_TOPOLOGY: usize = 20;
const COMMAND_LIST_RSSET_VIEWPORTS: usize = 21;
const COMMAND_LIST_RSSET_SCISSOR_RECTS: usize = 22;
const COMMAND_LIST_SET_PIPELINE_STATE: usize = 25;
const COMMAND_LIST_RESOURCE_BARRIER: usize = 26;
const COMMAND_LIST_SET_GRAPHICS_ROOT_SIGNATURE: usize = 30;
const COMMAND_LIST_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE: usize = 32;
const COMMAND_LIST_SET_GRAPHICS_ROOT_32BIT_CONSTANTS: usize = 36;
const COMMAND_LIST_SET_DESCRIPTOR_HEAPS: usize = 28;
const COMMAND_LIST_IASET_VERTEX_BUFFERS: usize = 44;
const COMMAND_LIST_IASET_INDEX_BUFFER: usize = 43;
const COMMAND_LIST_DRAW_INDEXED_INSTANCED: usize = 13;
const COMMAND_LIST_OMSET_RENDER_TARGETS: usize = 46;
const COMMAND_LIST_CLEAR_DEPTH_STENCIL_VIEW: usize = 47;
const COMMAND_LIST_CLEAR_RENDER_TARGET_VIEW: usize = 48;
// ID3D12Resource.
const RESOURCE_MAP: usize = 8;
const RESOURCE_UNMAP: usize = 9;
const RESOURCE_GET_GPU_VIRTUAL_ADDRESS: usize = 11;
// ID3D12Fence.
const FENCE_SET_EVENT_ON_COMPLETION: usize = 9;
// ID3D12DescriptorHeap.
const HEAP_GET_CPU_DESCRIPTOR_HANDLE: usize = 9;
const HEAP_GET_GPU_DESCRIPTOR_HANDLE: usize = 10;
// IDXGIFactory1 / IDXGIAdapter1 (the DXGI side: SetPrivateData comes
// before GetPrivateData, the opposite of the D3D12 side).
const FACTORY_ENUM_ADAPTERS1: usize = 12;
const ADAPTER_GET_DESC1: usize = 10;
// The IDXGIFactory4's EnumWarpAdapter (the WARP fallback's own slot: the
// IUnknown's 0-2, the IDXGIObject's 3-6, the IDXGIFactory's 7-11, the
// IDXGIFactory1's 12-13, the IDXGIFactory2's 14-24, the IDXGIFactory3's
// 25, the IDXGIFactory4's EnumAdapterByLuid 26 and EnumWarpAdapter 27).
const FACTORY4_ENUM_WARP_ADAPTER: usize = 27;
// The IUnknown's QueryInterface (the factory's own slot 0).
const UNKNOWN_QUERY_INTERFACE: usize = 0;
// ID3D10Blob (the d3dcompiler's blob).
const BLOB_GET_BUFFER_POINTER: usize = 3;
const BLOB_GET_BUFFER_SIZE: usize = 4;

// ---------------------------------------------------------------------------
// The readback staging buffer's layout (the same three-plane layout the
// Wine probes validated)
// ---------------------------------------------------------------------------

/// The colour plane's bytes (the 320x240 RGBA8 the probes copied).
const STAGING_COLOUR_BYTES: usize = 320 * 240 * 4;
/// The depth plane's words (one 32-bit word per pixel).
const STAGING_DEPTH_BYTES: usize = 320 * 240 * 4;
/// The stencil plane's row pitch (256 * ceil(width / 256) = 512 for the
/// 320-wide frame).
const STAGING_STENCIL_PITCH: usize = 256 * ((320 + 255) / 256);
/// The stencil plane's bytes (240 rows of the 256-byte pitch).
const STAGING_STENCIL_BYTES: usize = 240 * STAGING_STENCIL_PITCH;
/// The staging buffer's total size (the three planes).
const STAGING_SIZE: usize = STAGING_COLOUR_BYTES + STAGING_DEPTH_BYTES + STAGING_STENCIL_BYTES;
/// The depth plane's offset inside the staging buffer.
const STAGING_DEPTH_OFFSET: usize = STAGING_COLOUR_BYTES;
/// The stencil plane's offset inside the staging buffer.
const STAGING_STENCIL_OFFSET: usize = STAGING_COLOUR_BYTES + STAGING_DEPTH_BYTES;

/// The diagnostic triangle's uniform block (the same bytes the Vulkan,
/// the OpenGL and the Metal blocks carry: the MVP column-major followed
/// by the RGBA colour). The 20-word block the root signature's
/// constants parameter reads.
#[repr(C)]
struct TriUniform {
    mvp: [[f32; 4]; 4],
    color: [f32; 4],
}

/// The lights' uniform block (M5): the same layout every backend maps —
/// the sun's direction (xyz + pad), the sun's colour (xyz) and intensity
/// (w), the four point lights' position (xyz) and radius (w), the four
/// point lights' colour (xyz) and intensity (w), the params (x = the
/// live point-light count). The block is exactly 256 bytes (the CBV's
/// alignment rule the vkd3d validation recorded).
#[repr(C)]
#[derive(Clone, Copy)]
struct LightsUniform {
    sun_direction: [f32; 4],
    sun_color_intensity: [f32; 4],
    point_position_radius: [[f32; 4]; 4],
    point_color_intensity: [[f32; 4]; 4],
    params: [f32; 4],
    // The tail pad: the block's byte size the CBV reads (256; the
    // declared fields land at 176, the pad completes the buffer — the
    // copies' 256-byte reads stay in bounds).
    _pad: [f32; 20],
}

/// The light volume's uniform block (M8): the origin (xyz + pad), the
/// dims (xyz + pad; the depth carries the z * 6 layers) and the params
/// (x = the cell size, y = the indirect intensity, z = the toggle).
/// The same three vec4s the POST_GBUFFER pixel shader's Volume block
/// declares (the 48-byte std140 layout).
#[repr(C)]
#[derive(Clone, Copy)]
struct VolumeUniform {
    origin: [f32; 4],
    dims: [f32; 4],
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
            _pad: [0.0; 20],
        }
    }

    /// The block from the contract's lights: the sun (the direction
    /// normalised) plus up to four point lights (the rest stay zero).
    fn from_lights(sun: &DirectionalLight, lights: &[DynamicLight]) -> Self {
        let direction = sun.direction.normalised();
        let mut block = LightsUniform {
            sun_direction: [direction.x, direction.y, direction.z, 0.0],
            sun_color_intensity: [sun.color.x, sun.color.y, sun.color.z, sun.intensity],
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
                block.point_color_intensity[index] =
                    [light.color.x, light.color.y, light.color.z, light.intensity];
            }
        }
        block.params[0] = lights.len().min(4) as f32;
        block
    }
}

/// The D3D12 backend's configuration. The same fields the other backends'
/// configs carry.
#[derive(Clone, Debug)]
pub struct D3d12BackendConfig {
    pub width: u32,
    pub height: u32,
    /// Draw the diagnostic triangle each frame.
    pub draw_triangle: bool,
    pub triangle_color: ClearColor,
    /// Read back the centre pixel each frame.
    pub readback_center: bool,
    pub clear_buffers: ClearBuffer,
    pub clear_depth: f32,
    pub clear_stencil: i32,
}

impl Default for D3d12BackendConfig {
    fn default() -> Self {
        D3d12BackendConfig {
            width: 320,
            height: 240,
            draw_triangle: false,
            triangle_color: ClearColor {
                r: 200,
                g: 40,
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

/// The D3D12 backend's own texture: a device resource plus the contract's
/// shared handle state.
// The resource handle is written at creation; the release path (the next
// milestones) releases it.
#[allow(dead_code)]
struct D3d12Texture2D {
    inner: Arc<Mutex<TextureViewInner>>,
    resource: d3d::D3d12ResourceHandle,
    released: bool,
    // The resource's current state: the barrier bookkeeping the fill's
    // transitions read (the COMMON the creation path leaves it in, or
    // the render states the render-target path creates it in).
    state: u32,
}
/// The backend-owned state of one uploaded mesh: the two UPLOAD-heap
/// buffers (the PNU bytes and the u32 indices), the draw's counts, the
/// mesh's CBV_SRV_UAV descriptor heap (the CBV at slot 0, the SRV at
/// slot 1) and the 256-byte uniform buffer the CBV reads (the
/// CBV's size must be 256-byte aligned, the D3D12 rule).
#[derive(Clone)]
struct D3d12MeshState {
    vertex_buffer: d3d::D3d12ResourceHandle,
    index_buffer: d3d::D3d12ResourceHandle,
    vertex_bytes: u32,
    index_count: u32,
    srv_heap: d3d::D3d12DescriptorHeapHandle,
    uniform_buffer: d3d::D3d12ResourceHandle,
}

/// The D3D12 backend's own render target: the colour attachments plus the
/// depth/stencil attachment, each a device resource.
// The resource handles are written at creation; the release path (the
// next milestones) releases them.
#[allow(dead_code)]
struct D3d12RenderTarget {
    inner: Arc<Mutex<RenderTargetInner>>,
    views: Vec<D3d12Texture2D>,
    depth_view: Option<D3d12Texture2D>,
    /// The deferred path's own RTV/DSV heaps (the three colour
    /// attachments' render-target views, the depth's DSV; the creation
    /// happens once at the first deferred pass).
    rtv_heap: d3d::D3d12DescriptorHeapHandle,
    dsv_heap: d3d::D3d12DescriptorHeapHandle,
    /// Whether the colour attachments sit in the
    /// PIXEL_SHADER_RESOURCE state (the last lighting pass's hand-off;
    /// the next geometry pass's barriers read it).
    views_state_read: bool,
    released: bool,
}

/// The headless D3D12 backend.
#[allow(dead_code)]
pub struct D3d12Backend {
    config: D3d12BackendConfig,
    loader: D3d12Loader,
    device: D3d12DeviceHandle,
    queue: d3d::D3d12CommandQueueHandle,
    allocator: d3d::D3d12CommandAllocatorHandle,
    command_list: D3d12GraphicsCommandListHandle,
    // The device's identity (the describe() reports it).
    adapter_desc: String,
    adapter_vendor: u32,
    feature_level: u32,
    // The off-screen frame target's colour and depth/stencil resources.
    color_texture: d3d::D3d12ResourceHandle,
    depth_texture: d3d::D3d12ResourceHandle,
    // The RTV and DSV descriptor heaps (one descriptor each) and the
    // handles the clears and the OMSetRenderTargets read.
    rtv_heap: d3d::D3d12DescriptorHeapHandle,
    dsv_heap: d3d::D3d12DescriptorHeapHandle,
    rtv_handle: D3d12CpuDescriptorHandle,
    dsv_handle: D3d12CpuDescriptorHandle,
    // The diagnostic triangle's pipeline, root signature and vertex buffer.
    tri_pipeline: d3d::D3d12PipelineStateHandle,
    tri_root_signature: d3d::D3d12RootSignatureHandle,
    tri_vertex_buffer: d3d::D3d12ResourceHandle,
    tri_ready: bool,
    // The readback staging buffer (the three-plane layout), its fence and
    // the fence's value (the value grows each frame).
    staging_buffer: d3d::D3d12ResourceHandle,
    fence: d3d::D3d12FenceHandle,
    fence_event: *mut c_void,
    fence_value: u64,
    // Whether the readback's copies left the colour and depth/stencil
    // resources in COPY_SOURCE: the return barriers only record after a
    // readback (the probe C's frame-2 rule).
    resources_in_copy_source: bool,
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
    targets: Vec<D3d12RenderTarget>,
    textures_2d: Vec<D3d12Texture2D>,
    textures_3d: Vec<D3d12Texture2D>,
    next_mesh_id: usize,
    meshes: Vec<Option<D3d12MeshState>>,
    mesh_ready: bool,
    mesh_root_signature: d3d::D3d12RootSignatureHandle,
    mesh_pipeline: d3d::D3d12PipelineStateHandle,
    // The lit mesh path's state (M5): the lit PSO and its root signature
    // (the CBV at b0, the SRV at t0, the lights' CBV at b1) plus the
    // shared 256-byte lights' buffer the draws write.
    lit_ready: bool,
    lit_root_signature: d3d::D3d12RootSignatureHandle,
    lit_pipeline: d3d::D3d12PipelineStateHandle,
    lit_lights_buffer: d3d::D3d12ResourceHandle,
    lit_block: LightsUniform,
    // The deferred lighting path's state (M6): the G-buffer's geometry
    // PSO (the three-attachment pixel shader) and the fullscreen
    // lighting PSO, each with its own root signature, plus the shared
    // 256-byte lights' buffer the lighting draw's CBV at b1 reads.
    gbuffer_ready: bool,
    gbuffer_root_signature: d3d::D3d12RootSignatureHandle,
    gbuffer_pipeline: d3d::D3d12PipelineStateHandle,
    lighting_ready: bool,
    lighting_root_signature: d3d::D3d12RootSignatureHandle,
    lighting_pipeline: d3d::D3d12PipelineStateHandle,
    lighting_lights_buffer: d3d::D3d12ResourceHandle,
    /// The lighting pass's own shader-visible heap (four descriptors:
    /// the three G-buffer SRVs and the lights' CBV; the mesh state's
    /// heap carries three descriptors and the lighting pass's fourth
    /// would overrun it).
    lighting_srv_heap: d3d::D3d12DescriptorHeapHandle,
    // The post-processing chain's state (M7): the HDR geometry PSO and
    // the fullscreen composite PSO, each with its own root signature,
    // plus the composite's own config CBV buffer and heap.
    post_gbuffer_ready: bool,
    post_gbuffer_root_signature: d3d::D3d12RootSignatureHandle,
    post_gbuffer_pipeline: d3d::D3d12PipelineStateHandle,
    post_composite_ready: bool,
    post_composite_root_signature: d3d::D3d12RootSignatureHandle,
    post_composite_pipeline: d3d::D3d12PipelineStateHandle,
    post_lights_buffer: d3d::D3d12ResourceHandle,
    post_config_buffer: d3d::D3d12ResourceHandle,
    post_volume_buffer: d3d::D3d12ResourceHandle,
    post_srv_heap: d3d::D3d12DescriptorHeapHandle,
    /// The M8 light volume's state: the 3D texture the set_light_volume
    /// call owns and the uniform block's values the gbuffer pass's
    /// Volume block reads (the origin, the dims z * 6 and the cell size).
    light_volume: Option<TextureView>,
    volume_uniform: [f32; 12],
    bound_texture: Option<usize>,
    default_texture: Option<usize>,
    readback_queue: Vec<(u32, u32)>,
    last_readback: Vec<ReadbackSample>,
    has_center: bool,
    center: ClearColor,
    center_depth: f32,
    center_stencil: i32,
}

// ---------------------------------------------------------------------------
// The helper functions the setup path calls (the probe C's sequence,
// step by step)
// ---------------------------------------------------------------------------

/// A vtable call's HRESULT: S_OK's 0.
const S_OK: i32 = 0;
/// The stencil op's KEEP (the D3D12 stencil op table's 1; the
/// d3d12_loader's table has no named constant for it).
const D3D12_STENCIL_OP_KEEP: u32 = 1;

/// Calls the published CreateDXGIFactory1 entry point: the factory the
/// adapter enumeration walks (the probe C's first step).
unsafe fn self_create_factory(loader: &D3d12Loader) -> Result<DxgiFactoryHandle, RenderError> {
    let mut factory: *mut c_void = std::ptr::null_mut();
    let hr = loader
        .create_dxgi_factory1
        .expect("the loader resolved CreateDXGIFactory1")(
        IID_IDXGI_FACTORY1_WORDS.as_ptr() as *const u32,
        &mut factory,
    );
    if hr != S_OK || factory.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the DXGI factory could not be created \
             (hr=0x{:08x})",
            hr
        )));
    }
    Ok(factory)
}

/// Reads the first adapter (the factory's EnumAdapters1 slot 12) and its
/// description: the describe() report's vendor and renderer strings.
unsafe fn self_read_adapter(
    factory: DxgiFactoryHandle,
) -> Result<(DxgiAdapterHandle, String, u32), RenderError> {
    let mut adapter: *mut c_void = std::ptr::null_mut();
    // EnumAdapters1's signature carries no IID: (Adapter: UINT,
    // ppAdapter: *mut *mut c_void) — the Wine probe's crash recorded
    // the write into the read-only GUID words the extra argument
    // caused.
    let hr = call_slot_i32_2(
        factory as ComObject,
        FACTORY_ENUM_ADAPTERS1,
        0 as *mut c_void,
        &mut adapter as *mut _ as *mut c_void,
    );
    if hr != S_OK || adapter.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the first adapter could not be enumerated \
             (hr=0x{:08x})",
            hr
        )));
    }
    let mut desc = DxgiAdapterDesc1 {
        description: [0; 128],
        vendor_id: 0,
        device_id: 0,
        sub_sys_id: 0,
        revision: 0,
        dedicated_video_memory: 0,
        dedicated_system_memory: 0,
        shared_system_memory: 0,
        adapter_luid_low: 0,
        adapter_luid_high: 0,
        flags: 0,
    };
    let f: unsafe extern "system" fn(ComObject, *mut DxgiAdapterDesc1) -> i32 =
        std::mem::transmute(*vtable_of(adapter as ComObject).add(ADAPTER_GET_DESC1));
    let hr = f(adapter as ComObject, &mut desc);
    if hr != S_OK {
        return Err(RenderError::from(format!(
            "D3d12Backend: the adapter's description could not be read \
             (hr=0x{:08x})",
            hr
        )));
    }
    // The WCHAR[128] description: the UTF-16 units up to the zero.
    let units: Vec<u16> = desc
        .description
        .iter()
        .take_while(|u| **u != 0)
        .copied()
        .collect();
    let name = String::from_utf16_lossy(&units);
    Ok((adapter, name, desc.vendor_id))
}

/// Queries the factory's IDXGIFactory4 interface (the QueryInterface
/// slot 0): the WARP fallback's own entry. The Windows 10's factory
/// carries the interface; the Wine's factory may not (the query
/// returns a null pointer then).
unsafe fn self_query_factory4(
    factory: DxgiFactoryHandle,
) -> Option<DxgiFactoryHandle> {
    let mut factory4: *mut c_void = std::ptr::null_mut();
    let hr = call_slot_i32_2(
        factory as ComObject,
        UNKNOWN_QUERY_INTERFACE,
        IID_IDXGI_FACTORY4_WORDS.as_ptr() as *const c_void as *mut c_void,
        &mut factory4 as *mut *mut c_void as *mut c_void,
    );
    if hr != S_OK || factory4.is_null() {
        return None;
    }
    Some(factory4)
}

/// Reads the WARP adapter through the factory4's EnumWarpAdapter and
/// its description: the describe() report's vendor and renderer
/// strings when the hardware path fails (the same DXGK description
/// string the real adapters carry — the WARP's own is the "Microsoft
/// Basic Render Driver").
unsafe fn self_read_warp_adapter(
    factory4: DxgiFactoryHandle,
) -> Result<(DxgiAdapterHandle, String, u32), RenderError> {
    let mut adapter: *mut c_void = std::ptr::null_mut();
    let hr = call_slot_i32_2(
        factory4 as ComObject,
        FACTORY4_ENUM_WARP_ADAPTER,
        IID_IDXGI_ADAPTER1_WORDS.as_ptr() as *const c_void as *mut c_void,
        &mut adapter as *mut *mut c_void as *mut c_void,
    );
    if hr != S_OK || adapter.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the WARP adapter could not be enumerated \
             (hr=0x{:08x})",
            hr
        )));
    }
    let mut desc = DxgiAdapterDesc1 {
        description: [0; 128],
        vendor_id: 0,
        device_id: 0,
        sub_sys_id: 0,
        revision: 0,
        dedicated_video_memory: 0,
        dedicated_system_memory: 0,
        shared_system_memory: 0,
        adapter_luid_low: 0,
        adapter_luid_high: 0,
        flags: 0,
    };
    let f: unsafe extern "system" fn(ComObject, *mut DxgiAdapterDesc1) -> i32 =
        std::mem::transmute(*vtable_of(adapter as ComObject).add(ADAPTER_GET_DESC1));
    let hr = f(adapter as ComObject, &mut desc);
    if hr != S_OK {
        return Err(RenderError::from(format!(
            "D3d12Backend: the WARP adapter's description could not be \
             read (hr=0x{:08x})",
            hr
        )));
    }
    let units: Vec<u16> = desc
        .description
        .iter()
        .take_while(|u| **u != 0)
        .copied()
        .collect();
    let name = String::from_utf16_lossy(&units);
    Ok((adapter, name, desc.vendor_id))
}

/// Calls the published D3D12CreateDevice entry point: the device at the
/// probes' minimum feature level 11.0.
unsafe fn self_create_device(
    loader: &D3d12Loader,
    adapter: DxgiAdapterHandle,
) -> Result<D3d12DeviceHandle, RenderError> {
    let mut device: *mut c_void = std::ptr::null_mut();
    let hr = loader
        .d3d12_create_device
        .expect("the loader resolved D3D12CreateDevice")(
        adapter,
        D3D_FEATURE_LEVEL_11_0,
        IID_ID3D12_DEVICE_WORDS.as_ptr() as *const u32,
        &mut device,
    );
    if hr != S_OK || device.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the D3D12 device could not be created \
             (hr=0x{:08x})",
            hr
        )));
    }
    Ok(device)
}

/// Reads the device's highest supported feature level (the
/// CheckFeatureSupport slot 13, the FEATURE levels query): the
/// describe() report's feature level.
unsafe fn self_check_feature_level(device: D3d12DeviceHandle) -> u32 {
    let levels = [D3D_FEATURE_LEVEL_11_0];
    let mut data = D3d12FeatureDataFeatureLevels {
        num_feature_levels: 1,
        feature_levels_requested: levels.as_ptr(),
        max_supported_feature_level: 0,
    };
    let hr = call_slot_i32_4(
        device as ComObject,
        DEVICE_CHECK_FEATURE_SUPPORT,
        D3D12_FEATURE_FEATURE_LEVELS as *mut c_void,
        &mut data as *mut D3d12FeatureDataFeatureLevels as *mut c_void,
        std::mem::size_of::<D3d12FeatureDataFeatureLevels>() as *mut c_void,
        std::ptr::null_mut(),
    );
    if hr != S_OK {
        return D3D_FEATURE_LEVEL_11_0;
    }
    data.max_supported_feature_level
}

/// Creates the RTV and the DSV descriptors (the device's slots 20 and
/// 21): the colour view's D3D12RenderTargetViewDesc is null (the format
/// comes from the resource), the depth view's D3D12DepthStencilViewDesc
/// carries the D24S8 format and the TEXTURE2D dimension.
unsafe fn self_create_rtv_dsv(
    device: D3d12DeviceHandle,
    color_texture: d3d::D3d12ResourceHandle,
    depth_texture: d3d::D3d12ResourceHandle,
    rtv_handle: D3d12CpuDescriptorHandle,
    dsv_handle: D3d12CpuDescriptorHandle,
) {
    let dsv_desc = D3d12DepthStencilViewDesc {
        format: DXGI_FORMAT_D24_UNORM_S8_UINT,
        view_dimension: D3D12_DSV_DIMENSION_TEXTURE2D,
        flags: 0,
        texture_2d: D3d12Tex2DDsv { mip_slice: 0 },
    };
    call_slot_i32_4(
        device as ComObject,
        DEVICE_CREATE_RENDER_TARGET_VIEW,
        color_texture,
        std::ptr::null_mut(),
        rtv_handle.ptr as *mut c_void,
        std::ptr::null_mut(),
    );
    call_slot_i32_3(
        device as ComObject,
        DEVICE_CREATE_DEPTH_STENCIL_VIEW,
        depth_texture,
        &dsv_desc as *const D3d12DepthStencilViewDesc as *mut c_void,
        dsv_handle.ptr as *mut c_void,
    );
}

/// Calls the device's CreateCommandQueue (slot 8): the DIRECT
/// queue the frame's commands execute on.
unsafe fn self_create_queue(
    device: D3d12DeviceHandle,
) -> Result<d3d::D3d12CommandQueueHandle, RenderError> {
    let desc = D3d12CommandQueueDesc {
        type_: D3D12_COMMAND_LIST_TYPE_DIRECT,
        priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
        flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
        node_mask: 0,
    };
    let mut queue: *mut c_void = std::ptr::null_mut();
    let hr = call_slot_i32_4(
        device as ComObject,
        DEVICE_CREATE_COMMAND_QUEUE,
        &desc as *const D3d12CommandQueueDesc as *mut c_void,
        IID_ID3D12_COMMAND_QUEUE_WORDS.as_ptr() as *mut c_void,
        &mut queue as *mut _ as *mut c_void,
        std::ptr::null_mut(),
    );
    if hr != S_OK || queue.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the command queue could not be created \
             (hr=0x{:08x})",
            hr
        )));
    }
    Ok(queue)
}

/// Calls the device's CreateCommandAllocator (slot 9): the DIRECT
/// allocator the command list resets.
unsafe fn self_create_allocator(
    device: D3d12DeviceHandle,
) -> Result<d3d::D3d12CommandAllocatorHandle, RenderError> {
    let mut allocator: *mut c_void = std::ptr::null_mut();
    let hr = call_slot_i32_4(
        device as ComObject,
        DEVICE_CREATE_COMMAND_ALLOCATOR,
        D3D12_COMMAND_LIST_TYPE_DIRECT as *mut c_void,
        IID_ID3D12_COMMAND_ALLOCATOR_WORDS.as_ptr() as *mut c_void,
        &mut allocator as *mut _ as *mut c_void,
        std::ptr::null_mut(),
    );
    if hr != S_OK || allocator.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the command allocator could not be created \
             (hr=0x{:08x})",
            hr
        )));
    }
    Ok(allocator)
}

/// Calls the device's CreateCommandList (slot 12) and performs the one
/// initial Close() the Wine probe validated: a freshly created command
/// list is in the recording state, so the first Reset pair is only legal
/// after one Close.
unsafe fn self_create_command_list(
    device: D3d12DeviceHandle,
    allocator: d3d::D3d12CommandAllocatorHandle,
) -> Result<D3d12GraphicsCommandListHandle, RenderError> {
    let mut list: *mut c_void = std::ptr::null_mut();
    let hr = call_slot_i32_6(
        device as ComObject,
        DEVICE_CREATE_COMMAND_LIST,
        std::ptr::null_mut(),
        D3D12_COMMAND_LIST_TYPE_DIRECT as *mut c_void,
        allocator,
        std::ptr::null_mut(),
        IID_ID3D12_GRAPHICS_COMMAND_LIST_WORDS.as_ptr() as *mut c_void,
        &mut list as *mut _ as *mut c_void,
    );
    if hr != S_OK || list.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the command list could not be created \
             (hr=0x{:08x})",
            hr
        )));
    }
    let hr = call_slot_i32_0(list as ComObject, COMMAND_LIST_CLOSE);
    if hr != S_OK {
        return Err(RenderError::from(format!(
            "D3d12Backend: the command list's initial close failed \
             (hr=0x{:08x})",
            hr
        )));
    }
    Ok(list)
}

/// Calls the device's CreateCommittedResource (slot 27) for one off-screen
/// texture: the DEFAULT heap, the committed resource's initial state and
/// the clear value the clear loads read.
unsafe fn self_create_texture(
    device: D3d12DeviceHandle,
    format: u32,
    width: u32,
    height: u32,
    flags: u32,
    initial_state: u32,
    clear_value: Option<&D3d12ClearValue>,
) -> Result<d3d::D3d12ResourceHandle, RenderError> {
    let heap = D3d12HeapProperties {
        type_: D3D12_HEAP_TYPE_DEFAULT,
        cpu_page_property: 0,
        memory_pool_preference: 0,
        creation_node_mask: 0,
        visible_node_mask: 0,
    };
    let desc = D3d12ResourceDesc {
        dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        alignment: 0,
        width: width as u64,
        height,
        depth_or_array_size: 1,
        mip_levels: 1,
        format,
        sample_desc: D3d12SampleDesc {
            count: 1,
            quality: 0,
        },
        layout: 0,
        flags,
    };
    let mut resource: *mut c_void = std::ptr::null_mut();
    let clear_ptr = match clear_value {
        Some(value) => value as *const D3d12ClearValue as *mut c_void,
        None => std::ptr::null_mut(),
    };
    let hr = call_slot_i32_8(
        device as ComObject,
        DEVICE_CREATE_COMMITTED_RESOURCE,
        &heap as *const D3d12HeapProperties as *mut c_void,
        D3D12_HEAP_FLAG_NONE as *mut c_void,
        &desc as *const D3d12ResourceDesc as *mut c_void,
        initial_state as *mut c_void,
        clear_ptr,
        IID_ID3D12_RESOURCE_WORDS.as_ptr() as *mut c_void,
        &mut resource as *mut _ as *mut c_void,
        std::ptr::null_mut(),
    );
    if hr != S_OK || resource.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the texture resource could not be created \
             (hr=0x{:08x})",
            hr
        )));
    }
    Ok(resource)
}

unsafe fn self_create_texture_3d(
    device: D3d12DeviceHandle,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    flags: u32,
    initial_state: u32,
) -> Result<d3d::D3d12ResourceHandle, RenderError> {
    let heap = D3d12HeapProperties {
        type_: D3D12_HEAP_TYPE_DEFAULT,
        cpu_page_property: 0,
        memory_pool_preference: 0,
        creation_node_mask: 0,
        visible_node_mask: 0,
    };
    let desc = D3d12ResourceDesc {
        dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        alignment: 0,
        width: width as u64,
        height,
        depth_or_array_size: depth as u16,
        mip_levels: 1,
        format,
        sample_desc: D3d12SampleDesc {
            count: 1,
            quality: 0,
        },
        layout: 0,
        flags,
    };
    let mut resource: *mut c_void = std::ptr::null_mut();
    let hr = call_slot_i32_8(
        device as ComObject,
        DEVICE_CREATE_COMMITTED_RESOURCE,
        &heap as *const D3d12HeapProperties as *mut c_void,
        D3D12_HEAP_FLAG_NONE as *mut c_void,
        &desc as *const D3d12ResourceDesc as *mut c_void,
        initial_state as *mut c_void,
        std::ptr::null_mut(),
        IID_ID3D12_RESOURCE_WORDS.as_ptr() as *mut c_void,
        &mut resource as *mut _ as *mut c_void,
        std::ptr::null_mut(),
    );
    if hr != S_OK || resource.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the 3D texture resource could not be created \
             (hr=0x{:08x})",
            hr
        )));
    }
    Ok(resource)
}

/// Calls the device's CreateCommittedResource (slot 27) for one buffer:
/// the READBACK heap's staging buffer or the UPLOAD heap's vertex buffer.
unsafe fn self_create_buffer(
    device: D3d12DeviceHandle,
    bytes: u64,
    heap_type: u32,
    initial_state: u32,
) -> Result<d3d::D3d12ResourceHandle, RenderError> {
    let heap = D3d12HeapProperties {
        type_: heap_type,
        cpu_page_property: 0,
        memory_pool_preference: 0,
        creation_node_mask: 0,
        visible_node_mask: 0,
    };
    let desc = D3d12ResourceDesc {
        dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        alignment: 0,
        width: bytes,
        height: 1,
        depth_or_array_size: 1,
        mip_levels: 1,
        format: DXGI_FORMAT_UNKNOWN,
        sample_desc: D3d12SampleDesc {
            count: 1,
            quality: 0,
        },
        layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        flags: 0,
    };
    let mut resource: *mut c_void = std::ptr::null_mut();
    let hr = call_slot_i32_8(
        device as ComObject,
        DEVICE_CREATE_COMMITTED_RESOURCE,
        &heap as *const D3d12HeapProperties as *mut c_void,
        D3D12_HEAP_FLAG_NONE as *mut c_void,
        &desc as *const D3d12ResourceDesc as *mut c_void,
        initial_state as *mut c_void,
        std::ptr::null_mut(),
        IID_ID3D12_RESOURCE_WORDS.as_ptr() as *mut c_void,
        &mut resource as *mut _ as *mut c_void,
        std::ptr::null_mut(),
    );
    if hr != S_OK || resource.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the buffer resource could not be created \
             (hr=0x{:08x})",
            hr
        )));
    }
    Ok(resource)
}

/// Calls the device's CreateDescriptorHeap (slot 14) for one RTV or DSV
/// descriptor.
unsafe fn self_create_descriptor_heap(
    device: D3d12DeviceHandle,
    heap_type: u32,
    num_descriptors: u32,
    flags: u32,
) -> Result<d3d::D3d12DescriptorHeapHandle, RenderError> {
    let desc = D3d12DescriptorHeapDesc {
        type_: heap_type,
        num_descriptors,
        flags,
        node_mask: 0,
    };
    let mut heap: *mut c_void = std::ptr::null_mut();
    let hr = call_slot_i32_4(
        device as ComObject,
        DEVICE_CREATE_DESCRIPTOR_HEAP,
        &desc as *const D3d12DescriptorHeapDesc as *mut c_void,
        IID_ID3D12_DESCRIPTOR_HEAP_WORDS.as_ptr() as *mut c_void,
        &mut heap as *mut _ as *mut c_void,
        std::ptr::null_mut(),
    );
    if hr != S_OK || heap.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the descriptor heap could not be created \
             (hr=0x{:08x})",
            hr
        )));
    }
    Ok(heap)
}

/// Reads the heap's CPU descriptor handle (slot 9): the hidden
/// aggregate-return pointer the C vtable's ABI carries.
unsafe fn self_heap_cpu_handle(heap: d3d::D3d12DescriptorHeapHandle) -> D3d12CpuDescriptorHandle {
    let mut handle = D3d12CpuDescriptorHandle { ptr: 0 };
    let f: unsafe extern "system" fn(ComObject, *mut D3d12CpuDescriptorHandle) =
        std::mem::transmute(*vtable_of(heap as ComObject).add(HEAP_GET_CPU_DESCRIPTOR_HANDLE));
    f(heap as ComObject, &mut handle);
    handle
}

/// Reads the descriptor heap's increment size (slot 15): the device's
/// one word per descriptor the handle offsets carry.
unsafe fn self_descriptor_increment_size(device: D3d12DeviceHandle, heap_type: u32) -> u32 {
    call_slot_i32_u32(
        device as ComObject,
        DEVICE_GET_DESCRIPTOR_HANDLE_INCREMENT_SIZE,
        heap_type,
    ) as u32
}

/// Reads the heap's CPU descriptor handle at one index: the increment
/// size times the index added to the heap's start handle.
unsafe fn self_heap_cpu_handle_offset(
    heap: d3d::D3d12DescriptorHeapHandle,
    device: D3d12DeviceHandle,
    heap_type: u32,
    index: usize,
) -> D3d12CpuDescriptorHandle {
    let mut handle = self_heap_cpu_handle(heap);
    let increment = self_descriptor_increment_size(device, heap_type);
    handle.ptr += (increment as u64) * index as u64;
    handle
}

/// Reads the heap's GPU descriptor handle (slot 10): the hidden
/// aggregate-return pointer the C vtable's ABI carries.
unsafe fn self_heap_gpu_handle(heap: d3d::D3d12DescriptorHeapHandle) -> D3d12CpuDescriptorHandle {
    let mut handle = D3d12CpuDescriptorHandle { ptr: 0 };
    let f: unsafe extern "system" fn(ComObject, *mut D3d12CpuDescriptorHandle) =
        std::mem::transmute(*vtable_of(heap as ComObject).add(HEAP_GET_GPU_DESCRIPTOR_HANDLE));
    f(heap as ComObject, &mut handle);
    handle
}

/// Reads the heap's GPU descriptor handle at one index: the same
/// offset rule the CPU handle follows.
unsafe fn self_heap_gpu_handle_offset(
    heap: d3d::D3d12DescriptorHeapHandle,
    device: D3d12DeviceHandle,
    heap_type: u32,
    index: usize,
) -> D3d12CpuDescriptorHandle {
    let mut handle = self_heap_gpu_handle(heap);
    let increment = self_descriptor_increment_size(device, heap_type);
    handle.ptr += (increment as u64) * index as u64;
    handle
}

/// Calls the device's CreateFence (slot 36): the flush fence the queue
/// signals and the wait waits on.
unsafe fn self_create_fence(
    device: D3d12DeviceHandle,
) -> Result<d3d::D3d12FenceHandle, RenderError> {
    let mut fence: *mut c_void = std::ptr::null_mut();
    let hr = call_slot_i32_4(
        device as ComObject,
        DEVICE_CREATE_FENCE,
        0 as *mut c_void,
        D3D12_FENCE_FLAG_NONE as *mut c_void,
        IID_ID3D12_FENCE_WORDS.as_ptr() as *mut c_void,
        &mut fence as *mut _ as *mut c_void,
    );
    if hr != S_OK || fence.is_null() {
        return Err(RenderError::from(format!(
            "D3d12Backend: the fence could not be created (hr=0x{:08x})",
            hr
        )));
    }
    Ok(fence)
}

/// Creates the fence's event (kernel32's CreateEventW): the manual-reset
/// event the flush waits on.
unsafe fn self_create_fence_event(loader: &D3d12Loader) -> Result<*mut c_void, RenderError> {
    let event = loader
        .create_event_w
        .expect("the loader resolved CreateEventW")(
        std::ptr::null_mut(), 0, 0, std::ptr::null()
    );
    if event.is_null() {
        return Err(RenderError::from(
            "D3d12Backend: the fence event could not be created",
        ));
    }
    Ok(event)
}

// ---------------------------------------------------------------------------
// The setup path
// ---------------------------------------------------------------------------

impl D3d12Backend {
    /// Creates a backend instance. Throws `RenderError` when the platform
    /// has no D3D12 runtime, no device can be created, or the frame
    /// resources cannot be allocated.
    pub fn new(config: D3d12BackendConfig) -> Result<Self, RenderError> {
        let clear_depth = config.clear_depth;
        let clear_stencil = config.clear_stencil;
        if config.width == 0 || config.height == 0 {
            return Err(RenderError::from(
                "D3d12Backend: width and height must be positive",
            ));
        }
        let mut loader = D3d12Loader::new();
        let mut missing = String::new();
        if !loader.load(&mut missing) {
            return Err(RenderError::from(format!(
                "D3d12Backend: the platform's Direct3D 12 runtime could \
                 not be loaded ({})",
                missing
            )));
        }
        // The adapter and the device: the factory enumerates the first
        // adapter, the device is created at feature level 11.0 (the
        // probes' minimum), and the adapter's description is the
        // describe() report. The machine without a D3D12-capable GPU
        // falls back to the WARP adapter: the Microsoft Basic Render
        // Driver's own software rasterizer, the D3D12 runtime's
        // guaranteed path.
        let factory = unsafe { self_create_factory(&loader)? };
        let (_adapter, adapter_desc, adapter_vendor, device) = {
            let (adapter, adapter_desc, adapter_vendor) =
                unsafe { self_read_adapter(factory)? };
            if let Ok(device) = unsafe { self_create_device(&loader, adapter) } {
                (adapter, adapter_desc, adapter_vendor, device)
            } else {
                // The WARP fallback: the factory4's own EnumWarpAdapter
                // (the query may return None where the factory does not
                // carry the interface, the same rule the loader's
                // optional entry points carry).
                let factory4 = match unsafe { self_query_factory4(factory) } {
                    Some(f) => f,
                    None => {
                        return Err(RenderError::from(
                            "D3d12Backend: the first adapter does not \
                             carry the D3D12 device and the factory does \
                             not carry the IDXGIFactory4 (the WARP \
                             fallback)",
                        ));
                    }
                };
                let (warp_adapter, warp_desc, warp_vendor) =
                    unsafe { self_read_warp_adapter(factory4)? };
                let device = unsafe { self_create_device(&loader, warp_adapter)? };
                (warp_adapter, warp_desc, warp_vendor, device)
            }
        };
        let feature_level = unsafe { self_check_feature_level(device) };
        let queue = unsafe { self_create_queue(device)? };
        let allocator = unsafe { self_create_allocator(device)? };
        let command_list = unsafe { self_create_command_list(device, allocator)? };
        let fence = unsafe { self_create_fence(device)? };
        let fence_event = unsafe { self_create_fence_event(&loader)? };
        let mut backend = D3d12Backend {
            config,
            loader,
            device,
            queue,
            allocator,
            command_list,
            adapter_desc,
            adapter_vendor,
            feature_level,
            color_texture: std::ptr::null_mut(),
            depth_texture: std::ptr::null_mut(),
            rtv_heap: std::ptr::null_mut(),
            dsv_heap: std::ptr::null_mut(),
            rtv_handle: D3d12CpuDescriptorHandle { ptr: 0 },
            dsv_handle: D3d12CpuDescriptorHandle { ptr: 0 },
            tri_pipeline: std::ptr::null_mut(),
            tri_root_signature: std::ptr::null_mut(),
            tri_vertex_buffer: std::ptr::null_mut(),
            tri_ready: false,
            staging_buffer: std::ptr::null_mut(),
            fence,
            fence_event,
            fence_value: 0,
            resources_in_copy_source: false,
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
            mesh_ready: false,
            mesh_root_signature: std::ptr::null_mut(),
            mesh_pipeline: std::ptr::null_mut(),
            lit_ready: false,
            lit_root_signature: std::ptr::null_mut(),
            lit_pipeline: std::ptr::null_mut(),
            lit_lights_buffer: std::ptr::null_mut(),
            lit_block: LightsUniform::none(),
            gbuffer_ready: false,
            gbuffer_root_signature: std::ptr::null_mut(),
            gbuffer_pipeline: std::ptr::null_mut(),
            lighting_ready: false,
            lighting_root_signature: std::ptr::null_mut(),
            lighting_pipeline: std::ptr::null_mut(),
            lighting_lights_buffer: std::ptr::null_mut(),
            lighting_srv_heap: std::ptr::null_mut(),
            post_gbuffer_ready: false,
            post_gbuffer_root_signature: std::ptr::null_mut(),
            post_gbuffer_pipeline: std::ptr::null_mut(),
            post_composite_ready: false,
            post_composite_root_signature: std::ptr::null_mut(),
            post_composite_pipeline: std::ptr::null_mut(),
            post_lights_buffer: std::ptr::null_mut(),
            post_config_buffer: std::ptr::null_mut(),
            post_volume_buffer: std::ptr::null_mut(),
            post_srv_heap: std::ptr::null_mut(),
            light_volume: None,
            volume_uniform: [0.0; 12],
            bound_texture: None,
            default_texture: None,
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

    /// The off-screen frame target: the colour texture (RGBA8, the
    /// RENDER_TARGET flag), the depth/stencil texture (D24S8, the
    /// DEPTH_STENCIL flag), the RTV and DSV descriptor heaps and the
    /// staging buffer (the three-plane readback layout).
    /// The contract's PixelFormat to the DXGI format the device's
    /// resource carries (the same mapping the Metal backend's
    /// to_mtl_format records).
    fn to_d3d_format(&self, format: PixelFormat) -> u32 {
        match format {
            PixelFormat::Rgba8 => DXGI_FORMAT_R8G8B8A8_UNORM,
            PixelFormat::Rgba16F => DXGI_FORMAT_R16G16B16A16_FLOAT,
            PixelFormat::R8 => DXGI_FORMAT_R8_UNORM,
            PixelFormat::Depth24Stencil8 => DXGI_FORMAT_D24_UNORM_S8_UINT,
        }
    }

    fn initialise(&mut self) -> Result<(), RenderError> {
        let colour_clear = D3d12ClearValue {
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            color: [0.0, 0.0, 0.0, 1.0],
        };
        self.color_texture = unsafe {
            self_create_texture(
                self.device,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                self.config.width,
                self.config.height,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                Some(&colour_clear),
            )?
        };
        let depth_clear = D3d12ClearValue {
            format: DXGI_FORMAT_D24_UNORM_S8_UINT,
            color: [self.clear_depth, 0.0, 0.0, 0.0],
        };
        self.depth_texture = unsafe {
            self_create_texture(
                self.device,
                DXGI_FORMAT_D24_UNORM_S8_UINT,
                self.config.width,
                self.config.height,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                Some(&depth_clear),
            )?
        };
        self.staging_buffer = unsafe {
            self_create_buffer(
                self.device,
                STAGING_SIZE as u64,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?
        };
        self.rtv_heap = unsafe {
            self_create_descriptor_heap(
                self.device,
                D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                1,
                D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            )?
        };
        self.dsv_heap = unsafe {
            self_create_descriptor_heap(
                self.device,
                D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                1,
                D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            )?
        };
        self.rtv_handle = unsafe { self_heap_cpu_handle(self.rtv_heap) };
        self.dsv_handle = unsafe { self_heap_cpu_handle(self.dsv_heap) };
        unsafe {
            self_create_rtv_dsv(
                self.device,
                self.color_texture,
                self.depth_texture,
                self.rtv_handle,
                self.dsv_handle,
            );
        }
        // The default texture: one 1x1 opaque white texel the mesh path's
        // sampler reads when no texture is bound (the same default the
        // Vulkan and the OpenGL paths carry).
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
        Ok(())
    }

    /// Compiles the two HLSL sources into the diagnostic triangle's
    /// pipeline (D3DCompile at runtime, the same rule the OpenGL backend's
    /// GLSL runtime compilation records), serialises the root signature
    /// and creates the vertex buffer.
    fn ensure_triangle_resources(&mut self) -> bool {
        if self.tri_ready {
            return true;
        }
        let mut vs_blob: D3dBlobHandle = std::ptr::null_mut();
        let mut ps_blob: D3dBlobHandle = std::ptr::null_mut();
        let mut rs_blob: D3dBlobHandle = std::ptr::null_mut();
        let mut error_blob: D3dBlobHandle = std::ptr::null_mut();
        let vs_source = super::shader_library::TRIANGLE_VERTEX_SHADER;
        let ps_source = super::shader_library::TRIANGLE_PIXEL_SHADER;
        unsafe {
            let compile = self
                .loader
                .d3d_compile
                .expect("the loader resolved D3DCompile");
            let hr = compile(
                vs_source.as_ptr() as *const c_void,
                vs_source.len() as u64,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                b"vsMain\0".as_ptr() as *const c_char,
                b"vs_5_0\0".as_ptr() as *const c_char,
                0,
                0,
                &mut vs_blob,
                &mut error_blob,
            );
            if hr != S_OK || vs_blob.is_null() {
                return false;
            }
            let hr = compile(
                ps_source.as_ptr() as *const c_void,
                ps_source.len() as u64,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                b"psMain\0".as_ptr() as *const c_char,
                b"ps_5_0\0".as_ptr() as *const c_char,
                0,
                0,
                &mut ps_blob,
                &mut error_blob,
            );
            if hr != S_OK || ps_blob.is_null() {
                return false;
            }
        }
        // The blob sizes: the shader bytecodes' lengths (the PSO desc's
        // bytecode carries).
        let vs_blob_size = unsafe { call_slot_u64_0(vs_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
        let ps_blob_size = unsafe { call_slot_u64_0(ps_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
        // The root signature: one 32-bit constants root parameter (the
        // 20-word block the draw writes), SHADER_VISIBILITY_ALL.
        let constants = D3d12RootConstants {
            shader_register: 0,
            register_space: 0,
            num_32bit_values: 20,
        };
        let parameter = D3d12RootParameter {
            parameter_type: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            union_head_padding: [0],
            constants_or_table: d3d::D3d12RootParameterUnion { constants },
            union_tail_padding: [0],
            visibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let root_signature_desc = D3d12RootSignatureDesc {
            num_parameters: 1,
            parameters: &parameter,
            num_static_samplers: 0,
            static_samplers: std::ptr::null(),
            flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        };
        unsafe {
            let serialize = self
                .loader
                .d3d12_serialize_root_signature
                .expect("the loader resolved D3D12SerializeRootSignature");
            let hr = serialize(
                &root_signature_desc,
                d3d::D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut rs_blob,
                &mut error_blob,
            );
            if hr != S_OK || rs_blob.is_null() {
                return false;
            }
            let rs_bytes = std::slice::from_raw_parts(
                call_slot_ptr_0(rs_blob as ComObject, BLOB_GET_BUFFER_POINTER) as *const u8,
                call_slot_u64_0(rs_blob as ComObject, BLOB_GET_BUFFER_SIZE) as usize,
            );
            let mut root_signature: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_5(
                self.device as ComObject,
                DEVICE_CREATE_ROOT_SIGNATURE,
                0 as *mut c_void,
                rs_bytes.as_ptr() as *const c_void as *mut c_void,
                rs_bytes.len() as *mut c_void,
                IID_ID3D12_ROOT_SIGNATURE_WORDS.as_ptr() as *mut c_void,
                &mut root_signature as *mut _ as *mut c_void,
            );
            if hr != S_OK || root_signature.is_null() {
                return false;
            }
            self.tri_root_signature = root_signature;
        }
        // The PSO: the blend state writes all channels, the rasterizer
        // clips, the depth/stencil state is disabled (the triangle never
        // writes depth), the input layout carries one R32G32B32 position.
        let position = b"POSITION\0";
        let input_element = D3d12InputElementDesc {
            semantic_name: position.as_ptr() as *const c_char,
            semantic_index: 0,
            format: DXGI_FORMAT_R32G32B32_FLOAT,
            input_slot: 0,
            aligned_byte_offset: 0,
            input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            instance_data_step_rate: 0,
        };
        let blend_target = D3d12RenderTargetBlendDesc {
            blend_enable: 0,
            logic_op_enable: 0,
            src_blend: D3D12_BLEND_ONE,
            dest_blend: D3D12_BLEND_ZERO,
            blend_op: D3D12_BLEND_OP_ADD,
            src_blend_alpha: D3D12_BLEND_ONE,
            dest_blend_alpha: D3D12_BLEND_ZERO,
            blend_op_alpha: D3D12_BLEND_OP_ADD,
            logic_op: 0,
            render_target_write_mask: D3D12_COLOR_WRITE_ENABLE_ALL,
        };
        let blend_state = D3d12BlendDesc {
            alpha_to_coverage_enable: 0,
            independent_blend_enable: 0,
            render_target: [blend_target; 8],
        };
        let rasterizer = D3d12RasterizerDesc {
            fill_mode: D3D12_FILL_MODE_SOLID,
            cull_mode: D3D12_CULL_MODE_NONE,
            front_counter_clockwise: 0,
            depth_bias: 0,
            depth_bias_clamp: 0.0,
            slope_scaled_depth_bias: 0.0,
            depth_clip_enable: 1,
            multisample_enable: 0,
            antialiased_line_enable: 0,
            forced_sample_count: 0,
            conservative_raster: 0,
        };
        let depth_stencil = D3d12DepthStencilDesc {
            depth_enable: 0,
            depth_write_mask: D3D12_DEPTH_WRITE_MASK_ZERO,
            depth_func: 0,
            stencil_enable: 0,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            front_face: D3d12DepthStencilopDesc {
                stencil_fail_op: 0,
                stencil_depth_fail_op: 0,
                stencil_pass_op: 0,
                stencil_func: 0,
            },
            back_face: D3d12DepthStencilopDesc {
                stencil_fail_op: 0,
                stencil_depth_fail_op: 0,
                stencil_pass_op: 0,
                stencil_func: 0,
            },
        };
        let input_layout = D3d12InputLayoutDesc {
            num_elements: 1,
            elements: &input_element,
        };
        // The PSO desc's bytecode carries the blob's buffer (the
        // GetBufferPointer's pointer), not the blob interface.
        let (vs_bytecode, ps_bytecode) = unsafe {
            (
                call_slot_ptr_0(vs_blob as ComObject, BLOB_GET_BUFFER_POINTER),
                call_slot_ptr_0(ps_blob as ComObject, BLOB_GET_BUFFER_POINTER),
            )
        };
        let pso_desc = D3d12GraphicsPipelineStateDesc {
            root_signature: self.tri_root_signature,
            vs: D3d12ShaderBytecode {
                bytecode: vs_bytecode,
                length: vs_blob_size,
            },
            ps: D3d12ShaderBytecode {
                bytecode: ps_bytecode,
                length: ps_blob_size,
            },
            ds: D3d12ShaderBytecode {
                bytecode: std::ptr::null(),
                length: 0,
            },
            hs: D3d12ShaderBytecode {
                bytecode: std::ptr::null(),
                length: 0,
            },
            gs: D3d12ShaderBytecode {
                bytecode: std::ptr::null(),
                length: 0,
            },
            stream_output: D3d12StreamOutputDesc {
                so_declaration: std::ptr::null(),
                num_entries: 0,
                head_padding: [0],
                buffer_strides: std::ptr::null(),
                num_strides: 0,
                rasterized_stream: 0,
            },
            blend_state,
            sample_mask: 0xffffffff,
            rasterizer_state: rasterizer,
            depth_stencil_state: depth_stencil,
            input_layout,
            ib_strip_cut_value: 0,
            primitive_topology_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            num_render_targets: 1,
            rtv_formats: [DXGI_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 0, 0, 0],
            dsv_format: DXGI_FORMAT_D24_UNORM_S8_UINT,
            sample_desc: D3d12SampleDesc {
                count: 1,
                quality: 0,
            },
            node_mask: 0,
            cached_pso: D3d12CachedPipelineState {
                cached_blob: std::ptr::null_mut(),
                cached_blob_size: 0,
            },
            flags: 0,
        };
        let mut pipeline: *mut c_void = std::ptr::null_mut();
        unsafe {
            let hr = call_slot_i32_4(
                self.device as ComObject,
                DEVICE_CREATE_GRAPHICS_PIPELINE_STATE,
                &pso_desc as *const D3d12GraphicsPipelineStateDesc as *mut c_void,
                IID_ID3D12_PIPELINE_STATE_WORDS.as_ptr() as *mut c_void,
                &mut pipeline as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr != S_OK || pipeline.is_null() {
                return false;
            }
        }
        // The vertex buffer: the UPLOAD heap's mapped resource carrying
        // the three vertices.
        let vertices: [f32; 9] = [
            0.0, 0.8, 0.0, //
            -0.8, -0.8, 0.0, //
            0.8, -0.8, 0.0,
        ];
        let vertex_buffer = unsafe {
            self_create_buffer(
                self.device,
                (vertices.len() * 4) as u64,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )
        };
        let vertex_buffer = match vertex_buffer {
            Ok(buffer) => buffer,
            Err(_) => return false,
        };
        let mut map: *mut c_void = std::ptr::null_mut();
        let mapped = unsafe {
            call_slot_i32_4(
                vertex_buffer as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        if mapped != S_OK || map.is_null() {
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                map as *mut u8,
                vertices.len() * 4,
            );
            call_slot_i32_2(
                vertex_buffer as ComObject,
                RESOURCE_UNMAP,
                0 as *mut c_void,
                std::ptr::null_mut(),
            );
        }
        self.tri_pipeline = pipeline;
        self.tri_vertex_buffer = vertex_buffer;
        self.tri_ready = true;
        true
    }

    /// The mesh path's resources: the root signature (one descriptor
    /// table carrying the CBV at b0 and the SRV at t0, plus one static
    /// sampler at s0), and the pipeline state whose input layout carries
    /// the PNU's three attributes (position R32G32B32 at 0, normal
    /// R32G32B32 at 12, texcoord RG32 at 24) and whose depth/stencil
    /// state tests LESS and writes all.
    fn ensure_mesh_resources(&mut self) -> bool {
        if self.mesh_ready {
            return true;
        }
        let mut vs_blob: D3dBlobHandle = std::ptr::null_mut();
        let mut ps_blob: D3dBlobHandle = std::ptr::null_mut();
        let mut rs_blob: D3dBlobHandle = std::ptr::null_mut();
        let mut error_blob: D3dBlobHandle = std::ptr::null_mut();
        let vs_source = super::shader_library::MESH_VERTEX_SHADER;
        let ps_source = super::shader_library::MESH_PIXEL_SHADER;
        unsafe {
            let compile = self
                .loader
                .d3d_compile
                .expect("the loader resolved D3DCompile");
            let hr = compile(
                vs_source.as_ptr() as *const c_void,
                vs_source.len() as u64,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                b"vsMain\0".as_ptr() as *const c_char,
                b"vs_5_0\0".as_ptr() as *const c_char,
                0,
                0,
                &mut vs_blob,
                &mut error_blob,
            );
            if hr != S_OK || vs_blob.is_null() {
                return false;
            }
            let hr = compile(
                ps_source.as_ptr() as *const c_void,
                ps_source.len() as u64,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                b"psMain\0".as_ptr() as *const c_char,
                b"ps_5_0\0".as_ptr() as *const c_char,
                0,
                0,
                &mut ps_blob,
                &mut error_blob,
            );
            if hr != S_OK || ps_blob.is_null() {
                return false;
            }
        }
        let vs_blob_size = unsafe { call_slot_u64_0(vs_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
        let ps_blob_size = unsafe { call_slot_u64_0(ps_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
        // The root signature: one descriptor table (the CBV at b0 and the
        // SRV at t0) and one static sampler at s0 (the POINT filter the
        // smoke's NEAREST read carries).
        let ranges = [
            d3d::D3d12DescriptorRange {
                range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                num_descriptors: 1,
                base_shader_register: 0,
                register_space: 0,
                offset_in_descriptors_from_table_start: 0,
            },
            d3d::D3d12DescriptorRange {
                range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                num_descriptors: 1,
                base_shader_register: 0,
                register_space: 0,
                offset_in_descriptors_from_table_start: 1,
            },
        ];
        let table = d3d::D3d12RootDescriptorTable {
            num_descriptor_ranges: 2,
            descriptor_ranges: ranges.as_ptr(),
        };
        let parameter = D3d12RootParameter {
            parameter_type: d3d::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            union_head_padding: [0],
            constants_or_table: d3d::D3d12RootParameterUnion { table },
            union_tail_padding: [0],
            visibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let static_sampler = d3d::D3d12StaticSamplerDesc {
            filter: d3d::D3D12_FILTER_MIN_MAG_MIP_POINT,
            address_u: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            address_v: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            address_w: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            mip_lod_bias: 0.0,
            max_anisotropy: 0,
            comparison_func: d3d::D3D12_COMPARISON_FUNC_ALWAYS,
            border_color: d3d::D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
            min_lod: 0.0,
            max_lod: 3.402823466e38,
            shader_register: 0,
            register_space: 0,
            shader_visibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let root_signature_desc = D3d12RootSignatureDesc {
            num_parameters: 1,
            parameters: &parameter,
            num_static_samplers: 1,
            static_samplers: &static_sampler as *const _ as *const c_void,
            flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        };
        unsafe {
            let serialize = self
                .loader
                .d3d12_serialize_root_signature
                .expect("the loader resolved D3D12SerializeRootSignature");
            let hr = serialize(
                &root_signature_desc,
                d3d::D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut rs_blob,
                &mut error_blob,
            );
            if hr != S_OK || rs_blob.is_null() {
                return false;
            }
            let rs_bytes = std::slice::from_raw_parts(
                call_slot_ptr_0(rs_blob as ComObject, BLOB_GET_BUFFER_POINTER) as *const u8,
                call_slot_u64_0(rs_blob as ComObject, BLOB_GET_BUFFER_SIZE) as usize,
            );
            let mut root_signature: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_5(
                self.device as ComObject,
                DEVICE_CREATE_ROOT_SIGNATURE,
                0 as *mut c_void,
                rs_bytes.as_ptr() as *const c_void as *mut c_void,
                rs_bytes.len() as *mut c_void,
                IID_ID3D12_ROOT_SIGNATURE_WORDS.as_ptr() as *mut c_void,
                &mut root_signature as *mut _ as *mut c_void,
            );
            if hr != S_OK || root_signature.is_null() {
                return false;
            }
            self.mesh_root_signature = root_signature;
        }
        // The PSO: the same blend state the triangle path records, the
        // input layout carries the PNU's three attributes, the
        // depth/stencil state tests LESS and writes all.
        let position = b"POSITION\0";
        let normal = b"NORMAL\0";
        let texcoord = b"TEXCOORD\0";
        let input_elements = [
            D3d12InputElementDesc {
                semantic_name: position.as_ptr() as *const c_char,
                semantic_index: 0,
                format: DXGI_FORMAT_R32G32B32_FLOAT,
                input_slot: 0,
                aligned_byte_offset: 0,
                input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                instance_data_step_rate: 0,
            },
            D3d12InputElementDesc {
                semantic_name: normal.as_ptr() as *const c_char,
                semantic_index: 0,
                format: DXGI_FORMAT_R32G32B32_FLOAT,
                input_slot: 0,
                aligned_byte_offset: 12,
                input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                instance_data_step_rate: 0,
            },
            D3d12InputElementDesc {
                semantic_name: texcoord.as_ptr() as *const c_char,
                semantic_index: 0,
                format: DXGI_FORMAT_R32G32_FLOAT,
                input_slot: 0,
                aligned_byte_offset: 24,
                input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                instance_data_step_rate: 0,
            },
        ];
        let blend_target = D3d12RenderTargetBlendDesc {
            blend_enable: 0,
            logic_op_enable: 0,
            src_blend: D3D12_BLEND_ONE,
            dest_blend: D3D12_BLEND_ZERO,
            blend_op: D3D12_BLEND_OP_ADD,
            src_blend_alpha: D3D12_BLEND_ONE,
            dest_blend_alpha: D3D12_BLEND_ZERO,
            blend_op_alpha: D3D12_BLEND_OP_ADD,
            logic_op: 0,
            render_target_write_mask: D3D12_COLOR_WRITE_ENABLE_ALL,
        };
        let blend_state = D3d12BlendDesc {
            alpha_to_coverage_enable: 0,
            independent_blend_enable: 0,
            render_target: [blend_target; 8],
        };
        let rasterizer = D3d12RasterizerDesc {
            fill_mode: D3D12_FILL_MODE_SOLID,
            cull_mode: D3D12_CULL_MODE_NONE,
            front_counter_clockwise: 0,
            depth_bias: 0,
            depth_bias_clamp: 0.0,
            slope_scaled_depth_bias: 0.0,
            depth_clip_enable: 1,
            multisample_enable: 0,
            antialiased_line_enable: 0,
            forced_sample_count: 0,
            conservative_raster: 0,
        };
        let depth_stencil = D3d12DepthStencilDesc {
            depth_enable: 1,
            depth_write_mask: D3D12_DEPTH_WRITE_MASK_ALL,
            depth_func: D3D12_COMPARISON_FUNC_LESS,
            stencil_enable: 0,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            front_face: D3d12DepthStencilopDesc {
                stencil_fail_op: 0,
                stencil_depth_fail_op: 0,
                stencil_pass_op: 0,
                stencil_func: 0,
            },
            back_face: D3d12DepthStencilopDesc {
                stencil_fail_op: 0,
                stencil_depth_fail_op: 0,
                stencil_pass_op: 0,
                stencil_func: 0,
            },
        };
        let input_layout = D3d12InputLayoutDesc {
            num_elements: 3,
            elements: input_elements.as_ptr(),
        };
        let (vs_bytecode, ps_bytecode) = unsafe {
            (
                call_slot_ptr_0(vs_blob as ComObject, BLOB_GET_BUFFER_POINTER),
                call_slot_ptr_0(ps_blob as ComObject, BLOB_GET_BUFFER_POINTER),
            )
        };
        let pso_desc = D3d12GraphicsPipelineStateDesc {
            root_signature: self.mesh_root_signature,
            vs: D3d12ShaderBytecode {
                bytecode: vs_bytecode,
                length: vs_blob_size,
            },
            ps: D3d12ShaderBytecode {
                bytecode: ps_bytecode,
                length: ps_blob_size,
            },
            ds: D3d12ShaderBytecode {
                bytecode: std::ptr::null(),
                length: 0,
            },
            hs: D3d12ShaderBytecode {
                bytecode: std::ptr::null(),
                length: 0,
            },
            gs: D3d12ShaderBytecode {
                bytecode: std::ptr::null(),
                length: 0,
            },
            stream_output: D3d12StreamOutputDesc {
                so_declaration: std::ptr::null(),
                num_entries: 0,
                head_padding: [0],
                buffer_strides: std::ptr::null(),
                num_strides: 0,
                rasterized_stream: 0,
            },
            blend_state,
            sample_mask: 0xffffffff,
            rasterizer_state: rasterizer,
            depth_stencil_state: depth_stencil,
            input_layout,
            ib_strip_cut_value: 0,
            primitive_topology_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            num_render_targets: 1,
            rtv_formats: [DXGI_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 0, 0, 0],
            dsv_format: DXGI_FORMAT_D24_UNORM_S8_UINT,
            sample_desc: D3d12SampleDesc {
                count: 1,
                quality: 0,
            },
            node_mask: 0,
            cached_pso: D3d12CachedPipelineState {
                cached_blob: std::ptr::null_mut(),
                cached_blob_size: 0,
            },
            flags: 0,
        };
        let mut pipeline: *mut c_void = std::ptr::null_mut();
        unsafe {
            let hr = call_slot_i32_4(
                self.device as ComObject,
                DEVICE_CREATE_GRAPHICS_PIPELINE_STATE,
                &pso_desc as *const D3d12GraphicsPipelineStateDesc as *mut c_void,
                IID_ID3D12_PIPELINE_STATE_WORDS.as_ptr() as *mut c_void,
                &mut pipeline as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr != S_OK || pipeline.is_null() {
                return false;
            }
        }
        self.mesh_pipeline = pipeline;
        self.mesh_ready = true;
        true
    }

    /// The lit mesh path's resources (M5): the LIT shader pair compiled
    /// by D3DCompile, the root signature whose table carries the frame's
    /// CBV at b0, the texture's SRV at t0 and the lights' CBV at b1, and
    /// the PSO whose input layout carries the PNU's three attributes.
    /// The same lazy ensure the unlit path follows; the compilation
    /// happens before a frame opens (the upload's ensure, not the
    /// draw's).
    fn ensure_lit_resources(&mut self) -> bool {
        if self.lit_ready {
            return true;
        }
        let mut vs_blob: D3dBlobHandle = std::ptr::null_mut();
        let mut ps_blob: D3dBlobHandle = std::ptr::null_mut();
        let mut rs_blob: D3dBlobHandle = std::ptr::null_mut();
        let mut error_blob: D3dBlobHandle = std::ptr::null_mut();
        let vs_source = super::shader_library::LIT_VERTEX_SHADER;
        let ps_source = super::shader_library::LIT_PIXEL_SHADER;
        unsafe {
            let compile = self
                .loader
                .d3d_compile
                .expect("the loader resolved D3DCompile");
            let hr = compile(
                vs_source.as_ptr() as *const c_void,
                vs_source.len() as u64,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                b"vsMain\0".as_ptr() as *const c_char,
                b"vs_5_0\0".as_ptr() as *const c_char,
                0,
                0,
                &mut vs_blob,
                &mut error_blob,
            );
            if hr != S_OK || vs_blob.is_null() {
                return false;
            }
            let hr = compile(
                ps_source.as_ptr() as *const c_void,
                ps_source.len() as u64,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                b"psMain\0".as_ptr() as *const c_char,
                b"ps_5_0\0".as_ptr() as *const c_char,
                0,
                0,
                &mut ps_blob,
                &mut error_blob,
            );
            if hr != S_OK || ps_blob.is_null() {
                return false;
            }
        }
        let vs_blob_size = unsafe { call_slot_u64_0(vs_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
        let ps_blob_size = unsafe { call_slot_u64_0(ps_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
        // The root signature: one descriptor table (the frame's CBV at
        // b0, the SRV at t0, the lights' CBV at b1 — the heap's slots 0,
        // 1 and 2) and one static sampler at s0.
        let ranges = [
            d3d::D3d12DescriptorRange {
                range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                num_descriptors: 1,
                base_shader_register: 0,
                register_space: 0,
                offset_in_descriptors_from_table_start: 0,
            },
            d3d::D3d12DescriptorRange {
                range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                num_descriptors: 1,
                base_shader_register: 0,
                register_space: 0,
                offset_in_descriptors_from_table_start: 1,
            },
            d3d::D3d12DescriptorRange {
                range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                num_descriptors: 1,
                base_shader_register: 1,
                register_space: 0,
                offset_in_descriptors_from_table_start: 2,
            },
        ];
        let table = d3d::D3d12RootDescriptorTable {
            num_descriptor_ranges: 3,
            descriptor_ranges: ranges.as_ptr(),
        };
        let parameter = D3d12RootParameter {
            parameter_type: d3d::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            union_head_padding: [0],
            constants_or_table: d3d::D3d12RootParameterUnion { table },
            union_tail_padding: [0],
            visibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let static_sampler = d3d::D3d12StaticSamplerDesc {
            filter: d3d::D3D12_FILTER_MIN_MAG_MIP_POINT,
            address_u: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            address_v: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            address_w: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            mip_lod_bias: 0.0,
            max_anisotropy: 0,
            comparison_func: d3d::D3D12_COMPARISON_FUNC_ALWAYS,
            border_color: d3d::D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
            min_lod: 0.0,
            max_lod: 3.402823466e38,
            shader_register: 0,
            register_space: 0,
            shader_visibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let root_signature_desc = D3d12RootSignatureDesc {
            num_parameters: 1,
            parameters: &parameter,
            num_static_samplers: 1,
            static_samplers: &static_sampler as *const _ as *const c_void,
            flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        };
        unsafe {
            let serialize = self
                .loader
                .d3d12_serialize_root_signature
                .expect("the loader resolved D3D12SerializeRootSignature");
            let hr = serialize(
                &root_signature_desc,
                d3d::D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut rs_blob,
                &mut error_blob,
            );
            if hr != S_OK || rs_blob.is_null() {
                return false;
            }
            let rs_bytes = std::slice::from_raw_parts(
                call_slot_ptr_0(rs_blob as ComObject, BLOB_GET_BUFFER_POINTER) as *const u8,
                call_slot_u64_0(rs_blob as ComObject, BLOB_GET_BUFFER_SIZE) as usize,
            );
            let mut root_signature: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_5(
                self.device as ComObject,
                DEVICE_CREATE_ROOT_SIGNATURE,
                0 as *mut c_void,
                rs_bytes.as_ptr() as *const c_void as *mut c_void,
                rs_bytes.len() as *mut c_void,
                IID_ID3D12_ROOT_SIGNATURE_WORDS.as_ptr() as *mut c_void,
                &mut root_signature as *mut _ as *mut c_void,
            );
            if hr != S_OK || root_signature.is_null() {
                return false;
            }
            self.lit_root_signature = root_signature;
        }
        // The PSO: the same states the mesh PSO carries (the PNU input
        // layout, LESS depth, no culling), the LIT shaders' bytecode.
        let position = b"POSITION\0";
        let normal = b"NORMAL\0";
        let texcoord = b"TEXCOORD\0";
        let input_elements = [
            D3d12InputElementDesc {
                semantic_name: position.as_ptr() as *const c_char,
                semantic_index: 0,
                format: DXGI_FORMAT_R32G32B32_FLOAT,
                input_slot: 0,
                aligned_byte_offset: 0,
                input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                instance_data_step_rate: 0,
            },
            D3d12InputElementDesc {
                semantic_name: normal.as_ptr() as *const c_char,
                semantic_index: 0,
                format: DXGI_FORMAT_R32G32B32_FLOAT,
                input_slot: 0,
                aligned_byte_offset: 12,
                input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                instance_data_step_rate: 0,
            },
            D3d12InputElementDesc {
                semantic_name: texcoord.as_ptr() as *const c_char,
                semantic_index: 0,
                format: DXGI_FORMAT_R32G32_FLOAT,
                input_slot: 0,
                aligned_byte_offset: 24,
                input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                instance_data_step_rate: 0,
            },
        ];
        let blend_target = D3d12RenderTargetBlendDesc {
            blend_enable: 0,
            logic_op_enable: 0,
            src_blend: D3D12_BLEND_ONE,
            dest_blend: D3D12_BLEND_ZERO,
            blend_op: D3D12_BLEND_OP_ADD,
            src_blend_alpha: D3D12_BLEND_ONE,
            dest_blend_alpha: D3D12_BLEND_ZERO,
            blend_op_alpha: D3D12_BLEND_OP_ADD,
            logic_op: 0,
            render_target_write_mask: D3D12_COLOR_WRITE_ENABLE_ALL,
        };
        let blend_state = D3d12BlendDesc {
            alpha_to_coverage_enable: 0,
            independent_blend_enable: 0,
            render_target: [blend_target; 8],
        };
        let rasterizer = D3d12RasterizerDesc {
            fill_mode: D3D12_FILL_MODE_SOLID,
            cull_mode: D3D12_CULL_MODE_NONE,
            front_counter_clockwise: 0,
            depth_bias: 0,
            depth_bias_clamp: 0.0,
            slope_scaled_depth_bias: 0.0,
            depth_clip_enable: 1,
            multisample_enable: 0,
            antialiased_line_enable: 0,
            forced_sample_count: 0,
            conservative_raster: 0,
        };
        let depth_stencil = D3d12DepthStencilDesc {
            depth_enable: 1,
            depth_write_mask: D3D12_DEPTH_WRITE_MASK_ALL,
            depth_func: D3D12_COMPARISON_FUNC_LESS,
            stencil_enable: 0,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            front_face: D3d12DepthStencilopDesc {
                stencil_fail_op: 0,
                stencil_depth_fail_op: 0,
                stencil_pass_op: 0,
                stencil_func: 0,
            },
            back_face: D3d12DepthStencilopDesc {
                stencil_fail_op: 0,
                stencil_depth_fail_op: 0,
                stencil_pass_op: 0,
                stencil_func: 0,
            },
        };
        let input_layout = D3d12InputLayoutDesc {
            num_elements: 3,
            elements: input_elements.as_ptr(),
        };
        let (vs_bytecode, ps_bytecode) = unsafe {
            (
                call_slot_ptr_0(vs_blob as ComObject, BLOB_GET_BUFFER_POINTER),
                call_slot_ptr_0(ps_blob as ComObject, BLOB_GET_BUFFER_POINTER),
            )
        };
        let pso_desc = D3d12GraphicsPipelineStateDesc {
            root_signature: self.lit_root_signature,
            vs: D3d12ShaderBytecode {
                bytecode: vs_bytecode,
                length: vs_blob_size,
            },
            ps: D3d12ShaderBytecode {
                bytecode: ps_bytecode,
                length: ps_blob_size,
            },
            ds: D3d12ShaderBytecode {
                bytecode: std::ptr::null(),
                length: 0,
            },
            hs: D3d12ShaderBytecode {
                bytecode: std::ptr::null(),
                length: 0,
            },
            gs: D3d12ShaderBytecode {
                bytecode: std::ptr::null(),
                length: 0,
            },
            stream_output: D3d12StreamOutputDesc {
                so_declaration: std::ptr::null(),
                num_entries: 0,
                head_padding: [0],
                buffer_strides: std::ptr::null(),
                num_strides: 0,
                rasterized_stream: 0,
            },
            blend_state,
            sample_mask: 0xffffffff,
            rasterizer_state: rasterizer,
            depth_stencil_state: depth_stencil,
            input_layout,
            ib_strip_cut_value: 0,
            primitive_topology_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            num_render_targets: 1,
            rtv_formats: [DXGI_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 0, 0, 0],
            dsv_format: DXGI_FORMAT_D24_UNORM_S8_UINT,
            sample_desc: D3d12SampleDesc {
                count: 1,
                quality: 0,
            },
            node_mask: 0,
            cached_pso: D3d12CachedPipelineState {
                cached_blob: std::ptr::null_mut(),
                cached_blob_size: 0,
            },
            flags: 0,
        };
        let mut pipeline: *mut c_void = std::ptr::null_mut();
        unsafe {
            let hr = call_slot_i32_4(
                self.device as ComObject,
                DEVICE_CREATE_GRAPHICS_PIPELINE_STATE,
                &pso_desc as *const D3d12GraphicsPipelineStateDesc as *mut c_void,
                IID_ID3D12_PIPELINE_STATE_WORDS.as_ptr() as *mut c_void,
                &mut pipeline as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr != S_OK || pipeline.is_null() {
                return false;
            }
        }
        self.lit_pipeline = pipeline;
        self.lit_ready = true;
        true
    }

    /// The deferred lighting path's resources (M6): the two PSOs (the
    /// G-buffer's geometry PSO with the three-attachment pixel shader,
    /// the fullscreen lighting PSO) and their root signatures, plus the
    /// shared 256-byte lights' buffer the lighting draw's CBV at b1
    /// reads. The same outside-the-frame rule the lit path's ensure
    /// follows (the PSO creation inside an open frame's recording is
    /// the crash the debug run caught).
    fn ensure_deferred_resources(&mut self) -> bool {
        if self.gbuffer_ready && self.lighting_ready {
            return true;
        }
        // ---- The G-buffer's root signature + PSO ----
        // The same table the lit path carries: the frame's CBV at b0,
        // the SRV at t0, the lights' CBV at b1 (the lighting pixel
        // shader's own copy waits for its signature below); one static
        // sampler at s0.
        if !self.gbuffer_ready {
            let mut rs_blob: D3dBlobHandle = std::ptr::null_mut();
            let mut error_blob: D3dBlobHandle = std::ptr::null_mut();
            let vs_source = super::shader_library::GBUFFER_VERTEX_SHADER;
            let ps_source = super::shader_library::GBUFFER_PIXEL_SHADER;
            let mut vs_blob: D3dBlobHandle = std::ptr::null_mut();
            let mut ps_blob: D3dBlobHandle = std::ptr::null_mut();
            unsafe {
                let compile = self
                    .loader
                    .d3d_compile
                    .expect("the loader resolved D3DCompile");
                let hr = compile(
                    vs_source.as_ptr() as *const c_void,
                    vs_source.len() as u64,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    b"vsMain\0".as_ptr() as *const c_char,
                    b"vs_5_0\0".as_ptr() as *const c_char,
                    0,
                    0,
                    &mut vs_blob,
                    &mut error_blob,
                );
                if hr != S_OK || vs_blob.is_null() {
                    return false;
                }
                let hr = compile(
                    ps_source.as_ptr() as *const c_void,
                    ps_source.len() as u64,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    b"psMain\0".as_ptr() as *const c_char,
                    b"ps_5_0\0".as_ptr() as *const c_char,
                    0,
                    0,
                    &mut ps_blob,
                    &mut error_blob,
                );
                if hr != S_OK || ps_blob.is_null() {
                    return false;
                }
            }
            let vs_blob_size =
                unsafe { call_slot_u64_0(vs_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
            let ps_blob_size =
                unsafe { call_slot_u64_0(ps_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
            let ranges = [
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                    num_descriptors: 1,
                    base_shader_register: 0,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 0,
                },
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    num_descriptors: 1,
                    base_shader_register: 0,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 1,
                },
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                    num_descriptors: 1,
                    base_shader_register: 1,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 2,
                },
            ];
            let table = d3d::D3d12RootDescriptorTable {
                num_descriptor_ranges: 3,
                descriptor_ranges: ranges.as_ptr(),
            };
            let parameter = D3d12RootParameter {
                parameter_type: d3d::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                union_head_padding: [0],
                constants_or_table: d3d::D3d12RootParameterUnion { table },
                union_tail_padding: [0],
                visibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let static_sampler = d3d::D3d12StaticSamplerDesc {
                filter: d3d::D3D12_FILTER_MIN_MAG_MIP_POINT,
                address_u: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                address_v: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                address_w: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                mip_lod_bias: 0.0,
                max_anisotropy: 0,
                comparison_func: d3d::D3D12_COMPARISON_FUNC_ALWAYS,
                border_color: d3d::D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
                min_lod: 0.0,
                max_lod: 3.402823466e38,
                shader_register: 0,
                register_space: 0,
                shader_visibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let root_signature_desc = D3d12RootSignatureDesc {
                num_parameters: 1,
                parameters: &parameter,
                num_static_samplers: 1,
                static_samplers: &static_sampler as *const _ as *const c_void,
                flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            };
            unsafe {
                let serialize = self
                    .loader
                    .d3d12_serialize_root_signature
                    .expect("the loader resolved D3D12SerializeRootSignature");
                let hr = serialize(
                    &root_signature_desc,
                    d3d::D3D_ROOT_SIGNATURE_VERSION_1_0,
                    &mut rs_blob,
                    &mut error_blob,
                );
                if hr != S_OK || rs_blob.is_null() {
                    return false;
                }
                let rs_bytes = std::slice::from_raw_parts(
                    call_slot_ptr_0(rs_blob as ComObject, BLOB_GET_BUFFER_POINTER) as *const u8,
                    call_slot_u64_0(rs_blob as ComObject, BLOB_GET_BUFFER_SIZE) as usize,
                );
                let mut root_signature: *mut c_void = std::ptr::null_mut();
                let hr = call_slot_i32_5(
                    self.device as ComObject,
                    DEVICE_CREATE_ROOT_SIGNATURE,
                    0 as *mut c_void,
                    rs_bytes.as_ptr() as *const c_void as *mut c_void,
                    rs_bytes.len() as *mut c_void,
                    IID_ID3D12_ROOT_SIGNATURE_WORDS.as_ptr() as *mut c_void,
                    &mut root_signature as *mut _ as *mut c_void,
                );
                if hr != S_OK || root_signature.is_null() {
                    return false;
                }
                self.gbuffer_root_signature = root_signature;
            }
            // The PSO: the same states the lit PSO carries, the
            // three-attachment pixel shader's bytecode, NumRenderTargets
            // 3 with the G-buffer's formats.
            let position = b"POSITION\0";
            let normal = b"NORMAL\0";
            let texcoord = b"TEXCOORD\0";
            let input_elements = [
                D3d12InputElementDesc {
                    semantic_name: position.as_ptr() as *const c_char,
                    semantic_index: 0,
                    format: DXGI_FORMAT_R32G32B32_FLOAT,
                    input_slot: 0,
                    aligned_byte_offset: 0,
                    input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    instance_data_step_rate: 0,
                },
                D3d12InputElementDesc {
                    semantic_name: normal.as_ptr() as *const c_char,
                    semantic_index: 0,
                    format: DXGI_FORMAT_R32G32B32_FLOAT,
                    input_slot: 0,
                    aligned_byte_offset: 12,
                    input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    instance_data_step_rate: 0,
                },
                D3d12InputElementDesc {
                    semantic_name: texcoord.as_ptr() as *const c_char,
                    semantic_index: 0,
                    format: DXGI_FORMAT_R32G32_FLOAT,
                    input_slot: 0,
                    aligned_byte_offset: 24,
                    input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    instance_data_step_rate: 0,
                },
            ];
            let blend_target = D3d12RenderTargetBlendDesc {
                blend_enable: 0,
                logic_op_enable: 0,
                src_blend: D3D12_BLEND_ONE,
                dest_blend: D3D12_BLEND_ZERO,
                blend_op: D3D12_BLEND_OP_ADD,
                src_blend_alpha: D3D12_BLEND_ONE,
                dest_blend_alpha: D3D12_BLEND_ZERO,
                blend_op_alpha: D3D12_BLEND_OP_ADD,
                logic_op: 0,
                render_target_write_mask: D3D12_COLOR_WRITE_ENABLE_ALL,
            };
            let blend_state = D3d12BlendDesc {
                alpha_to_coverage_enable: 0,
                independent_blend_enable: 0,
                render_target: [blend_target; 8],
            };
            let rasterizer = D3d12RasterizerDesc {
                fill_mode: D3D12_FILL_MODE_SOLID,
                cull_mode: D3D12_CULL_MODE_NONE,
                front_counter_clockwise: 0,
                depth_bias: 0,
                depth_bias_clamp: 0.0,
                slope_scaled_depth_bias: 0.0,
                depth_clip_enable: 1,
                multisample_enable: 0,
                antialiased_line_enable: 0,
                forced_sample_count: 0,
                conservative_raster: 0,
            };
            let depth_stencil = D3d12DepthStencilDesc {
                depth_enable: 1,
                depth_write_mask: D3D12_DEPTH_WRITE_MASK_ALL,
                depth_func: D3D12_COMPARISON_FUNC_LESS,
                stencil_enable: 0,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                front_face: D3d12DepthStencilopDesc {
                    stencil_fail_op: 0,
                    stencil_depth_fail_op: 0,
                    stencil_pass_op: 0,
                    stencil_func: 0,
                },
                back_face: D3d12DepthStencilopDesc {
                    stencil_fail_op: 0,
                    stencil_depth_fail_op: 0,
                    stencil_pass_op: 0,
                    stencil_func: 0,
                },
            };
            let input_layout = D3d12InputLayoutDesc {
                num_elements: 3,
                elements: input_elements.as_ptr(),
            };
            let (vs_bytecode, ps_bytecode) = unsafe {
                (
                    call_slot_ptr_0(vs_blob as ComObject, BLOB_GET_BUFFER_POINTER),
                    call_slot_ptr_0(ps_blob as ComObject, BLOB_GET_BUFFER_POINTER),
                )
            };
            let pso_desc = D3d12GraphicsPipelineStateDesc {
                root_signature: self.gbuffer_root_signature,
                vs: D3d12ShaderBytecode {
                    bytecode: vs_bytecode,
                    length: vs_blob_size,
                },
                ps: D3d12ShaderBytecode {
                    bytecode: ps_bytecode,
                    length: ps_blob_size,
                },
                ds: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                hs: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                gs: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                stream_output: D3d12StreamOutputDesc {
                    so_declaration: std::ptr::null(),
                    num_entries: 0,
                    head_padding: [0],
                    buffer_strides: std::ptr::null(),
                    num_strides: 0,
                    rasterized_stream: 0,
                },
                blend_state,
                sample_mask: 0xffffffff,
                rasterizer_state: rasterizer,
                depth_stencil_state: depth_stencil,
                input_layout,
                ib_strip_cut_value: 0,
                primitive_topology_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                num_render_targets: 3,
                rtv_formats: [
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    DXGI_FORMAT_R16G16B16A16_FLOAT,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dsv_format: DXGI_FORMAT_D24_UNORM_S8_UINT,
                sample_desc: D3d12SampleDesc {
                    count: 1,
                    quality: 0,
                },
                node_mask: 0,
                cached_pso: D3d12CachedPipelineState {
                    cached_blob: std::ptr::null_mut(),
                    cached_blob_size: 0,
                },
                flags: 0,
            };
            let mut pipeline: *mut c_void = std::ptr::null_mut();
            unsafe {
                let hr = call_slot_i32_4(
                    self.device as ComObject,
                    DEVICE_CREATE_GRAPHICS_PIPELINE_STATE,
                    &pso_desc as *const D3d12GraphicsPipelineStateDesc as *mut c_void,
                    IID_ID3D12_PIPELINE_STATE_WORDS.as_ptr() as *mut c_void,
                    &mut pipeline as *mut _ as *mut c_void,
                    std::ptr::null_mut(),
                );
                if hr != S_OK || pipeline.is_null() {
                    return false;
                }
            }
            self.gbuffer_pipeline = pipeline;
            self.gbuffer_ready = true;
        }
        // ---- The lighting pass's root signature + PSO ----
        if !self.lighting_ready {
            let mut vs_blob: D3dBlobHandle = std::ptr::null_mut();
            let mut ps_blob: D3dBlobHandle = std::ptr::null_mut();
            let mut rs_blob: D3dBlobHandle = std::ptr::null_mut();
            let mut error_blob: D3dBlobHandle = std::ptr::null_mut();
            let vs_source = super::shader_library::LIGHTING_VERTEX_SHADER;
            let ps_source = super::shader_library::LIGHTING_PIXEL_SHADER;
            unsafe {
                let compile = self
                    .loader
                    .d3d_compile
                    .expect("the loader resolved D3DCompile");
                let hr = compile(
                    vs_source.as_ptr() as *const c_void,
                    vs_source.len() as u64,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    b"vsMain\0".as_ptr() as *const c_char,
                    b"vs_5_0\0".as_ptr() as *const c_char,
                    0,
                    0,
                    &mut vs_blob,
                    &mut error_blob,
                );
                if hr != S_OK || vs_blob.is_null() {
                    return false;
                }
                let hr = compile(
                    ps_source.as_ptr() as *const c_void,
                    ps_source.len() as u64,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    b"psMain\0".as_ptr() as *const c_char,
                    b"ps_5_0\0".as_ptr() as *const c_char,
                    0,
                    0,
                    &mut ps_blob,
                    &mut error_blob,
                );
                if hr != S_OK || ps_blob.is_null() {
                    return false;
                }
            }
            let vs_blob_size =
                unsafe { call_slot_u64_0(vs_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
            let ps_blob_size =
                unsafe { call_slot_u64_0(ps_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
            // The root signature: the lights' CBV at b1 and the three
            // SRVs at t0-t2 (the descriptor table's slots), one static
            // sampler at s0. No input assembler (the fullscreen
            // triangle's vertex ID generates the corners).
            let ranges = [
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    num_descriptors: 3,
                    base_shader_register: 0,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 0,
                },
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                    num_descriptors: 1,
                    base_shader_register: 1,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 3,
                },
            ];
            let table = d3d::D3d12RootDescriptorTable {
                num_descriptor_ranges: 2,
                descriptor_ranges: ranges.as_ptr(),
            };
            let parameter = D3d12RootParameter {
                parameter_type: d3d::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                union_head_padding: [0],
                constants_or_table: d3d::D3d12RootParameterUnion { table },
                union_tail_padding: [0],
                visibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let static_sampler = d3d::D3d12StaticSamplerDesc {
                filter: d3d::D3D12_FILTER_MIN_MAG_MIP_POINT,
                address_u: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                address_v: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                address_w: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                mip_lod_bias: 0.0,
                max_anisotropy: 0,
                comparison_func: d3d::D3D12_COMPARISON_FUNC_ALWAYS,
                border_color: d3d::D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
                min_lod: 0.0,
                max_lod: 3.402823466e38,
                shader_register: 0,
                register_space: 0,
                shader_visibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let root_signature_desc = D3d12RootSignatureDesc {
                num_parameters: 1,
                parameters: &parameter,
                num_static_samplers: 1,
                static_samplers: &static_sampler as *const _ as *const c_void,
                flags: 0,
            };
            unsafe {
                let serialize = self
                    .loader
                    .d3d12_serialize_root_signature
                    .expect("the loader resolved D3D12SerializeRootSignature");
                let hr = serialize(
                    &root_signature_desc,
                    d3d::D3D_ROOT_SIGNATURE_VERSION_1_0,
                    &mut rs_blob,
                    &mut error_blob,
                );
                if hr != S_OK || rs_blob.is_null() {
                    return false;
                }
                let rs_bytes = std::slice::from_raw_parts(
                    call_slot_ptr_0(rs_blob as ComObject, BLOB_GET_BUFFER_POINTER) as *const u8,
                    call_slot_u64_0(rs_blob as ComObject, BLOB_GET_BUFFER_SIZE) as usize,
                );
                let mut root_signature: *mut c_void = std::ptr::null_mut();
                let hr = call_slot_i32_5(
                    self.device as ComObject,
                    DEVICE_CREATE_ROOT_SIGNATURE,
                    0 as *mut c_void,
                    rs_bytes.as_ptr() as *const c_void as *mut c_void,
                    rs_bytes.len() as *mut c_void,
                    IID_ID3D12_ROOT_SIGNATURE_WORDS.as_ptr() as *mut c_void,
                    &mut root_signature as *mut _ as *mut c_void,
                );
                if hr != S_OK || root_signature.is_null() {
                    return false;
                }
                self.lighting_root_signature = root_signature;
            }
            // The PSO: no input layout, the depth test off (the
            // fullscreen triangle covers the frame without the depth
            // test), one render target.
            let blend_target = D3d12RenderTargetBlendDesc {
                blend_enable: 0,
                logic_op_enable: 0,
                src_blend: D3D12_BLEND_ONE,
                dest_blend: D3D12_BLEND_ZERO,
                blend_op: D3D12_BLEND_OP_ADD,
                src_blend_alpha: D3D12_BLEND_ONE,
                dest_blend_alpha: D3D12_BLEND_ZERO,
                blend_op_alpha: D3D12_BLEND_OP_ADD,
                logic_op: 0,
                render_target_write_mask: D3D12_COLOR_WRITE_ENABLE_ALL,
            };
            let blend_state = D3d12BlendDesc {
                alpha_to_coverage_enable: 0,
                independent_blend_enable: 0,
                render_target: [blend_target; 8],
            };
            let rasterizer = D3d12RasterizerDesc {
                fill_mode: D3D12_FILL_MODE_SOLID,
                cull_mode: D3D12_CULL_MODE_NONE,
                front_counter_clockwise: 0,
                depth_bias: 0,
                depth_bias_clamp: 0.0,
                slope_scaled_depth_bias: 0.0,
                depth_clip_enable: 1,
                multisample_enable: 0,
                antialiased_line_enable: 0,
                forced_sample_count: 0,
                conservative_raster: 0,
            };
            let depth_stencil = D3d12DepthStencilDesc {
                depth_enable: 0,
                depth_write_mask: D3D12_DEPTH_WRITE_MASK_ZERO,
                depth_func: D3D12_COMPARISON_FUNC_LESS,
                stencil_enable: 0,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                front_face: D3d12DepthStencilopDesc {
                    stencil_fail_op: 0,
                    stencil_depth_fail_op: 0,
                    stencil_pass_op: 0,
                    stencil_func: 0,
                },
                back_face: D3d12DepthStencilopDesc {
                    stencil_fail_op: 0,
                    stencil_depth_fail_op: 0,
                    stencil_pass_op: 0,
                    stencil_func: 0,
                },
            };
            let input_layout = D3d12InputLayoutDesc {
                num_elements: 0,
                elements: std::ptr::null(),
            };
            let (vs_bytecode, ps_bytecode) = unsafe {
                (
                    call_slot_ptr_0(vs_blob as ComObject, BLOB_GET_BUFFER_POINTER),
                    call_slot_ptr_0(ps_blob as ComObject, BLOB_GET_BUFFER_POINTER),
                )
            };
            let pso_desc = D3d12GraphicsPipelineStateDesc {
                root_signature: self.lighting_root_signature,
                vs: D3d12ShaderBytecode {
                    bytecode: vs_bytecode,
                    length: vs_blob_size,
                },
                ps: D3d12ShaderBytecode {
                    bytecode: ps_bytecode,
                    length: ps_blob_size,
                },
                ds: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                hs: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                gs: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                stream_output: D3d12StreamOutputDesc {
                    so_declaration: std::ptr::null(),
                    num_entries: 0,
                    head_padding: [0],
                    buffer_strides: std::ptr::null(),
                    num_strides: 0,
                    rasterized_stream: 0,
                },
                blend_state,
                sample_mask: 0xffffffff,
                rasterizer_state: rasterizer,
                depth_stencil_state: depth_stencil,
                input_layout,
                ib_strip_cut_value: 0,
                primitive_topology_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                num_render_targets: 1,
                rtv_formats: [DXGI_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 0, 0, 0],
                dsv_format: DXGI_FORMAT_D24_UNORM_S8_UINT,
                sample_desc: D3d12SampleDesc {
                    count: 1,
                    quality: 0,
                },
                node_mask: 0,
                cached_pso: D3d12CachedPipelineState {
                    cached_blob: std::ptr::null_mut(),
                    cached_blob_size: 0,
                },
                flags: 0,
            };
            let mut pipeline: *mut c_void = std::ptr::null_mut();
            unsafe {
                let hr = call_slot_i32_4(
                    self.device as ComObject,
                    DEVICE_CREATE_GRAPHICS_PIPELINE_STATE,
                    &pso_desc as *const D3d12GraphicsPipelineStateDesc as *mut c_void,
                    IID_ID3D12_PIPELINE_STATE_WORDS.as_ptr() as *mut c_void,
                    &mut pipeline as *mut _ as *mut c_void,
                    std::ptr::null_mut(),
                );
                if hr != S_OK || pipeline.is_null() {
                    return false;
                }
            }
            self.lighting_pipeline = pipeline;
            // The lights' shared buffer (the same 256-byte UPLOAD-heap
            // buffer the lit draw's CBV reads).
            if self.lighting_lights_buffer.is_null() {
                match unsafe {
                    self_create_buffer(
                        self.device,
                        256,
                        D3D12_HEAP_TYPE_UPLOAD,
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                    )
                } {
                    Ok(buffer) => self.lighting_lights_buffer = buffer,
                    Err(_) => return false,
                }
            }
            // The lighting pass's own shader-visible heap: four
            // descriptors (the three G-buffer SRVs at slots 0-2, the
            // lights' CBV at slot 3).
            if self.lighting_srv_heap.is_null() {
                match unsafe {
                    self_create_descriptor_heap(
                        self.device,
                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        4,
                        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    )
                } {
                    Ok(heap) => self.lighting_srv_heap = heap,
                    Err(_) => return false,
                }
            }
            self.lighting_ready = true;
        }
        true
    }

    /// The post-processing chain's resources (M7): the HDR geometry
    /// pass's PSO (the same PNU vertex shape, the two-attachment pixel
    /// shader, the lights' CBV at b1) and the composite pass's PSO (the
    /// fullscreen triangle, the config CBV at b0, the two HDR SRVs at
    /// t0-t1, the depth test off, one render target).
    fn ensure_post_resources(&mut self) -> bool {
        if self.post_gbuffer_ready && self.post_composite_ready {
            return true;
        }
        // ---- The post geometry pass's root signature + PSO ----
        if !self.post_gbuffer_ready {
            let mut rs_blob: D3dBlobHandle = std::ptr::null_mut();
            let mut error_blob: D3dBlobHandle = std::ptr::null_mut();
            let vs_source = super::shader_library::GBUFFER_VERTEX_SHADER;
            let ps_source = super::shader_library::POST_GBUFFER_PIXEL_SHADER;
            let mut vs_blob: D3dBlobHandle = std::ptr::null_mut();
            let mut ps_blob: D3dBlobHandle = std::ptr::null_mut();
            unsafe {
                let compile = self
                    .loader
                    .d3d_compile
                    .expect("the loader resolved D3DCompile");
                let hr = compile(
                    vs_source.as_ptr() as *const c_void,
                    vs_source.len() as u64,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    b"vsMain\0".as_ptr() as *const c_char,
                    b"vs_5_0\0".as_ptr() as *const c_char,
                    0,
                    0,
                    &mut vs_blob,
                    &mut error_blob,
                );
                if hr != S_OK || vs_blob.is_null() {
                    return false;
                }
                let hr = compile(
                    ps_source.as_ptr() as *const c_void,
                    ps_source.len() as u64,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    b"psMain\0".as_ptr() as *const c_char,
                    b"ps_5_0\0".as_ptr() as *const c_char,
                    0,
                    0,
                    &mut ps_blob,
                    &mut error_blob,
                );
                if hr != S_OK || ps_blob.is_null() {
                    return false;
                }
            }
            let vs_blob_size =
                unsafe { call_slot_u64_0(vs_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
            let ps_blob_size =
                unsafe { call_slot_u64_0(ps_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
            // The root signature: the same table the gbuffer PSO carries
            // (the frame's CBV at b0, the SRV at t0, the lights' CBV at
            // b1, the volume's SRV at t1 and the volume's CBV at b2, M8),
            // one static sampler at s0.
            let ranges = [
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                    num_descriptors: 1,
                    base_shader_register: 0,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 0,
                },
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    num_descriptors: 1,
                    base_shader_register: 0,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 1,
                },
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                    num_descriptors: 1,
                    base_shader_register: 1,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 2,
                },
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    num_descriptors: 1,
                    base_shader_register: 1,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 3,
                },
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                    num_descriptors: 1,
                    base_shader_register: 2,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 4,
                },
            ];
            let table = d3d::D3d12RootDescriptorTable {
                num_descriptor_ranges: 5,
                descriptor_ranges: ranges.as_ptr(),
            };
            let parameter = D3d12RootParameter {
                parameter_type: d3d::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                union_head_padding: [0],
                constants_or_table: d3d::D3d12RootParameterUnion { table },
                union_tail_padding: [0],
                visibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let static_sampler = d3d::D3d12StaticSamplerDesc {
                filter: d3d::D3D12_FILTER_MIN_MAG_MIP_POINT,
                address_u: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                address_v: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                address_w: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                mip_lod_bias: 0.0,
                max_anisotropy: 0,
                comparison_func: d3d::D3D12_COMPARISON_FUNC_ALWAYS,
                border_color: d3d::D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
                min_lod: 0.0,
                max_lod: 3.402823466e38,
                shader_register: 0,
                register_space: 0,
                shader_visibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let root_signature_desc = D3d12RootSignatureDesc {
                num_parameters: 1,
                parameters: &parameter,
                num_static_samplers: 1,
                static_samplers: &static_sampler as *const _ as *const c_void,
                flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            };
            unsafe {
                let serialize = self
                    .loader
                    .d3d12_serialize_root_signature
                    .expect("the loader resolved D3D12SerializeRootSignature");
                let hr = serialize(
                    &root_signature_desc,
                    d3d::D3D_ROOT_SIGNATURE_VERSION_1_0,
                    &mut rs_blob,
                    &mut error_blob,
                );
                if hr != S_OK || rs_blob.is_null() {
                    return false;
                }
                let rs_bytes = std::slice::from_raw_parts(
                    call_slot_ptr_0(rs_blob as ComObject, BLOB_GET_BUFFER_POINTER) as *const u8,
                    call_slot_u64_0(rs_blob as ComObject, BLOB_GET_BUFFER_SIZE) as usize,
                );
                let mut root_signature: *mut c_void = std::ptr::null_mut();
                let hr = call_slot_i32_5(
                    self.device as ComObject,
                    DEVICE_CREATE_ROOT_SIGNATURE,
                    0 as *mut c_void,
                    rs_bytes.as_ptr() as *const c_void as *mut c_void,
                    rs_bytes.len() as *mut c_void,
                    IID_ID3D12_ROOT_SIGNATURE_WORDS.as_ptr() as *const c_void as *mut c_void,
                    &mut root_signature as *mut _ as *mut c_void,
                );
                if hr != S_OK || root_signature.is_null() {
                    return false;
                }
                self.post_gbuffer_root_signature = root_signature;
            }
            // The PSO: the same states the gbuffer PSO carries, the
            // two-attachment pixel shader's bytecode, NumRenderTargets 2
            // with the HDR target's formats.
            let position = b"POSITION\0";
            let normal = b"NORMAL\0";
            let texcoord = b"TEXCOORD\0";
            let input_elements = [
                D3d12InputElementDesc {
                    semantic_name: position.as_ptr() as *const c_char,
                    semantic_index: 0,
                    format: DXGI_FORMAT_R32G32B32_FLOAT,
                    input_slot: 0,
                    aligned_byte_offset: 0,
                    input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    instance_data_step_rate: 0,
                },
                D3d12InputElementDesc {
                    semantic_name: normal.as_ptr() as *const c_char,
                    semantic_index: 0,
                    format: DXGI_FORMAT_R32G32B32_FLOAT,
                    input_slot: 0,
                    aligned_byte_offset: 12,
                    input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    instance_data_step_rate: 0,
                },
                D3d12InputElementDesc {
                    semantic_name: texcoord.as_ptr() as *const c_char,
                    semantic_index: 0,
                    format: DXGI_FORMAT_R32G32_FLOAT,
                    input_slot: 0,
                    aligned_byte_offset: 24,
                    input_slot_class: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    instance_data_step_rate: 0,
                },
            ];
            let blend_target = D3d12RenderTargetBlendDesc {
                blend_enable: 0,
                logic_op_enable: 0,
                src_blend: D3D12_BLEND_ONE,
                dest_blend: D3D12_BLEND_ZERO,
                blend_op: D3D12_BLEND_OP_ADD,
                src_blend_alpha: D3D12_BLEND_ONE,
                dest_blend_alpha: D3D12_BLEND_ZERO,
                blend_op_alpha: D3D12_BLEND_OP_ADD,
                logic_op: 0,
                render_target_write_mask: D3D12_COLOR_WRITE_ENABLE_ALL,
            };
            let blend_state = D3d12BlendDesc {
                alpha_to_coverage_enable: 0,
                independent_blend_enable: 0,
                render_target: [blend_target; 8],
            };
            let rasterizer = D3d12RasterizerDesc {
                fill_mode: D3D12_FILL_MODE_SOLID,
                cull_mode: D3D12_CULL_MODE_NONE,
                front_counter_clockwise: 0,
                depth_bias: 0,
                depth_bias_clamp: 0.0,
                slope_scaled_depth_bias: 0.0,
                depth_clip_enable: 1,
                multisample_enable: 0,
                antialiased_line_enable: 0,
                forced_sample_count: 0,
                conservative_raster: 0,
            };
            let depth_stencil = D3d12DepthStencilDesc {
                depth_enable: 1,
                depth_write_mask: D3D12_DEPTH_WRITE_MASK_ALL,
                depth_func: D3D12_COMPARISON_FUNC_LESS,
                stencil_enable: 0,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                front_face: D3d12DepthStencilopDesc {
                    stencil_fail_op: D3D12_STENCIL_OP_KEEP,
                    stencil_depth_fail_op: D3D12_STENCIL_OP_KEEP,
                    stencil_pass_op: D3D12_STENCIL_OP_KEEP,
                    stencil_func: D3D12_COMPARISON_FUNC_ALWAYS,
                },
                back_face: D3d12DepthStencilopDesc {
                    stencil_fail_op: D3D12_STENCIL_OP_KEEP,
                    stencil_depth_fail_op: D3D12_STENCIL_OP_KEEP,
                    stencil_pass_op: D3D12_STENCIL_OP_KEEP,
                    stencil_func: D3D12_COMPARISON_FUNC_ALWAYS,
                },
            };
            let input_layout = D3d12InputLayoutDesc {
                num_elements: 3,
                elements: input_elements.as_ptr(),
            };
            let (vs_bytecode, ps_bytecode) = unsafe {
                (
                    call_slot_ptr_0(vs_blob as ComObject, BLOB_GET_BUFFER_POINTER),
                    call_slot_ptr_0(ps_blob as ComObject, BLOB_GET_BUFFER_POINTER),
                )
            };
            let pso_desc = D3d12GraphicsPipelineStateDesc {
                root_signature: self.post_gbuffer_root_signature,
                vs: D3d12ShaderBytecode {
                    bytecode: vs_bytecode,
                    length: vs_blob_size,
                },
                ps: D3d12ShaderBytecode {
                    bytecode: ps_bytecode,
                    length: ps_blob_size,
                },
                ds: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                hs: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                gs: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                stream_output: D3d12StreamOutputDesc {
                    so_declaration: std::ptr::null(),
                    num_entries: 0,
                    head_padding: [0],
                    buffer_strides: std::ptr::null(),
                    num_strides: 0,
                    rasterized_stream: 0,
                },
                blend_state,
                sample_mask: 0xffffffff,
                rasterizer_state: rasterizer,
                depth_stencil_state: depth_stencil,
                input_layout,
                ib_strip_cut_value: 0,
                primitive_topology_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                num_render_targets: 2,
                rtv_formats: [
                    DXGI_FORMAT_R16G16B16A16_FLOAT,
                    DXGI_FORMAT_R16G16B16A16_FLOAT,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dsv_format: DXGI_FORMAT_D24_UNORM_S8_UINT,
                sample_desc: D3d12SampleDesc {
                    count: 1,
                    quality: 0,
                },
                node_mask: 0,
                cached_pso: D3d12CachedPipelineState {
                    cached_blob: std::ptr::null_mut(),
                    cached_blob_size: 0,
                },
                flags: 0,
            };
            let mut pipeline: *mut c_void = std::ptr::null_mut();
            unsafe {
                let hr = call_slot_i32_4(
                    self.device as ComObject,
                    DEVICE_CREATE_GRAPHICS_PIPELINE_STATE,
                    &pso_desc as *const D3d12GraphicsPipelineStateDesc as *mut c_void,
                    IID_ID3D12_PIPELINE_STATE_WORDS.as_ptr() as *const c_void as *mut c_void,
                    &mut pipeline as *mut _ as *mut c_void,
                    std::ptr::null_mut(),
                );
                if hr != S_OK || pipeline.is_null() {
                    return false;
                }
            }
            self.post_gbuffer_pipeline = pipeline;
            self.post_gbuffer_ready = true;
        }
        // ---- The composite pass's root signature + PSO ----
        if !self.post_composite_ready {
            let mut rs_blob: D3dBlobHandle = std::ptr::null_mut();
            let mut error_blob: D3dBlobHandle = std::ptr::null_mut();
            let vs_source = super::shader_library::LIGHTING_VERTEX_SHADER;
            let ps_source = super::shader_library::POST_COMPOSITE_PIXEL_SHADER;
            let mut vs_blob: D3dBlobHandle = std::ptr::null_mut();
            let mut ps_blob: D3dBlobHandle = std::ptr::null_mut();
            unsafe {
                let compile = self
                    .loader
                    .d3d_compile
                    .expect("the loader resolved D3DCompile");
                let hr = compile(
                    vs_source.as_ptr() as *const c_void,
                    vs_source.len() as u64,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    b"vsMain\0".as_ptr() as *const c_char,
                    b"vs_5_0\0".as_ptr() as *const c_char,
                    0,
                    0,
                    &mut vs_blob,
                    &mut error_blob,
                );
                if hr != S_OK || vs_blob.is_null() {
                    return false;
                }
                let hr = compile(
                    ps_source.as_ptr() as *const c_void,
                    ps_source.len() as u64,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    b"psMain\0".as_ptr() as *const c_char,
                    b"ps_5_0\0".as_ptr() as *const c_char,
                    0,
                    0,
                    &mut ps_blob,
                    &mut error_blob,
                );
                if hr != S_OK || ps_blob.is_null() {
                    return false;
                }
            }
            let vs_blob_size =
                unsafe { call_slot_u64_0(vs_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
            let ps_blob_size =
                unsafe { call_slot_u64_0(ps_blob as ComObject, BLOB_GET_BUFFER_SIZE) };
            // The root signature: the config CBV at b0 and the two HDR
            // SRVs at t0-t1 (the descriptor table's slots), one static
            // sampler at s0. No input assembler (the fullscreen
            // triangle's vertex ID generates the corners).
            let ranges = [
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    num_descriptors: 2,
                    base_shader_register: 0,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 0,
                },
                d3d::D3d12DescriptorRange {
                    range_type: d3d::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                    num_descriptors: 1,
                    base_shader_register: 0,
                    register_space: 0,
                    offset_in_descriptors_from_table_start: 2,
                },
            ];
            let table = d3d::D3d12RootDescriptorTable {
                num_descriptor_ranges: 2,
                descriptor_ranges: ranges.as_ptr(),
            };
            let parameter = D3d12RootParameter {
                parameter_type: d3d::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                union_head_padding: [0],
                constants_or_table: d3d::D3d12RootParameterUnion { table },
                union_tail_padding: [0],
                visibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let static_sampler = d3d::D3d12StaticSamplerDesc {
                filter: d3d::D3D12_FILTER_MIN_MAG_MIP_POINT,
                address_u: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                address_v: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                address_w: d3d::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                mip_lod_bias: 0.0,
                max_anisotropy: 0,
                comparison_func: d3d::D3D12_COMPARISON_FUNC_ALWAYS,
                border_color: d3d::D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
                min_lod: 0.0,
                max_lod: 3.402823466e38,
                shader_register: 0,
                register_space: 0,
                shader_visibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let root_signature_desc = D3d12RootSignatureDesc {
                num_parameters: 1,
                parameters: &parameter,
                num_static_samplers: 1,
                static_samplers: &static_sampler as *const _ as *const c_void,
                flags: 0,
            };
            unsafe {
                let serialize = self
                    .loader
                    .d3d12_serialize_root_signature
                    .expect("the loader resolved D3D12SerializeRootSignature");
                let hr = serialize(
                    &root_signature_desc,
                    d3d::D3D_ROOT_SIGNATURE_VERSION_1_0,
                    &mut rs_blob,
                    &mut error_blob,
                );
                if hr != S_OK || rs_blob.is_null() {
                    return false;
                }
                let rs_bytes = std::slice::from_raw_parts(
                    call_slot_ptr_0(rs_blob as ComObject, BLOB_GET_BUFFER_POINTER) as *const u8,
                    call_slot_u64_0(rs_blob as ComObject, BLOB_GET_BUFFER_SIZE) as usize,
                );
                let mut root_signature: *mut c_void = std::ptr::null_mut();
                let hr = call_slot_i32_5(
                    self.device as ComObject,
                    DEVICE_CREATE_ROOT_SIGNATURE,
                    0 as *mut c_void,
                    rs_bytes.as_ptr() as *const c_void as *mut c_void,
                    rs_bytes.len() as *mut c_void,
                    IID_ID3D12_ROOT_SIGNATURE_WORDS.as_ptr() as *const c_void as *mut c_void,
                    &mut root_signature as *mut _ as *mut c_void,
                );
                if hr != S_OK || root_signature.is_null() {
                    return false;
                }
                self.post_composite_root_signature = root_signature;
            }
            // The PSO: no input layout, the depth test off (the
            // fullscreen triangle covers the frame), one render target.
            let blend_target = D3d12RenderTargetBlendDesc {
                blend_enable: 0,
                logic_op_enable: 0,
                src_blend: D3D12_BLEND_ONE,
                dest_blend: D3D12_BLEND_ZERO,
                blend_op: D3D12_BLEND_OP_ADD,
                src_blend_alpha: D3D12_BLEND_ONE,
                dest_blend_alpha: D3D12_BLEND_ZERO,
                blend_op_alpha: D3D12_BLEND_OP_ADD,
                logic_op: 0,
                render_target_write_mask: D3D12_COLOR_WRITE_ENABLE_ALL,
            };
            let blend_state = D3d12BlendDesc {
                alpha_to_coverage_enable: 0,
                independent_blend_enable: 0,
                render_target: [blend_target; 8],
            };
            let rasterizer = D3d12RasterizerDesc {
                fill_mode: D3D12_FILL_MODE_SOLID,
                cull_mode: D3D12_CULL_MODE_NONE,
                front_counter_clockwise: 0,
                depth_bias: 0,
                depth_bias_clamp: 0.0,
                slope_scaled_depth_bias: 0.0,
                depth_clip_enable: 1,
                multisample_enable: 0,
                antialiased_line_enable: 0,
                forced_sample_count: 0,
                conservative_raster: 0,
            };
            let depth_stencil = D3d12DepthStencilDesc {
                depth_enable: 0,
                depth_write_mask: D3D12_DEPTH_WRITE_MASK_ZERO,
                depth_func: D3D12_COMPARISON_FUNC_LESS,
                stencil_enable: 0,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                front_face: D3d12DepthStencilopDesc {
                    stencil_fail_op: D3D12_STENCIL_OP_KEEP,
                    stencil_depth_fail_op: D3D12_STENCIL_OP_KEEP,
                    stencil_pass_op: D3D12_STENCIL_OP_KEEP,
                    stencil_func: D3D12_COMPARISON_FUNC_ALWAYS,
                },
                back_face: D3d12DepthStencilopDesc {
                    stencil_fail_op: D3D12_STENCIL_OP_KEEP,
                    stencil_depth_fail_op: D3D12_STENCIL_OP_KEEP,
                    stencil_pass_op: D3D12_STENCIL_OP_KEEP,
                    stencil_func: D3D12_COMPARISON_FUNC_ALWAYS,
                },
            };
            let input_layout = D3d12InputLayoutDesc {
                num_elements: 0,
                elements: std::ptr::null(),
            };
            let (vs_bytecode, ps_bytecode) = unsafe {
                (
                    call_slot_ptr_0(vs_blob as ComObject, BLOB_GET_BUFFER_POINTER),
                    call_slot_ptr_0(ps_blob as ComObject, BLOB_GET_BUFFER_POINTER),
                )
            };
            let pso_desc = D3d12GraphicsPipelineStateDesc {
                root_signature: self.post_composite_root_signature,
                vs: D3d12ShaderBytecode {
                    bytecode: vs_bytecode,
                    length: vs_blob_size,
                },
                ps: D3d12ShaderBytecode {
                    bytecode: ps_bytecode,
                    length: ps_blob_size,
                },
                ds: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                hs: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                gs: D3d12ShaderBytecode {
                    bytecode: std::ptr::null(),
                    length: 0,
                },
                stream_output: D3d12StreamOutputDesc {
                    so_declaration: std::ptr::null(),
                    num_entries: 0,
                    head_padding: [0],
                    buffer_strides: std::ptr::null(),
                    num_strides: 0,
                    rasterized_stream: 0,
                },
                blend_state,
                sample_mask: 0xffffffff,
                rasterizer_state: rasterizer,
                depth_stencil_state: depth_stencil,
                input_layout,
                ib_strip_cut_value: 0,
                primitive_topology_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                num_render_targets: 1,
                rtv_formats: [DXGI_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 0, 0, 0],
                dsv_format: DXGI_FORMAT_D24_UNORM_S8_UINT,
                sample_desc: D3d12SampleDesc {
                    count: 1,
                    quality: 0,
                },
                node_mask: 0,
                cached_pso: D3d12CachedPipelineState {
                    cached_blob: std::ptr::null_mut(),
                    cached_blob_size: 0,
                },
                flags: 0,
            };
            let mut pipeline: *mut c_void = std::ptr::null_mut();
            unsafe {
                let hr = call_slot_i32_4(
                    self.device as ComObject,
                    DEVICE_CREATE_GRAPHICS_PIPELINE_STATE,
                    &pso_desc as *const D3d12GraphicsPipelineStateDesc as *mut c_void,
                    IID_ID3D12_PIPELINE_STATE_WORDS.as_ptr() as *mut c_void,
                    &mut pipeline as *mut _ as *mut c_void,
                                    std::ptr::null_mut(),
                );
                if hr != S_OK || pipeline.is_null() {
                    return false;
                }
            }
            self.post_composite_pipeline = pipeline;
            // The config CBV's buffer (the same 256-byte UPLOAD-heap
            // buffer the lights' CBV reads).
            if self.post_config_buffer.is_null() {
                match unsafe {
                    self_create_buffer(
                        self.device,
                        256,
                        D3D12_HEAP_TYPE_UPLOAD,
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                    )
                } {
                    Ok(buffer) => self.post_config_buffer = buffer,
                    Err(_) => return false,
                }
            }
            // The post chain's own shader-visible heap: four descriptors
            // (the two HDR SRVs at slots 0-1, the config CBV at slot 2).
            if self.post_srv_heap.is_null() {
                match unsafe {
                    self_create_descriptor_heap(
                        self.device,
                        d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        4,
                        d3d::D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    )
                } {
                    Ok(heap) => self.post_srv_heap = heap,
                    Err(_) => return false,
                }
            }
            self.post_composite_ready = true;
        }
        true
    }

    /// Builds the triangle's uniform block bytes (the MVP column-major
    /// followed by the colour).
    fn build_tri_uniform(&self) -> TriUniform {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let projection = Mat4::perspective(std::f32::consts::FRAC_PI_2, aspect, 0.1, 100.0);
        let view = Mat4::look_at(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let mvp = projection.multiply(view);
        let colour = self.config.triangle_color;
        TriUniform {
            mvp: mvp.m,
            color: [
                colour.r as f32 / 255.0,
                colour.g as f32 / 255.0,
                colour.b as f32 / 255.0,
                colour.a as f32 / 255.0,
            ],
        }
    }

    /// The diagnostic triangle's draw: the frame's command list carries
    /// the pipeline, the root signature, the 20-word constants block and
    /// the three positions.
    fn draw_triangle(&mut self) {
        if self.shutdown || !self.tri_ready {
            return;
        }
        let uniform = self.build_tri_uniform();
        let uniform_words =
            unsafe { std::slice::from_raw_parts(&uniform as *const TriUniform as *const u32, 20) };
        let viewport = D3d12Viewport {
            top_left_x: 0.0,
            top_left_y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = D3d12Rect {
            left: 0,
            top: 0,
            right: self.config.width as i32,
            bottom: self.config.height as i32,
        };
        let vertex_buffer_view = D3d12VertexBufferView {
            buffer_location: unsafe {
                call_slot_u64_0(
                    self.tri_vertex_buffer as ComObject,
                    RESOURCE_GET_GPU_VIRTUAL_ADDRESS,
                )
            },
            size_in_bytes: 36,
            stride_in_bytes: 12,
        };
        unsafe {
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_PIPELINE_STATE,
                self.tri_pipeline,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_VIEWPORTS,
                1 as *mut c_void,
                &viewport as *const D3d12Viewport as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_SCISSOR_RECTS,
                1 as *mut c_void,
                &scissor as *const D3d12Rect as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_OMSET_RENDER_TARGETS,
                1 as *mut c_void,
                &self.rtv_handle as *const D3d12CpuDescriptorHandle as *mut c_void,
                0 as *mut c_void,
                &self.dsv_handle as *const D3d12CpuDescriptorHandle as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_SIGNATURE,
                self.tri_root_signature,
                std::ptr::null_mut(),
            );
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_32BIT_CONSTANTS,
                0 as *mut c_void,
                20 as *mut c_void,
                uniform_words.as_ptr() as *const c_void as *mut c_void,
                0 as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_VERTEX_BUFFERS,
                0 as *mut c_void,
                1 as *mut c_void,
                &vertex_buffer_view as *const D3d12VertexBufferView as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_PRIMITIVE_TOPOLOGY,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_DRAW_INSTANCED,
                3 as *mut c_void,
                1 as *mut c_void,
                std::ptr::null_mut(),
            );
        }
    }

    /// The frame's clear: the colour and the depth/stencil clears are
    /// recorded into the command list (the load actions the other
    /// backends' clear carries).
    fn clear_frame(&mut self) {
        // The readback's copies left the resources in the COPY_SOURCE
        // state; the second frame's barriers bring them back into the
        // render states before the clears (the two-frame probe's rule,
        // with the all-subresources transition).
        let colour_return = D3d12ResourceBarrier {
            type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            transition: D3d12ResourceTransitionBarrier {
                resource: self.color_texture,
                subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                state_before: D3D12_RESOURCE_STATE_COPY_SOURCE,
                state_after: D3D12_RESOURCE_STATE_RENDER_TARGET,
            },
        };
        let depth_return = D3d12ResourceBarrier {
            type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            transition: D3d12ResourceTransitionBarrier {
                resource: self.depth_texture,
                subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                state_before: D3D12_RESOURCE_STATE_COPY_SOURCE,
                state_after: D3D12_RESOURCE_STATE_DEPTH_WRITE,
            },
        };
        if self.resources_in_copy_source {
            unsafe {
                // The probe C's frame-2 rule: one barrier per call, the
                // two back-transitions recorded separately (only after a
                // readback actually moved the resources).
                call_slot_i32_3(
                    self.command_list as ComObject,
                    COMMAND_LIST_RESOURCE_BARRIER,
                    1 as *mut c_void,
                    &colour_return as *const D3d12ResourceBarrier as *mut c_void,
                    std::ptr::null_mut(),
                );
                call_slot_i32_3(
                    self.command_list as ComObject,
                    COMMAND_LIST_RESOURCE_BARRIER,
                    1 as *mut c_void,
                    &depth_return as *const D3d12ResourceBarrier as *mut c_void,
                    std::ptr::null_mut(),
                );
            }
            self.resources_in_copy_source = false;
        }
        let clear_colour = [
            self.clear_color.r as f32 / 255.0,
            self.clear_color.g as f32 / 255.0,
            self.clear_color.b as f32 / 255.0,
            self.clear_color.a as f32 / 255.0,
        ];
        let no_rect = D3d12Rect {
            left: 0,
            top: 0,
            right: self.config.width as i32,
            bottom: self.config.height as i32,
        };
        unsafe {
            call_slot_i32_4(
                self.command_list as ComObject,
                COMMAND_LIST_CLEAR_RENDER_TARGET_VIEW,
                // The clear's descriptor handle travels by value: the
                // probe C's ClearRenderTargetView record carries the
                // handle itself, and the Wine run's vkd3d read the
                // stack garbage the pointer form left in the slot.
                self.rtv_handle.ptr as *mut c_void,
                clear_colour.as_ptr() as *const c_void as *mut c_void,
                1 as *mut c_void,
                &no_rect as *const D3d12Rect as *mut c_void,
            );
            call_slot_i32_f32(
                self.command_list as ComObject,
                COMMAND_LIST_CLEAR_DEPTH_STENCIL_VIEW,
                self.dsv_handle.ptr as *mut c_void,
                (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL) as u32,
                self.clear_depth as f32,
                self.clear_stencil as u32,
                1,
                &no_rect as *const D3d12Rect as *mut c_void,
            );
        }
    }

    /// The readback's one pixel: the three plane copies (the colour plane,
    /// the depth plane's words, the stencil plane's bytes) are recorded
    /// into the command list before the Close, the frame executes, and the
    /// mapped staging buffer carries the pixel.
    fn readback_pixel(&mut self, x: u32, y: u32) {
        if self.color_texture.is_null() || self.depth_texture.is_null() {
            return;
        }
        // The readback's copies need an open command list: a previous
        // readback's execute closed the list (the same rule the fill's
        // one-off frame follows).
        if !self.frame_open {
            let hr = unsafe { call_slot_i32_0(self.allocator as ComObject, ALLOCATOR_RESET) };
            if hr != S_OK {
                return;
            }
            let hr = unsafe {
                call_slot_i32_2(
                    self.command_list as ComObject,
                    COMMAND_LIST_RESET,
                    self.allocator,
                    self.tri_pipeline,
                )
            };
            if hr != S_OK {
                return;
            }
        }
        let colour_pitch = d3d::colour_row_pitch(self.config.width);
        let stencil_pitch = d3d::stencil_row_pitch(self.config.width);
        let colour_footprint = D3d12PlacedFootprint {
            offset: 0,
            footprint: D3d12SubresourceFootprint {
                format: DXGI_FORMAT_R8G8B8A8_UNORM,
                width: self.config.width,
                height: self.config.height,
                depth: 1,
                row_pitch: colour_pitch,
            },
        };
        let depth_footprint = D3d12PlacedFootprint {
            offset: STAGING_COLOUR_BYTES as u64,
            footprint: D3d12SubresourceFootprint {
                format: DXGI_FORMAT_R32_FLOAT,
                width: self.config.width,
                height: self.config.height,
                depth: 1,
                row_pitch: colour_pitch,
            },
        };
        let stencil_footprint = D3d12PlacedFootprint {
            offset: STAGING_STENCIL_OFFSET as u64,
            footprint: D3d12SubresourceFootprint {
                format: DXGI_FORMAT_R8_UINT,
                width: self.config.width,
                height: self.config.height,
                depth: 1,
                row_pitch: stencil_pitch,
            },
        };
        let colour_dst = D3d12TextureCopyLocation {
            resource: self.staging_buffer,
            type_: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            placed_footprint: colour_footprint,
        };
        let depth_dst = D3d12TextureCopyLocation {
            resource: self.staging_buffer,
            type_: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            placed_footprint: depth_footprint,
        };
        let stencil_dst = D3d12TextureCopyLocation {
            resource: self.staging_buffer,
            type_: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            placed_footprint: stencil_footprint,
        };
        let colour_src = D3d12TextureCopyLocation {
            resource: self.color_texture,
            type_: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            placed_footprint: D3d12PlacedFootprint {
                offset: 0,
                footprint: D3d12SubresourceFootprint {
                    format: 0,
                    width: 0,
                    height: 0,
                    depth: 0,
                    row_pitch: 0,
                },
            },
        };
        let depth_src = D3d12TextureCopyLocation {
            resource: self.depth_texture,
            type_: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            placed_footprint: D3d12PlacedFootprint {
                offset: 0,
                footprint: D3d12SubresourceFootprint {
                    format: 0,
                    width: 0,
                    height: 0,
                    depth: 0,
                    row_pitch: 0,
                },
            },
        };
        let stencil_src = D3d12TextureCopyLocation {
            resource: self.depth_texture,
            type_: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            placed_footprint: D3d12PlacedFootprint {
                // The union's first word is the subresource index: the
                // stencil plane reads as subresource 1 (the probe C's
                // rule), and the offset's low word carries it.
                offset: 1,
                footprint: D3d12SubresourceFootprint {
                    format: 0,
                    width: 0,
                    height: 0,
                    depth: 0,
                    row_pitch: 0,
                },
            },
        };
        let colour_barrier = D3d12ResourceBarrier {
            type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            transition: D3d12ResourceTransitionBarrier {
                resource: self.color_texture,
                subresource: 0,
                state_before: D3D12_RESOURCE_STATE_RENDER_TARGET,
                state_after: D3D12_RESOURCE_STATE_COPY_SOURCE,
            },
        };
        let depth_barrier = D3d12ResourceBarrier {
            type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            transition: D3d12ResourceTransitionBarrier {
                resource: self.depth_texture,
                subresource: 0,
                state_before: D3D12_RESOURCE_STATE_DEPTH_WRITE,
                state_after: D3D12_RESOURCE_STATE_COPY_SOURCE,
            },
        };
        unsafe {
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RESOURCE_BARRIER,
                1 as *mut c_void,
                &colour_barrier as *const D3d12ResourceBarrier as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_6(
                self.command_list as ComObject,
                COMMAND_LIST_COPY_TEXTURE_REGION,
                &colour_dst as *const D3d12TextureCopyLocation as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                &colour_src as *const D3d12TextureCopyLocation as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RESOURCE_BARRIER,
                1 as *mut c_void,
                &depth_barrier as *const D3d12ResourceBarrier as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_6(
                self.command_list as ComObject,
                COMMAND_LIST_COPY_TEXTURE_REGION,
                &depth_dst as *const D3d12TextureCopyLocation as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                &depth_src as *const D3d12TextureCopyLocation as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_6(
                self.command_list as ComObject,
                COMMAND_LIST_COPY_TEXTURE_REGION,
                &stencil_dst as *const D3d12TextureCopyLocation as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                &stencil_src as *const D3d12TextureCopyLocation as *mut c_void,
                std::ptr::null_mut(),
            );
        }
        // The copies left both resources in COPY_SOURCE: the next
        // clear_frame's return barriers bring them back (the probe C's
        // frame-2 rule).
        self.resources_in_copy_source = true;
        // The list closed: the next readback (or the next clear) must
        // reset it before recording (the same rule the fill's one-off
        // frame follows).
        self.frame_open = false;
        // The frame executes: the Close, the ExecuteCommandLists, the
        // queue's Signal and the wait (the GPU-side completion is what
        // the wait sees).
        let hr = unsafe { call_slot_i32_0(self.command_list as ComObject, COMMAND_LIST_CLOSE) };
        if hr != S_OK {
            return;
        }
        let lists = [self.command_list];
        unsafe {
            call_slot_i32_3(
                self.queue as ComObject,
                QUEUE_EXECUTE_COMMAND_LISTS,
                1 as *mut c_void,
                lists.as_ptr() as *const c_void as *mut c_void,
                std::ptr::null_mut(),
            );
            self.fence_value += 1;
            call_slot_i32_4(
                self.queue as ComObject,
                QUEUE_SIGNAL,
                self.fence,
                self.fence_value as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.fence as ComObject,
                FENCE_SET_EVENT_ON_COMPLETION,
                self.fence_value as *mut c_void,
                self.fence_event,
                std::ptr::null_mut(),
            );
            self.loader
                .wait_for_single_object
                .expect("the loader resolved WaitForSingleObject")(
                self.fence_event, 5000
            );
        }
        // The mapped staging buffer carries the three planes.
        let mut map: *mut c_void = std::ptr::null_mut();
        let mapped = unsafe {
            call_slot_i32_4(
                self.staging_buffer as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range {
                    begin: 0,
                    end: STAGING_SIZE as u64,
                } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        if mapped != S_OK || map.is_null() {
            return;
        }
        let map = map as *mut u8;
        let cx = x as usize;
        let cy = y as usize;
        let colour = unsafe {
            let row = map.add(cy * colour_pitch as usize);
            let p = row.add(cx * 4);
            ClearColor {
                r: *p,
                g: *p.add(1),
                b: *p.add(2),
                a: *p.add(3),
            }
        };
        let depth_word = unsafe {
            let words = map.add(STAGING_DEPTH_OFFSET) as *const u32;
            let row_words = colour_pitch as usize / 4;
            *words.add(cy * row_words + cx)
        };
        let depth = (depth_word & 0x00ff_ffff) as f32 / 16_777_215.0;
        let stencil = unsafe {
            let bytes = map.add(STAGING_STENCIL_OFFSET);
            let row = bytes.add(cy * stencil_pitch as usize);
            *row.add(cx) as i32
        };
        unsafe {
            call_slot_i32_3(
                self.staging_buffer as ComObject,
                RESOURCE_UNMAP,
                0 as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
        }
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

    /// The generic draw path: the mesh's buffers, the camera's
    /// view-projection and the model matrix (the root table's CBV), the
    /// bound texture's SRV, and the indexed draw. A draw with no open
    /// frame counts one violation (the same rule the other backends
    /// follow).

    pub(crate) fn draw_mesh_by_id(&mut self, mesh_id: usize, model: Mat4) {
        self.draw_mesh_inner(mesh_id, model, false);
    }

    /// The lit draw (M5): the same path, the lit PSO and the lights'
    /// CBV at b1 (the submit path carries the lights).
    fn draw_mesh_lit_by_id(&mut self, mesh_id: usize, model: Mat4) {
        self.draw_mesh_inner(mesh_id, model, true);
    }

    /// The mesh draw's shared body: the unlit path binds the mesh PSO
    /// and the mesh's two-descriptor table; the lit path binds the lit
    /// PSO, writes the lights' block into the shared buffer and points
    /// the table's third descriptor (the lights' CBV at b1) at it.
    fn draw_mesh_inner(&mut self, mesh_id: usize, model: Mat4, lit: bool) {
        if !self.frame_open {
            self.violations += 1;
            return;
        }
        if mesh_id >= self.meshes.len() {
            self.violations += 1;
            return;
        }
        if !self.mesh_ready {
            return;
        }
        if lit && !self.lit_ready {
            return;
        }
        let camera = self.camera;
        let aspect = self.config.width as f32 / self.config.height as f32;
        let view = Mat4::look_at(camera.eye, camera.target, camera.up);
        let projection = Mat4::perspective(camera.fov_y_rad, aspect, camera.near, camera.far);
        let view_projection = projection.multiply(view);
        // The uniform block: the two column-major matrices (the same
        // bytes the Vulkan block carries; the shader's column_major
        // qualifier reads them as-is).
        // The 256-byte CBV alignment: the two matrices live in the
        // first 128 bytes, the rest zero.
        let mut uniform = [0u8; 256];
        uniform[0..64].copy_from_slice(unsafe {
            std::slice::from_raw_parts(view_projection.m.as_ptr() as *const u8, 64)
        });
        uniform[64..128].copy_from_slice(unsafe {
            std::slice::from_raw_parts(model.m.as_ptr() as *const u8, 64)
        });
        // The lights' block bytes (the lit path writes them into the
        // shared buffer; the block the frame's draws read).
        let lights_bytes = if lit {
            let block = self.lit_block;
            let mut bytes = [0u8; 256];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &block as *const LightsUniform as *const u8,
                    bytes.as_mut_ptr(),
                    256,
                );
            }
            Some(bytes)
        } else {
            None
        };
        let state = self.meshes[mesh_id].as_ref().unwrap();
        let uniform_address = unsafe {
            call_slot_u64_0(
                state.uniform_buffer as ComObject,
                RESOURCE_GET_GPU_VIRTUAL_ADDRESS,
            )
        };
        // The uniform buffer's write through the Map/Unmap pair (the same
        // rule the upload follows).
        unsafe {
            let mut map: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_4(
                state.uniform_buffer as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr == S_OK && !map.is_null() {
                std::ptr::copy_nonoverlapping(uniform.as_ptr(), map as *mut u8, 256);
                call_slot_i32_2(
                    state.uniform_buffer as ComObject,
                    RESOURCE_UNMAP,
                    0 as *mut c_void,
                    std::ptr::null_mut(),
                );
            }
        }
        // The SRV: the bound texture's resource (the default texture's
        // resource when no texture is bound).
        let texture_resource = self
            .bound_texture
            .and_then(|i| self.textures_2d.get(i))
            .filter(|t| !t.released)
            .map(|t| t.resource)
            .or_else(|| {
                self.default_texture
                    .and_then(|i| self.textures_2d.get(i))
                    .filter(|t| !t.released)
                    .map(|t| t.resource)
            });
        let texture_resource = texture_resource.unwrap_or(std::ptr::null_mut());
        // The lights' CBV (the lit path): the shared buffer's address
        // (the descriptor the table's slot 2 reads). The block's write
        // runs through the Map/Unmap pair (the same rule the upload
        // follows).
        let lights_address = if lit {
            unsafe {
                let mut map: *mut c_void = std::ptr::null_mut();
                let hr = call_slot_i32_4(
                    self.lit_lights_buffer as ComObject,
                    RESOURCE_MAP,
                    0 as *mut c_void,
                    &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                    &mut map as *mut _ as *mut c_void,
                    std::ptr::null_mut(),
                );
                if hr == S_OK && !map.is_null() {
                    std::ptr::copy_nonoverlapping(
                        lights_bytes.as_ref().unwrap().as_ptr(),
                        map as *mut u8,
                        256,
                    );
                    call_slot_i32_2(
                        self.lit_lights_buffer as ComObject,
                        RESOURCE_UNMAP,
                        0 as *mut c_void,
                        std::ptr::null_mut(),
                    );
                }
                call_slot_u64_0(
                    self.lit_lights_buffer as ComObject,
                    RESOURCE_GET_GPU_VIRTUAL_ADDRESS,
                )
            }
        } else {
            0
        };
        let srv_desc = d3d::D3d12ShaderResourceViewDesc {
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            view_dimension: d3d::D3D12_SRV_DIMENSION_TEXTURE2D,
            shader_4_component_mapping: d3d::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            header_padding: [0],
            texture_2d: d3d::D3d12Tex2DSrv {
                most_detailed_mip: 0,
                mip_levels: 1,
                plane_slice: 0,
                resource_min_lod_clamp: 0.0,
            },
            union_tail_padding: [0, 0],
        };
        let cbv_desc = d3d::D3d12ConstantBufferViewDesc {
            buffer_location: uniform_address,
            size_in_bytes: 256,
            union_tail_padding: [0],
        };
        let lights_cbv_desc = d3d::D3d12ConstantBufferViewDesc {
            buffer_location: lights_address,
            size_in_bytes: 256,
            union_tail_padding: [0],
        };
        let vertex_buffer_view = D3d12VertexBufferView {
            buffer_location: unsafe {
                call_slot_u64_0(
                    state.vertex_buffer as ComObject,
                    RESOURCE_GET_GPU_VIRTUAL_ADDRESS,
                )
            },
            size_in_bytes: state.vertex_bytes,
            stride_in_bytes: 32,
        };
        let index_buffer_view = d3d::D3d12IndexBufferView {
            buffer_location: unsafe {
                call_slot_u64_0(
                    state.index_buffer as ComObject,
                    RESOURCE_GET_GPU_VIRTUAL_ADDRESS,
                )
            },
            size_in_bytes: state.index_count * 4,
            format: DXGI_FORMAT_R32_UINT,
        };
        let viewport = D3d12Viewport {
            top_left_x: 0.0,
            top_left_y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = D3d12Rect {
            left: 0,
            top: 0,
            right: self.config.width as i32,
            bottom: self.config.height as i32,
        };
        unsafe {
            // The CBV and the SRV: the draw's uniform bytes invalidate the
            // descriptor's address, so the Create calls re-record (the
            // same rule the Vulkan's descriptor write follows).
            call_slot_i32_2(
                self.device as ComObject,
                DEVICE_CREATE_CONSTANT_BUFFER_VIEW,
                &cbv_desc as *const d3d::D3d12ConstantBufferViewDesc as *mut c_void,
                self_heap_cpu_handle_offset(
                    state.srv_heap,
                    self.device,
                    d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    0,
                )
                .ptr as *mut c_void,
            );
            if lit {
                call_slot_i32_2(
                    self.device as ComObject,
                    DEVICE_CREATE_CONSTANT_BUFFER_VIEW,
                    &lights_cbv_desc as *const d3d::D3d12ConstantBufferViewDesc as *mut c_void,
                    self_heap_cpu_handle_offset(
                        state.srv_heap,
                        self.device,
                        d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        2,
                    )
                    .ptr as *mut c_void,
                );
            }
            call_slot_i32_3(
                self.device as ComObject,
                DEVICE_CREATE_SHADER_RESOURCE_VIEW,
                texture_resource,
                &srv_desc as *const d3d::D3d12ShaderResourceViewDesc as *mut c_void,
                self_heap_cpu_handle_offset(
                    state.srv_heap,
                    self.device,
                    d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    1,
                )
                .ptr as *mut c_void,
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_PIPELINE_STATE,
                if lit {
                    self.lit_pipeline
                } else {
                    self.mesh_pipeline
                },
                std::ptr::null_mut(),
            );
            // The frame's render targets: the same OMSetRenderTargets
            // call the triangle's draw records (the colour RTV and the
            // depth DSV, the draw's pixels land in them).
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_OMSET_RENDER_TARGETS,
                1 as *mut c_void,
                &self.rtv_handle as *const D3d12CpuDescriptorHandle as *mut c_void,
                0 as *mut c_void,
                &self.dsv_handle as *const D3d12CpuDescriptorHandle as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_SIGNATURE,
                if lit {
                    self.lit_root_signature
                } else {
                    self.mesh_root_signature
                },
                std::ptr::null_mut(),
            );
            let heaps = [state.srv_heap];
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_SET_DESCRIPTOR_HEAPS,
                1 as *mut c_void,
                heaps.as_ptr() as *const c_void as *mut c_void,
                std::ptr::null_mut(),
            );
            let gpu_handle = self_heap_gpu_handle_offset(
                state.srv_heap,
                self.device,
                d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                0,
            );
            call_slot_void_ptr_u64(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE,
                0 as *mut c_void,
                gpu_handle.ptr,
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_VIEWPORTS,
                1 as *mut c_void,
                &viewport as *const D3d12Viewport as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_SCISSOR_RECTS,
                1 as *mut c_void,
                &scissor as *const D3d12Rect as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_PRIMITIVE_TOPOLOGY,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_VERTEX_BUFFERS,
                0 as *mut c_void,
                1 as *mut c_void,
                &vertex_buffer_view as *const D3d12VertexBufferView as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_INDEX_BUFFER,
                &index_buffer_view as *const d3d::D3d12IndexBufferView as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_DRAW_INDEXED_INSTANCED,
                state.index_count as *mut c_void,
                1 as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                std::ptr::null_mut(),
            );
        }
    }
}

impl RenderBackend for D3d12Backend {
    fn describe(&self) -> String {
        format!(
            "d3d12 feature_level=0x{:x} vendor=0x{:04x} renderer={}",
            self.feature_level, self.adapter_vendor, self.adapter_desc
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
        // The frame's command list opens: the allocator and the list
        // reset, the clears and the diagnostic draw are recorded.
        let hr = unsafe { call_slot_i32_0(self.allocator as ComObject, ALLOCATOR_RESET) };
        if hr != S_OK {
            self.violations += 1;
            return;
        }
        let hr = unsafe {
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_RESET,
                self.allocator,
                self.tri_pipeline,
            )
        };
        if hr != S_OK {
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

    fn create_shader_program(&mut self, id: u32) -> Result<ShaderProgram, RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "create_shader_program: the backend is shut down",
            ));
        }
        Ok(ShaderProgram {
            inner: Arc::new(Mutex::new(ShaderProgramInner { id, compiled: true })),
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
            return Err(RenderError::from("upload_mesh: the backend is shut down"));
        }
        if layout.stride == 0 {
            return Err(RenderError::from(
                "upload_mesh: the layout's stride must be positive",
            ));
        }
        let mesh_id = mesh.inner.lock().unwrap().id;
        // The mesh path's pipeline: the same lazy ensure the triangle path
        // follows (one PSO serves every mesh).
        if !self.ensure_mesh_resources() {
            return Err(RenderError::from(
                "upload_mesh: the mesh's shader program failed to compile",
            ));
        }
        // The lit path's ensure (M5): the same before-the-frame rule (the
        // PSO's creation inside an open frame's recording is the
        // software rasterizer's crash the debug run caught).
        if !self.ensure_lit_resources() {
            return Err(RenderError::from(
                "upload_mesh: the lit shader program failed to compile",
            ));
        }
        // The lights' shared buffer (M5): one 256-byte UPLOAD-heap buffer
        // the lit draw's CBV at b1 reads (the block the D3D12 rule keeps
        // 256-byte aligned; one buffer serves every mesh's lit draws).
        if self.lit_lights_buffer.is_null() {
            self.lit_lights_buffer = unsafe {
                self_create_buffer(
                    self.device,
                    256,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )?
            };
        }
        // The two UPLOAD-heap buffers: the CPU writes them through the
        // Map/Unmap pair (the same rule the triangle's vertex buffer
        // follows). The vertex buffer carries the PNU bytes, the index
        // buffer the u32s.
        let vertex_buffer = unsafe {
            self_create_buffer(
                self.device,
                vertices.len() as u64,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?
        };
        let index_bytes = (indices.len() * std::mem::size_of::<u32>()) as u64;
        let index_buffer = unsafe {
            self_create_buffer(
                self.device,
                index_bytes,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?
        };
        unsafe {
            let mut map: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_4(
                vertex_buffer as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr != S_OK || map.is_null() {
                return Err(RenderError::from(
                    "upload_mesh: the vertex buffer's map failed",
                ));
            }
            std::ptr::copy_nonoverlapping(vertices.as_ptr(), map as *mut u8, vertices.len());
            call_slot_i32_2(
                vertex_buffer as ComObject,
                RESOURCE_UNMAP,
                0 as *mut c_void,
                std::ptr::null_mut(),
            );
            let mut map: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_4(
                index_buffer as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr != S_OK || map.is_null() {
                return Err(RenderError::from(
                    "upload_mesh: the index buffer's map failed",
                ));
            }
            std::ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                map as *mut u8,
                indices.len() * std::mem::size_of::<u32>(),
            );
            call_slot_i32_2(
                index_buffer as ComObject,
                RESOURCE_UNMAP,
                0 as *mut c_void,
                std::ptr::null_mut(),
            );
        }
        // The mesh's SRV heap: one CBV_SRV_UAV descriptor heap with two
        // descriptors the draw's root table reads (the CBV at slot 0,
        // the SRV at slot 1) — three when the lit path draws (the
        // lights' CBV at slot 2, M5) — five when the post chain's
        // gbuffer pass draws (the volume's SRV at slot 3 and the
        // volume's CBV at slot 4, M8).
        let srv_heap = unsafe {
            self_create_descriptor_heap(
                self.device,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                5,
                D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            )?
        };
        // The uniform buffer: the 256-byte UPLOAD-heap buffer the draw's
        // CBV reads (the two column-major matrices the camera's
        // view-projection and the model carry; the CBV's size must be
        // 256-byte aligned, the D3D12 rule).
        let uniform_buffer = unsafe {
            self_create_buffer(
                self.device,
                256,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?
        };
        let mut inner = mesh.inner.lock().unwrap();
        inner.vertex_count = (vertices.len() as u32) / layout.stride;
        inner.index_count = indices.len() as u32;
        inner.uploaded = true;
        drop(inner);
        let state = D3d12MeshState {
            vertex_buffer,
            index_buffer,
            vertex_bytes: vertices.len() as u32,
            index_count: indices.len() as u32,
            srv_heap,
            uniform_buffer,
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
        let mesh_id = submission.mesh.inner.lock().unwrap().id;
        let has_lights = submission.sun.is_some() || !submission.lights.is_empty();
        if has_lights {
            // The lit path (M5): the lights' block from the submission's
            // sun and point lights (the sun absent means intensity zero),
            // the lit PSO draws.
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

    /// The deferred lighting pass (M6): the two internal steps the
    /// contract's submit_lighting_pass carries. Step one (the geometry
    /// pass) sets the G-buffer's three RTVs, draws the mesh with the
    /// G-buffer PSO. Step two transitions the attachments into the
    /// pixel-shader-resource state, sets the default framebuffer's RTV,
    /// and draws the fullscreen triangle with the lighting PSO.
    fn submit_lighting_pass(&mut self, submission: &DrawSubmission, gbuffer: &RenderTarget) {
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
        if mesh_id >= self.meshes.len() || self.meshes[mesh_id].is_none() {
            self.violations += 1;
            return;
        }
        if !self.mesh_ready || !self.ensure_deferred_resources() {
            return;
        }
        if !self.frame_open {
            self.violations += 1;
            return;
        }
        // The lights' block: the same from_lights the forward path
        // carries.
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
        let projection = Mat4::perspective(camera.fov_y_rad, aspect, camera.near, camera.far);
        let view_projection = projection.multiply(view);
        let model = submission.model;
        // The G-buffer's camera uniform block: the same 256-byte CBV
        // alignment the mesh path's block carries.
        let mut uniform = [0u8; 256];
        uniform[0..64].copy_from_slice(unsafe {
            std::slice::from_raw_parts(view_projection.m.as_ptr() as *const u8, 64)
        });
        uniform[64..128].copy_from_slice(unsafe {
            std::slice::from_raw_parts(model.m.as_ptr() as *const u8, 64)
        });
        let (vertex_buffer, index_buffer, vertex_bytes, index_count, srv_heap, uniform_buffer) = {
            let state = self.meshes[mesh_id].as_ref().unwrap();
            (
                state.vertex_buffer,
                state.index_buffer,
                state.vertex_bytes,
                state.index_count,
                state.srv_heap,
                state.uniform_buffer,
            )
        };
        let texture_resource = self
            .bound_texture
            .and_then(|i| self.textures_2d.get(i))
            .filter(|t| !t.released)
            .map(|t| t.resource)
            .unwrap_or(
                self.default_texture
                    .and_then(|i| self.textures_2d.get(i))
                    .filter(|t| !t.released)
                    .map(|t| t.resource)
                    .unwrap_or(std::ptr::null_mut()),
            );
        let gbuffer_resources: Vec<d3d::D3d12ResourceHandle> = self.targets[target_index]
            .views
            .iter()
            .take(3)
            .map(|v| v.resource)
            .collect();
        if gbuffer_resources.len() != 3 {
            self.violations += 1;
            return;
        }
        // The G-buffer's RTV heap (one descriptor per attachment): the
        // creation happens once (the target's own heap field).
        if self.targets[target_index].rtv_heap.is_null() {
            match unsafe {
                self_create_descriptor_heap(
                    self.device,
                    D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    3,
                    D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                )
            } {
                Ok(heap) => self.targets[target_index].rtv_heap = heap,
                Err(_) => {
                    self.violations += 1;
                    return;
                }
            }
            for (i, resource) in gbuffer_resources.iter().enumerate() {
                let handle = unsafe {
                    self_heap_cpu_handle_offset(
                        self.targets[target_index].rtv_heap,
                        self.device,
                        D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        i,
                    )
                };
                unsafe {
                    call_slot_i32_4(
                        self.device as ComObject,
                        DEVICE_CREATE_RENDER_TARGET_VIEW,
                        *resource,
                        std::ptr::null_mut(),
                        handle.ptr as *mut c_void,
                        std::ptr::null_mut(),
                    );
                }
            }
        }
        let gbuffer_rtv_handles: [D3d12CpuDescriptorHandle; 3] = [
            unsafe {
                self_heap_cpu_handle_offset(
                    self.targets[target_index].rtv_heap,
                    self.device,
                    D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    0,
                )
            },
            unsafe {
                self_heap_cpu_handle_offset(
                    self.targets[target_index].rtv_heap,
                    self.device,
                    D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    1,
                )
            },
            unsafe {
                self_heap_cpu_handle_offset(
                    self.targets[target_index].rtv_heap,
                    self.device,
                    D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    2,
                )
            },
        ];
        let gbuffer_depth = self.targets[target_index]
            .depth_view
            .as_ref()
            .map(|v| v.resource);
        // The camera uniform's write through the mesh state's own uniform
        // buffer.
        let uniform_address = unsafe {
            call_slot_u64_0(
                uniform_buffer as ComObject,
                RESOURCE_GET_GPU_VIRTUAL_ADDRESS,
            )
        };
        unsafe {
            let mut map: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_4(
                uniform_buffer as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr == S_OK && !map.is_null() {
                std::ptr::copy_nonoverlapping(uniform.as_ptr(), map as *mut u8, 256);
                call_slot_i32_2(
                    uniform_buffer as ComObject,
                    RESOURCE_UNMAP,
                    0 as *mut c_void,
                    std::ptr::null_mut(),
                );
            }
        }
        // The lights' bytes into the deferred path's own buffer.
        unsafe {
            let mut map: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_4(
                self.lighting_lights_buffer as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr == S_OK && !map.is_null() {
                std::ptr::copy_nonoverlapping(
                    &block as *const LightsUniform as *const u8,
                    map as *mut u8,
                    256,
                );
                call_slot_i32_2(
                    self.lighting_lights_buffer as ComObject,
                    RESOURCE_UNMAP,
                    0 as *mut c_void,
                    std::ptr::null_mut(),
                );
            }
        }
        let lights_address = unsafe {
            call_slot_u64_0(
                self.lighting_lights_buffer as ComObject,
                RESOURCE_GET_GPU_VIRTUAL_ADDRESS,
            )
        };
        let cbv_desc = d3d::D3d12ConstantBufferViewDesc {
            buffer_location: uniform_address,
            size_in_bytes: 256,
            union_tail_padding: [0],
        };
        let lights_cbv_desc = d3d::D3d12ConstantBufferViewDesc {
            buffer_location: lights_address,
            size_in_bytes: 256,
            union_tail_padding: [0],
        };
        let srv_desc = d3d::D3d12ShaderResourceViewDesc {
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            view_dimension: d3d::D3D12_SRV_DIMENSION_TEXTURE2D,
            shader_4_component_mapping: d3d::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            header_padding: [0],
            texture_2d: d3d::D3d12Tex2DSrv {
                most_detailed_mip: 0,
                mip_levels: 1,
                plane_slice: 0,
                resource_min_lod_clamp: 0.0,
            },
            union_tail_padding: [0, 0],
        };
        let vertex_buffer_view = D3d12VertexBufferView {
            buffer_location: unsafe {
                call_slot_u64_0(vertex_buffer as ComObject, RESOURCE_GET_GPU_VIRTUAL_ADDRESS)
            },
            size_in_bytes: vertex_bytes,
            stride_in_bytes: 32,
        };
        let index_buffer_view = d3d::D3d12IndexBufferView {
            buffer_location: unsafe {
                call_slot_u64_0(index_buffer as ComObject, RESOURCE_GET_GPU_VIRTUAL_ADDRESS)
            },
            size_in_bytes: index_count * 4,
            format: DXGI_FORMAT_R32_UINT,
        };
        let viewport = D3d12Viewport {
            top_left_x: 0.0,
            top_left_y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = D3d12Rect {
            left: 0,
            top: 0,
            right: self.config.width as i32,
            bottom: self.config.height as i32,
        };
        let clear_colour = [
            self.clear_color.r as f32 / 255.0,
            self.clear_color.g as f32 / 255.0,
            self.clear_color.b as f32 / 255.0,
            self.clear_color.a as f32 / 255.0,
        ];
        let no_rect = D3d12Rect {
            left: 0,
            top: 0,
            right: self.config.width as i32,
            bottom: self.config.height as i32,
        };
        // ---- Step one: the geometry pass ----
        let rtvs: [D3d12CpuDescriptorHandle; 3] = gbuffer_rtv_handles;
        let mut gbuffer_dsv = D3d12CpuDescriptorHandle { ptr: 0 };
        let has_dsv = gbuffer_depth.is_some();
        if let Some(depth) = gbuffer_depth {
            // The target's DSV heap: one descriptor (the creation
            // happens once, the same rule the RTV heap follows).
            if self.targets[target_index].dsv_heap.is_null() {
                match unsafe {
                    self_create_descriptor_heap(
                        self.device,
                        D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                        1,
                        D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    )
                } {
                    Ok(heap) => self.targets[target_index].dsv_heap = heap,
                    Err(_) => {
                        self.violations += 1;
                        return;
                    }
                }
                let dsv_desc = D3d12DepthStencilViewDesc {
                    format: DXGI_FORMAT_D24_UNORM_S8_UINT,
                    view_dimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    flags: 0,
                    texture_2d: D3d12Tex2DDsv { mip_slice: 0 },
                };
                let handle = unsafe { self_heap_cpu_handle(self.targets[target_index].dsv_heap) };
                unsafe {
                    call_slot_i32_3(
                        self.device as ComObject,
                        DEVICE_CREATE_DEPTH_STENCIL_VIEW,
                        depth,
                        &dsv_desc as *const D3d12DepthStencilViewDesc as *mut c_void,
                        handle.ptr as *mut c_void,
                    );
                }
            }
            gbuffer_dsv = unsafe { self_heap_cpu_handle(self.targets[target_index].dsv_heap) };
        }
        unsafe {
            // The attachments come in at RENDER_TARGET (the creation's
            // state) or PIXEL_SHADER_RESOURCE (the last lighting pass's
            // state); the barriers bring them into RENDER_TARGET before
            // the draw.
            if self.targets[target_index].views_state_read {
                for resource in gbuffer_resources.iter() {
                    let barrier = D3d12ResourceBarrier {
                        type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                        flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        transition: D3d12ResourceTransitionBarrier {
                            resource: *resource,
                            subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            state_before: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                            state_after: D3D12_RESOURCE_STATE_RENDER_TARGET,
                        },
                    };
                    call_slot_i32_3(
                        self.command_list as ComObject,
                        COMMAND_LIST_RESOURCE_BARRIER,
                        1 as *mut c_void,
                        &barrier as *const D3d12ResourceBarrier as *mut c_void,
                        std::ptr::null_mut(),
                    );
                }
                self.targets[target_index].views_state_read = false;
            }
            // The clears: the three RTVs and the depth.
            for handle in rtvs.iter() {
                call_slot_i32_4(
                    self.command_list as ComObject,
                    COMMAND_LIST_CLEAR_RENDER_TARGET_VIEW,
                    handle.ptr as *mut c_void,
                    clear_colour.as_ptr() as *const c_void as *mut c_void,
                    1 as *mut c_void,
                    &no_rect as *const D3d12Rect as *mut c_void,
                );
            }
            if has_dsv {
                call_slot_i32_f32(
                    self.command_list as ComObject,
                    COMMAND_LIST_CLEAR_DEPTH_STENCIL_VIEW,
                    gbuffer_dsv.ptr as *mut c_void,
                    (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL) as u32,
                    self.clear_depth,
                    self.clear_stencil as u32,
                    1,
                    &no_rect as *const D3d12Rect as *mut c_void,
                );
            }
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_PIPELINE_STATE,
                self.gbuffer_pipeline,
                std::ptr::null_mut(),
            );
            let rtv_ptrs = [
                rtvs[0].ptr as *mut c_void,
                rtvs[1].ptr as *mut c_void,
                rtvs[2].ptr as *mut c_void,
            ];
            let rtv_count = 3;
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_OMSET_RENDER_TARGETS,
                rtv_count as *mut c_void,
                rtv_ptrs.as_ptr() as *const c_void as *mut c_void,
                0 as *mut c_void,
                if has_dsv {
                    &gbuffer_dsv as *const D3d12CpuDescriptorHandle as *mut c_void
                } else {
                    0 as *mut c_void
                },
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_SIGNATURE,
                self.gbuffer_root_signature,
                std::ptr::null_mut(),
            );
            // The descriptor writes: the camera's CBV at slot 0, the
            // texture's SRV at slot 1.
            call_slot_i32_2(
                self.device as ComObject,
                DEVICE_CREATE_CONSTANT_BUFFER_VIEW,
                &cbv_desc as *const d3d::D3d12ConstantBufferViewDesc as *mut c_void,
                self_heap_cpu_handle_offset(
                    srv_heap,
                    self.device,
                    d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    0,
                )
                .ptr as *mut c_void,
            );
            call_slot_i32_3(
                self.device as ComObject,
                DEVICE_CREATE_SHADER_RESOURCE_VIEW,
                texture_resource,
                &srv_desc as *const d3d::D3d12ShaderResourceViewDesc as *mut c_void,
                self_heap_cpu_handle_offset(
                    srv_heap,
                    self.device,
                    d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    1,
                )
                .ptr as *mut c_void,
            );
            let heaps = [srv_heap];
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_SET_DESCRIPTOR_HEAPS,
                1 as *mut c_void,
                heaps.as_ptr() as *const c_void as *mut c_void,
                std::ptr::null_mut(),
            );
            let gpu_handle = self_heap_gpu_handle_offset(
                srv_heap,
                self.device,
                d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                0,
            );
            call_slot_void_ptr_u64(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE,
                0 as *mut c_void,
                gpu_handle.ptr,
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_VIEWPORTS,
                1 as *mut c_void,
                &viewport as *const D3d12Viewport as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_SCISSOR_RECTS,
                1 as *mut c_void,
                &scissor as *const D3d12Rect as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_PRIMITIVE_TOPOLOGY,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_VERTEX_BUFFERS,
                0 as *mut c_void,
                1 as *mut c_void,
                &vertex_buffer_view as *const D3d12VertexBufferView as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_INDEX_BUFFER,
                &index_buffer_view as *const d3d::D3d12IndexBufferView as *mut c_void,
                std::ptr::null_mut(),
            );
            // The geometry draw.
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_DRAW_INDEXED_INSTANCED,
                index_count as *mut c_void,
                1 as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                std::ptr::null_mut(),
            );
        }
        // ---- Step two: the lighting pass ----
        unsafe {
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_PIPELINE_STATE,
                self.lighting_pipeline,
                std::ptr::null_mut(),
            );
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_OMSET_RENDER_TARGETS,
                1 as *mut c_void,
                &self.rtv_handle as *const D3d12CpuDescriptorHandle as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                std::ptr::null_mut(),
            );
            // The attachments hand off to the lighting pass's shader
            // reads: the RENDER_TARGET -> PIXEL_SHADER_RESOURCE
            // transitions (one barrier per call, the probe's rule). The
            // barriers land after the lighting pass's OMSet: the G-buffer
            // RTVs are no longer bound when the resources transition.
            for resource in gbuffer_resources.iter() {
                let barrier = D3d12ResourceBarrier {
                    type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    transition: D3d12ResourceTransitionBarrier {
                        resource: *resource,
                        subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        state_before: D3D12_RESOURCE_STATE_RENDER_TARGET,
                        // The lighting pass's sampler reads the
                        // attachments: the pixel-shader read state is the
                        // layout the sampler sees.
                        state_after: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    },
                };
                call_slot_i32_3(
                    self.command_list as ComObject,
                    COMMAND_LIST_RESOURCE_BARRIER,
                    1 as *mut c_void,
                    &barrier as *const D3d12ResourceBarrier as *mut c_void,
                    std::ptr::null_mut(),
                );
            }
            self.targets[target_index].views_state_read = true;
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_SIGNATURE,
                self.lighting_root_signature,
                std::ptr::null_mut(),
            );
            // The descriptor writes: the three G-buffer attachments'
            // SRVs at slots 0-2, the lights' CBV at slot 3.
            for (i, resource) in gbuffer_resources.iter().enumerate() {
                let gbuffer_srv = d3d::D3d12ShaderResourceViewDesc {
                    format: if i == 2 {
                        DXGI_FORMAT_R16G16B16A16_FLOAT
                    } else {
                        DXGI_FORMAT_R8G8B8A8_UNORM
                    },
                    view_dimension: d3d::D3D12_SRV_DIMENSION_TEXTURE2D,
                    shader_4_component_mapping: d3d::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    header_padding: [0],
                    texture_2d: d3d::D3d12Tex2DSrv {
                        most_detailed_mip: 0,
                        mip_levels: 1,
                        plane_slice: 0,
                        resource_min_lod_clamp: 0.0,
                    },
                    union_tail_padding: [0, 0],
                };
                call_slot_i32_3(
                    self.device as ComObject,
                    DEVICE_CREATE_SHADER_RESOURCE_VIEW,
                    *resource,
                    &gbuffer_srv as *const d3d::D3d12ShaderResourceViewDesc as *mut c_void,
                    self_heap_cpu_handle_offset(
                        self.lighting_srv_heap,
                        self.device,
                        d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        i,
                    )
                    .ptr as *mut c_void,
                );
            }
            call_slot_i32_2(
                self.device as ComObject,
                DEVICE_CREATE_CONSTANT_BUFFER_VIEW,
                &lights_cbv_desc as *const d3d::D3d12ConstantBufferViewDesc as *mut c_void,
                self_heap_cpu_handle_offset(
                    self.lighting_srv_heap,
                    self.device,
                    d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    3,
                )
                .ptr as *mut c_void,
            );
            let heaps = [self.lighting_srv_heap];
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_SET_DESCRIPTOR_HEAPS,
                1 as *mut c_void,
                heaps.as_ptr() as *const c_void as *mut c_void,
                std::ptr::null_mut(),
            );
            let gpu_handle = self_heap_gpu_handle_offset(
                self.lighting_srv_heap,
                self.device,
                d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                0,
            );
            call_slot_void_ptr_u64(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE,
                0 as *mut c_void,
                gpu_handle.ptr,
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_VIEWPORTS,
                1 as *mut c_void,
                &viewport as *const D3d12Viewport as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_SCISSOR_RECTS,
                1 as *mut c_void,
                &scissor as *const D3d12Rect as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_PRIMITIVE_TOPOLOGY,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_DRAW_INSTANCED,
                3 as *mut c_void,
                1 as *mut c_void,
                std::ptr::null_mut(),
            );
        }
    }

    /// The post-processing chain (M7): the HDR geometry pass writes the
    /// shaded radiance and the world position into the target's two
    /// Rgba16F attachments, the composite pass samples both and applies
    /// the enabled effects into the default framebuffer (the same two-
    /// pass flow the Vulkan and the OpenGL paths carry).
    fn submit_post_chain(
        &mut self,
        submission: &DrawSubmission,
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
        if mesh_id >= self.meshes.len() || self.meshes[mesh_id].is_none() {
            self.violations += 1;
            return;
        }
        if !self.mesh_ready || !self.ensure_post_resources() {
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
        let projection = Mat4::perspective(camera.fov_y_rad, aspect, camera.near, camera.far);
        let view_projection = projection.multiply(view);
        let model = submission.model;
        // The camera uniform block: the same 256-byte CBV alignment the
        // mesh path's block carries.
        let mut uniform = [0u8; 256];
        uniform[0..64].copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                view_projection.m.as_ptr() as *const u8,
                64,
            )
        });
        uniform[64..128].copy_from_slice(unsafe {
            std::slice::from_raw_parts(model.m.as_ptr() as *const u8, 64)
        });
        // The config block: the three vec4 params the composite shader's
        // Config block carries (the 48 bytes the toggles and the
        // parameters ride).
        let mut config_block = [0u8; 256];
        let params0 = [
            if config.tonemapping { 1.0f32 } else { 0.0 },
            if config.bloom { 1.0 } else { 0.0 },
            if config.depth_of_field { 1.0 } else { 0.0 },
            if config.ssao { 1.0 } else { 0.0 },
        ];
        let params1 = [
            config.bloom_threshold,
            config.bloom_knee,
            config.bloom_intensity,
            config.ssao_radius,
        ];
        let params2 = [
            config.focal_distance,
            config.focus_near_range,
            config.focus_far_range,
            if config.anti_aliasing { 1.0 } else { 0.0 },
        ];
        config_block[0..16].copy_from_slice(unsafe {
            std::slice::from_raw_parts(params0.as_ptr() as *const u8, 16)
        });
        config_block[16..32].copy_from_slice(unsafe {
            std::slice::from_raw_parts(params1.as_ptr() as *const u8, 16)
        });
        config_block[32..48].copy_from_slice(unsafe {
            std::slice::from_raw_parts(params2.as_ptr() as *const u8, 16)
        });
        // The volume's block: the origin, the dims, the cell size and
        // the indirect light's toggle and intensity the POST_GBUFFER
        // pixel shader's Volume block reads (the 48-byte std140 block).
        let volume_block = VolumeUniform {
            origin: [
                self.volume_uniform[0],
                self.volume_uniform[1],
                self.volume_uniform[2],
                0.0,
            ],
            dims: [
                self.volume_uniform[4],
                self.volume_uniform[5],
                self.volume_uniform[6],
                0.0,
            ],
            params: [
                self.volume_uniform[8],
                config.indirect_intensity,
                if config.indirect_light { 1.0f32 } else { 0.0 },
                0.0,
            ],
        };
        let (vertex_buffer, index_buffer, index_count, srv_heap, uniform_buffer) = {
            let state = self.meshes[mesh_id].as_ref().unwrap();
            (
                state.vertex_buffer,
                state.index_buffer,
                state.index_count,
                state.srv_heap,
                state.uniform_buffer,
            )
        };
        let vertex_bytes = {
            let state = self.meshes[mesh_id].as_ref().unwrap();
            state.vertex_bytes
        };
        let texture_resource = self
            .bound_texture
            .and_then(|i| self.textures_2d.get(i))
            .filter(|t| !t.released)
            .map(|t| t.resource)
            .unwrap_or(
                self.default_texture
                    .and_then(|i| self.textures_2d.get(i))
                    .filter(|t| !t.released)
                    .map(|t| t.resource)
                    .unwrap_or(std::ptr::null_mut()),
            );
        let post_resources: Vec<d3d::D3d12ResourceHandle> = self.targets[target_index]
            .views
            .iter()
            .take(2)
            .map(|v| v.resource)
            .collect();
        if post_resources.len() != 2 {
            self.violations += 1;
            return;
        }
        let post_depth = self.targets[target_index]
            .depth_view
            .as_ref()
            .map(|v| v.resource);
        // The target's RTV heap (one descriptor per attachment): the
        // creation happens once (the target's own heap field).
        if self.targets[target_index].rtv_heap.is_null() {
            match unsafe {
                self_create_descriptor_heap(
                    self.device,
                    D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    2,
                    D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                )
            } {
                Ok(heap) => self.targets[target_index].rtv_heap = heap,
                Err(_) => {
                    self.violations += 1;
                    return;
                }
            }
            for (i, resource) in post_resources.iter().enumerate() {
                let handle = unsafe {
                    self_heap_cpu_handle_offset(
                        self.targets[target_index].rtv_heap,
                        self.device,
                        D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        i,
                    )
                };
                unsafe {
                    call_slot_i32_4(
                        self.device as ComObject,
                        DEVICE_CREATE_RENDER_TARGET_VIEW,
                        *resource,
                        std::ptr::null_mut(),
                        handle.ptr as *mut c_void,
                        std::ptr::null_mut(),
                    );
                }
            }
        }
        let post_rtv_handles: [D3d12CpuDescriptorHandle; 2] = [
            unsafe {
                self_heap_cpu_handle_offset(
                    self.targets[target_index].rtv_heap,
                    self.device,
                    D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    0,
                )
            },
            unsafe {
                self_heap_cpu_handle_offset(
                    self.targets[target_index].rtv_heap,
                    self.device,
                    D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    1,
                )
            },
        ];
        // The camera uniform's write through the mesh state's own uniform
        // buffer.
        let uniform_address = unsafe {
            call_slot_u64_0(
                uniform_buffer as ComObject,
                RESOURCE_GET_GPU_VIRTUAL_ADDRESS,
            )
        };
        unsafe {
            let mut map: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_4(
                uniform_buffer as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr == S_OK && !map.is_null() {
                std::ptr::copy_nonoverlapping(uniform.as_ptr(), map as *mut u8, 256);
                call_slot_i32_2(
                    uniform_buffer as ComObject,
                    RESOURCE_UNMAP,
                    0 as *mut c_void,
                    std::ptr::null_mut(),
                );
            }
        }
        // The lights' bytes into the post path's own buffer.
        if self.post_lights_buffer.is_null() {
            match unsafe {
                self_create_buffer(
                    self.device,
                    256,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )
            } {
                Ok(buffer) => self.post_lights_buffer = buffer,
                Err(_) => {
                    self.violations += 1;
                    return;
                }
            }
        }
        unsafe {
            let mut map: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_4(
                self.post_lights_buffer as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr == S_OK && !map.is_null() {
                std::ptr::copy_nonoverlapping(
                    &block as *const LightsUniform as *const u8,
                    map as *mut u8,
                    256,
                );
                call_slot_i32_2(
                    self.post_lights_buffer as ComObject,
                    RESOURCE_UNMAP,
                    0 as *mut c_void,
                    std::ptr::null_mut(),
                );
            }
        }
        let lights_address = unsafe {
            call_slot_u64_0(
                self.post_lights_buffer as ComObject,
                RESOURCE_GET_GPU_VIRTUAL_ADDRESS,
            )
        };
        // The config bytes into the composite's config CBV buffer.
        let config_address = unsafe {
            call_slot_u64_0(
                self.post_config_buffer as ComObject,
                RESOURCE_GET_GPU_VIRTUAL_ADDRESS,
            )
        };
        unsafe {
            let mut map: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_4(
                self.post_config_buffer as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr == S_OK && !map.is_null() {
                std::ptr::copy_nonoverlapping(
                    config_block.as_ptr(),
                    map as *mut u8,
                    256,
                );
                call_slot_i32_2(
                    self.post_config_buffer as ComObject,
                    RESOURCE_UNMAP,
                    0 as *mut c_void,
                    std::ptr::null_mut(),
                );
            }
        }
        let cbv_desc = d3d::D3d12ConstantBufferViewDesc {
            buffer_location: uniform_address,
            size_in_bytes: 256,
            union_tail_padding: [0],
        };
        let lights_cbv_desc = d3d::D3d12ConstantBufferViewDesc {
            buffer_location: lights_address,
            size_in_bytes: 256,
            union_tail_padding: [0],
        };
        let config_cbv_desc = d3d::D3d12ConstantBufferViewDesc {
            buffer_location: config_address,
            size_in_bytes: 256,
            union_tail_padding: [0],
        };
        let srv_desc = d3d::D3d12ShaderResourceViewDesc {
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            view_dimension: d3d::D3D12_SRV_DIMENSION_TEXTURE2D,
            shader_4_component_mapping: d3d::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            header_padding: [0],
            texture_2d: d3d::D3d12Tex2DSrv {
                most_detailed_mip: 0,
                mip_levels: 1,
                plane_slice: 0,
                resource_min_lod_clamp: 0.0,
            },
            union_tail_padding: [0, 0],
        };
        let vertex_buffer_view = D3d12VertexBufferView {
            buffer_location: unsafe {
                call_slot_u64_0(vertex_buffer as ComObject, RESOURCE_GET_GPU_VIRTUAL_ADDRESS)
            },
            size_in_bytes: vertex_bytes,
            stride_in_bytes: 32,
        };
        let index_buffer_view = d3d::D3d12IndexBufferView {
            buffer_location: unsafe {
                call_slot_u64_0(index_buffer as ComObject, RESOURCE_GET_GPU_VIRTUAL_ADDRESS)
            },
            size_in_bytes: index_count * 4,
            format: DXGI_FORMAT_R32_UINT,
        };
        let viewport = D3d12Viewport {
            top_left_x: 0.0,
            top_left_y: 0.0,
            width: self.config.width as f32,
            height: self.config.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = D3d12Rect {
            left: 0,
            top: 0,
            right: self.config.width as i32,
            bottom: self.config.height as i32,
        };
        let clear_colour = [
            self.clear_color.r as f32 / 255.0,
            self.clear_color.g as f32 / 255.0,
            self.clear_color.b as f32 / 255.0,
            self.clear_color.a as f32 / 255.0,
        ];
        let no_rect = D3d12Rect {
            left: 0,
            top: 0,
            right: self.config.width as i32,
            bottom: self.config.height as i32,
        };
        // ---- Step one: the HDR geometry pass ----
        let rtvs: [D3d12CpuDescriptorHandle; 2] = post_rtv_handles;
        let mut post_dsv = D3d12CpuDescriptorHandle { ptr: 0 };
        let has_dsv = post_depth.is_some();
        if let Some(depth) = post_depth {
            // The target's DSV heap: one descriptor (the creation
            // happens once, the same rule the RTV heap follows).
            if self.targets[target_index].dsv_heap.is_null() {
                match unsafe {
                    self_create_descriptor_heap(
                        self.device,
                        D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                        1,
                        D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    )
                } {
                    Ok(heap) => self.targets[target_index].dsv_heap = heap,
                    Err(_) => {
                        self.violations += 1;
                        return;
                    }
                }
                let dsv_desc = D3d12DepthStencilViewDesc {
                    format: DXGI_FORMAT_D24_UNORM_S8_UINT,
                    view_dimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    flags: 0,
                    texture_2d: D3d12Tex2DDsv { mip_slice: 0 },
                };
                let handle = unsafe { self_heap_cpu_handle(self.targets[target_index].dsv_heap) };
                unsafe {
                    call_slot_i32_3(
                        self.device as ComObject,
                        DEVICE_CREATE_DEPTH_STENCIL_VIEW,
                        depth,
                        &dsv_desc as *const D3d12DepthStencilViewDesc as *mut c_void,
                        handle.ptr as *mut c_void,
                    );
                }
            }
            post_dsv = unsafe { self_heap_cpu_handle(self.targets[target_index].dsv_heap) };
        }
        unsafe {
            // The attachments come in at RENDER_TARGET (the creation's
            // state) or PIXEL_SHADER_RESOURCE (the last composite pass's
            // state); the barriers bring them into RENDER_TARGET before
            // the draw.
            if self.targets[target_index].views_state_read {
                for resource in post_resources.iter() {
                    let barrier = D3d12ResourceBarrier {
                        type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                        flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        transition: D3d12ResourceTransitionBarrier {
                            resource: *resource,
                            subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            state_before: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                            state_after: D3D12_RESOURCE_STATE_RENDER_TARGET,
                        },
                    };
                    call_slot_i32_3(
                        self.command_list as ComObject,
                        COMMAND_LIST_RESOURCE_BARRIER,
                        1 as *mut c_void,
                        &barrier as *const D3d12ResourceBarrier as *mut c_void,
                        std::ptr::null_mut(),
                    );
                }
                self.targets[target_index].views_state_read = false;
            }
            // The clears: the two RTVs and the depth.
            for handle in rtvs.iter() {
                call_slot_i32_4(
                    self.command_list as ComObject,
                    COMMAND_LIST_CLEAR_RENDER_TARGET_VIEW,
                    handle.ptr as *mut c_void,
                    clear_colour.as_ptr() as *const c_void as *mut c_void,
                    1 as *mut c_void,
                    &no_rect as *const D3d12Rect as *mut c_void,
                );
            }
            if has_dsv {
                call_slot_i32_f32(
                    self.command_list as ComObject,
                    COMMAND_LIST_CLEAR_DEPTH_STENCIL_VIEW,
                    post_dsv.ptr as *mut c_void,
                    (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL) as u32,
                    self.clear_depth,
                    self.clear_stencil as u32,
                    1,
                    &no_rect as *const D3d12Rect as *mut c_void,
                );
            }
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_PIPELINE_STATE,
                self.post_gbuffer_pipeline,
                std::ptr::null_mut(),
            );
            let rtv_ptrs = [
                rtvs[0].ptr as *mut c_void,
                rtvs[1].ptr as *mut c_void,
            ];
            let rtv_count = 2;
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_OMSET_RENDER_TARGETS,
                rtv_count as *mut c_void,
                rtv_ptrs.as_ptr() as *const c_void as *mut c_void,
                0 as *mut c_void,
                if has_dsv {
                    &post_dsv as *const D3d12CpuDescriptorHandle as *mut c_void
                } else {
                    0 as *mut c_void
                },
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_SIGNATURE,
                self.post_gbuffer_root_signature,
                std::ptr::null_mut(),
            );
            // The descriptor writes: the camera's CBV at slot 0, the
            // texture's SRV at slot 1, the lights' CBV at slot 2.
            call_slot_i32_2(
                self.device as ComObject,
                DEVICE_CREATE_CONSTANT_BUFFER_VIEW,
                &cbv_desc as *const d3d::D3d12ConstantBufferViewDesc as *mut c_void,
                self_heap_cpu_handle_offset(
                    srv_heap,
                    self.device,
                    d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    0,
                )
                .ptr as *mut c_void,
            );
            call_slot_i32_3(
                self.device as ComObject,
                DEVICE_CREATE_SHADER_RESOURCE_VIEW,
                texture_resource,
                &srv_desc as *const d3d::D3d12ShaderResourceViewDesc as *mut c_void,
                self_heap_cpu_handle_offset(
                    srv_heap,
                    self.device,
                    d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    1,
                )
                .ptr as *mut c_void,
            );
            call_slot_i32_2(
                self.device as ComObject,
                DEVICE_CREATE_CONSTANT_BUFFER_VIEW,
                &lights_cbv_desc as *const d3d::D3d12ConstantBufferViewDesc as *mut c_void,
                self_heap_cpu_handle_offset(
                    srv_heap,
                    self.device,
                    d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    2,
                )
                .ptr as *mut c_void,
            );
            // The volume's SRV (the 3D texture) at slot 3 and the
            // volume's CBV (the 256-byte uniform block) at slot 4: the
            // writes happen only when the volume texture exists (the
            // toggle-off path leaves the default descriptors).
            if let Some(volume) = self
                .light_volume
                .as_ref()
                .filter(|v| !v.inner.lock().unwrap().released)
            {
                let volume_index = self
                    .textures_3d
                    .iter()
                    .position(|t| Arc::ptr_eq(&t.inner, &volume.inner))
                    .unwrap();
                let volume_resource = self.textures_3d[volume_index].resource;
                let volume_srv = d3d::D3d12ShaderResourceViewDesc {
                    format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                    view_dimension: d3d::D3D12_SRV_DIMENSION_TEXTURE3D,
                    shader_4_component_mapping:
                        d3d::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    header_padding: [0],
                    // The union's storage: the tex3d member's three
                    // fields ride the tex2d member's layout (the mip
                    // slice, the mip levels, the minimum LOD clamp —
                    // the same 12 bytes the TEXTURE3D dimension reads).
                    texture_2d: d3d::D3d12Tex2DSrv {
                        most_detailed_mip: 0,
                        mip_levels: 1,
                        plane_slice: 0,
                        resource_min_lod_clamp: 0.0,
                    },
                    union_tail_padding: [0, 0],
                };
                call_slot_i32_3(
                    self.device as ComObject,
                    DEVICE_CREATE_SHADER_RESOURCE_VIEW,
                    volume_resource,
                    &volume_srv as *const d3d::D3d12ShaderResourceViewDesc
                        as *mut c_void,
                    self_heap_cpu_handle_offset(
                        srv_heap,
                        self.device,
                        d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        3,
                    )
                    .ptr as *mut c_void,
                );
            }
            if self.post_volume_buffer.is_null() {
                match {
                    self_create_buffer(
                        self.device,
                        256,
                        D3D12_HEAP_TYPE_UPLOAD,
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                    )
                } {
                    Ok(buffer) => self.post_volume_buffer = buffer,
                    Err(_) => {
                        self.violations += 1;
                        return;
                    }
                }
            }
            let volume_address = call_slot_u64_0(
                self.post_volume_buffer as ComObject,
                RESOURCE_GET_GPU_VIRTUAL_ADDRESS,
            );
            {
                let mut map: *mut c_void = std::ptr::null_mut();
                let hr = call_slot_i32_4(
                    self.post_volume_buffer as ComObject,
                    RESOURCE_MAP,
                    0 as *mut c_void,
                    &D3d12Range { begin: 0, end: 0 } as *const D3d12Range
                        as *mut c_void,
                    &mut map as *mut _ as *mut c_void,
                    std::ptr::null_mut(),
                );
                if hr == S_OK && !map.is_null() {
                    std::ptr::copy_nonoverlapping(
                        &volume_block as *const VolumeUniform as *const u8,
                        map as *mut u8,
                        std::mem::size_of::<VolumeUniform>(),
                    );
                    call_slot_i32_2(
                        self.post_volume_buffer as ComObject,
                        RESOURCE_UNMAP,
                        0 as *mut c_void,
                        std::ptr::null_mut(),
                    );
                }
            }
            let volume_cbv_desc = d3d::D3d12ConstantBufferViewDesc {
                buffer_location: volume_address,
                size_in_bytes: 256,
                union_tail_padding: [0],
            };
            call_slot_i32_2(
                self.device as ComObject,
                DEVICE_CREATE_CONSTANT_BUFFER_VIEW,
                &volume_cbv_desc as *const d3d::D3d12ConstantBufferViewDesc
                    as *mut c_void,
                self_heap_cpu_handle_offset(
                    srv_heap,
                    self.device,
                    d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    4,
                )
                .ptr as *mut c_void,
            );
            let heaps = [srv_heap];
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_SET_DESCRIPTOR_HEAPS,
                1 as *mut c_void,
                heaps.as_ptr() as *const c_void as *mut c_void,
                std::ptr::null_mut(),
            );
            let heaps = [srv_heap];
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_SET_DESCRIPTOR_HEAPS,
                1 as *mut c_void,
                heaps.as_ptr() as *const c_void as *mut c_void,
                std::ptr::null_mut(),
            );
            let gpu_handle = self_heap_gpu_handle_offset(
                srv_heap,
                self.device,
                d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                0,
            );
            call_slot_void_ptr_u64(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE,
                0 as *mut c_void,
                gpu_handle.ptr,
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_VIEWPORTS,
                1 as *mut c_void,
                &viewport as *const D3d12Viewport as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_SCISSOR_RECTS,
                1 as *mut c_void,
                &scissor as *const D3d12Rect as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_PRIMITIVE_TOPOLOGY,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_VERTEX_BUFFERS,
                0 as *mut c_void,
                1 as *mut c_void,
                &vertex_buffer_view as *const D3d12VertexBufferView as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_INDEX_BUFFER,
                &index_buffer_view as *const d3d::D3d12IndexBufferView as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            // The geometry draw.
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_DRAW_INDEXED_INSTANCED,
                index_count as *mut c_void,
                1 as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                std::ptr::null_mut(),
            );
        }
        // ---- Step two: the composite pass ----
        unsafe {
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_PIPELINE_STATE,
                self.post_composite_pipeline,
                std::ptr::null_mut(),
            );
            call_slot_i32_5(
                self.command_list as ComObject,
                COMMAND_LIST_OMSET_RENDER_TARGETS,
                1 as *mut c_void,
                &self.rtv_handle as *const D3d12CpuDescriptorHandle as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                std::ptr::null_mut(),
            );
            // The attachments hand off to the composite pass's shader
            // reads: the RENDER_TARGET -> PIXEL_SHADER_RESOURCE
            // transitions (one barrier per call, the probe's rule). The
            // barriers land after the composite pass's OMSet: the HDR
            // RTVs are no longer bound when the resources transition.
            for resource in post_resources.iter() {
                let barrier = D3d12ResourceBarrier {
                    type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    transition: D3d12ResourceTransitionBarrier {
                        resource: *resource,
                        subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        state_before: D3D12_RESOURCE_STATE_RENDER_TARGET,
                        state_after: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    },
                };
                call_slot_i32_3(
                    self.command_list as ComObject,
                    COMMAND_LIST_RESOURCE_BARRIER,
                    1 as *mut c_void,
                    &barrier as *const D3d12ResourceBarrier as *mut c_void,
                    std::ptr::null_mut(),
                );
            }
            self.targets[target_index].views_state_read = true;
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_SIGNATURE,
                self.post_composite_root_signature,
                std::ptr::null_mut(),
            );
            // The descriptor writes: the two HDR attachments' SRVs at
            // slots 0-1, the config CBV at slot 2.
            for (i, resource) in post_resources.iter().enumerate() {
                let hdr_srv = d3d::D3d12ShaderResourceViewDesc {
                    format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                    view_dimension: d3d::D3D12_SRV_DIMENSION_TEXTURE2D,
                    shader_4_component_mapping: d3d::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    header_padding: [0],
                    texture_2d: d3d::D3d12Tex2DSrv {
                        most_detailed_mip: 0,
                        mip_levels: 1,
                        plane_slice: 0,
                        resource_min_lod_clamp: 0.0,
                    },
                    union_tail_padding: [0, 0],
                };
                call_slot_i32_3(
                    self.device as ComObject,
                    DEVICE_CREATE_SHADER_RESOURCE_VIEW,
                    *resource,
                    &hdr_srv as *const d3d::D3d12ShaderResourceViewDesc as *mut c_void,
                    self_heap_cpu_handle_offset(
                        self.post_srv_heap,
                        self.device,
                        d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        i,
                    )
                    .ptr as *mut c_void,
                );
            }
            call_slot_i32_2(
                self.device as ComObject,
                DEVICE_CREATE_CONSTANT_BUFFER_VIEW,
                &config_cbv_desc as *const d3d::D3d12ConstantBufferViewDesc as *mut c_void,
                self_heap_cpu_handle_offset(
                    self.post_srv_heap,
                    self.device,
                    d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    2,
                )
                .ptr as *mut c_void,
            );
            let heaps = [self.post_srv_heap];
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_SET_DESCRIPTOR_HEAPS,
                1 as *mut c_void,
                heaps.as_ptr() as *const c_void as *mut c_void,
                std::ptr::null_mut(),
            );
            let gpu_handle = self_heap_gpu_handle_offset(
                self.post_srv_heap,
                self.device,
                d3d::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                0,
            );
            call_slot_void_ptr_u64(
                self.command_list as ComObject,
                COMMAND_LIST_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE,
                0 as *mut c_void,
                gpu_handle.ptr,
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_VIEWPORTS,
                1 as *mut c_void,
                &viewport as *const D3d12Viewport as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RSSET_SCISSOR_RECTS,
                1 as *mut c_void,
                &scissor as *const D3d12Rect as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_IASET_PRIMITIVE_TOPOLOGY,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_DRAW_INSTANCED,
                3 as *mut c_void,
                1 as *mut c_void,
                std::ptr::null_mut(),
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
            let format = desc.color_formats[i];
            let resource = unsafe {
                self_create_texture(
                    self.device,
                    self.to_d3d_format(format),
                    desc.width,
                    desc.height,
                    // The deferred lighting pass reads the attachments'
                    // SRVs: the resource flag carries both the render
                    // target and the shader resource (the same rule the
                    // Vulkan's ColorAttachment | Sampled carries).
                    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                )?
            };
            views.push(D3d12Texture2D {
                inner: Arc::new(Mutex::new(TextureViewInner {
                    width: desc.width,
                    height: desc.height,
                    depth: 1,
                    format,
                    is_depth: false,
                    released: false,
                })),
                resource,
                released: false,
                state: D3D12_RESOURCE_STATE_RENDER_TARGET,
            });
        }
        let depth_view = if desc.depth_attachment {
            let format = desc.depth_format;
            let resource = unsafe {
                self_create_texture(
                    self.device,
                    DXGI_FORMAT_D24_UNORM_S8_UINT,
                    desc.width,
                    desc.height,
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                    None,
                )?
            };
            Some(D3d12Texture2D {
                inner: Arc::new(Mutex::new(TextureViewInner {
                    width: desc.width,
                    height: desc.height,
                    depth: 1,
                    format,
                    is_depth: true,
                    released: false,
                })),
                resource,
                released: false,
                state: D3D12_RESOURCE_STATE_DEPTH_WRITE,
            })
        } else {
            None
        };
        let target = D3d12RenderTarget {
            inner: Arc::new(Mutex::new(RenderTargetInner {
                width: desc.width,
                height: desc.height,
                color_attachment_count: desc.color_attachment_count,
                depth_attachment: desc.depth_attachment,
                released: false,
            })),
            views,
            depth_view,
            rtv_heap: std::ptr::null_mut(),
            dsv_heap: std::ptr::null_mut(),
            views_state_read: false,
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
            let depth_view = t.depth_view.as_ref()?;
            return Some(TextureView {
                inner: Arc::clone(&depth_view.inner),
            });
        }
        let view = t.views.get(attachment as usize)?;
        Some(TextureView {
            inner: Arc::clone(&view.inner),
        })
    }

    fn bind_texture(&mut self, view: Option<&TextureView>, unit: u32) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        // The first sampler unit: the mesh path's draw binds the texture
        // (the later units record without the draw's use).
        if unit != 0 {
            return;
        }
        match view {
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
        desc.validate()?;
        let resource = unsafe {
            self_create_texture(
                self.device,
                self.to_d3d_format(desc.format),
                desc.width,
                desc.height,
                0,
                D3D12_RESOURCE_STATE_COMMON,
                None,
            )?
        };
        let view = D3d12Texture2D {
            inner: Arc::new(Mutex::new(TextureViewInner {
                width: desc.width,
                height: desc.height,
                depth: 1,
                format: desc.format,
                is_depth: false,
                released: false,
            })),
            resource,
            released: false,
            state: D3D12_RESOURCE_STATE_COMMON,
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

    fn fill_texture_2d(&mut self, view: &TextureView, data: &[u8]) -> Result<(), RenderError> {
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
        // The upload path: one UPLOAD-heap staging buffer the CPU writes
        // through the Map/Unmap pair, the command sequence the resource's
        // transition (COMMON -> COPY_DEST -> the shader-read state)
        // brackets, and the CopyTextureRegion the placed footprint
        // carries (the same rule the readback's staging copy follows).
        let (width, height, format) = {
            let inner = view.inner.lock().unwrap();
            (inner.width, inner.height, inner.format)
        };
        let bytes = bytes_per_pixel(format);
        let row_pitch = d3d::texture_row_pitch(width, bytes);
        let buffer_size = row_pitch as u64 * height as u64;
        let staging = unsafe {
            self_create_buffer(
                self.device,
                buffer_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?
        };
        unsafe {
            let mut map: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_4(
                staging as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr != S_OK || map.is_null() {
                return Err(RenderError::from(
                    "fill_texture_2d: the staging buffer's map failed",
                ));
            }
            let map = map as *mut u8;
            for row in 0..height as usize {
                let src = data.as_ptr().add(row * width as usize * bytes as usize);
                let dst = map.add(row * row_pitch as usize);
                std::ptr::copy_nonoverlapping(src, dst, width as usize * bytes as usize);
            }
            call_slot_i32_2(
                staging as ComObject,
                RESOURCE_UNMAP,
                0 as *mut c_void,
                std::ptr::null_mut(),
            );
        }
        let resource = self.textures_2d[index].resource;
        let current_state = self.textures_2d[index].state;
        let to_copy_dest = D3d12ResourceBarrier {
            type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            transition: D3d12ResourceTransitionBarrier {
                resource,
                subresource: 0,
                state_before: current_state,
                state_after: D3D12_RESOURCE_STATE_COPY_DEST,
            },
        };
        let to_shader_read = D3d12ResourceBarrier {
            type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            transition: D3d12ResourceTransitionBarrier {
                resource,
                subresource: 0,
                state_before: D3D12_RESOURCE_STATE_COPY_DEST,
                state_after: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            },
        };
        let dst = D3d12TextureCopyLocation {
            resource,
            type_: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            placed_footprint: D3d12PlacedFootprint {
                offset: 0,
                footprint: D3d12SubresourceFootprint {
                    format: 0,
                    width: 0,
                    height: 0,
                    depth: 0,
                    row_pitch: 0,
                },
            },
        };
        let src = D3d12TextureCopyLocation {
            resource: staging,
            type_: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            placed_footprint: D3d12PlacedFootprint {
                offset: 0,
                footprint: D3d12SubresourceFootprint {
                    format: self.to_d3d_format(format),
                    width,
                    height,
                    depth: 1,
                    row_pitch,
                },
            },
        };
        unsafe {
            if !self.frame_open {
                // The frame's command list opens: the allocator and the
                // list reset (the same rule the clear's open records).
                let hr = call_slot_i32_0(self.allocator as ComObject, ALLOCATOR_RESET);
                if hr != S_OK {
                    return Err(RenderError::from(
                        "fill_texture_2d: the allocator's reset failed",
                    ));
                }
                let hr = call_slot_i32_2(
                    self.command_list as ComObject,
                    COMMAND_LIST_RESET,
                    self.allocator,
                    self.tri_pipeline,
                );
                if hr != S_OK {
                    return Err(RenderError::from(
                        "fill_texture_2d: the command list's reset failed",
                    ));
                }
            }
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RESOURCE_BARRIER,
                1 as *mut c_void,
                &to_copy_dest as *const D3d12ResourceBarrier as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_6(
                self.command_list as ComObject,
                COMMAND_LIST_COPY_TEXTURE_REGION,
                &dst as *const D3d12TextureCopyLocation as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                &src as *const D3d12TextureCopyLocation as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RESOURCE_BARRIER,
                1 as *mut c_void,
                &to_shader_read as *const D3d12ResourceBarrier as *mut c_void,
                std::ptr::null_mut(),
            );
        }
        self.textures_2d[index].state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        if self.frame_open {
            // The open frame's execution submits the fill's commands; the
            // bookkeeping stays.
            return Ok(());
        }
        // The one-off frame executes: the Close, the ExecuteCommandLists,
        // the queue's Signal and the wait (the same sequence the
        // readback's copies follow).
        let hr = unsafe { call_slot_i32_0(self.command_list as ComObject, COMMAND_LIST_CLOSE) };
        if hr != S_OK {
            return Err(RenderError::from(
                "fill_texture_2d: the command list's close failed",
            ));
        }
        let lists = [self.command_list];
        unsafe {
            call_slot_i32_3(
                self.queue as ComObject,
                QUEUE_EXECUTE_COMMAND_LISTS,
                1 as *mut c_void,
                lists.as_ptr() as *const c_void as *mut c_void,
                std::ptr::null_mut(),
            );
            self.fence_value += 1;
            call_slot_i32_4(
                self.queue as ComObject,
                QUEUE_SIGNAL,
                self.fence,
                self.fence_value as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.fence as ComObject,
                FENCE_SET_EVENT_ON_COMPLETION,
                self.fence_value as *mut c_void,
                self.fence_event,
                std::ptr::null_mut(),
            );
            self.loader
                .wait_for_single_object
                .expect("the loader resolved WaitForSingleObject")(
                self.fence_event, 5000
            );
        }
        self.frame_open = false;
        Ok(())
    }

    fn read_texture_2d(&self, view: &TextureView, data: &mut [u8]) -> Result<bool, RenderError> {
        let index = self
            .textures_2d
            .iter()
            .position(|t| Arc::ptr_eq(&t.inner, &view.inner));
        let index = match index {
            Some(index) => index,
            None => {
                return Err(RenderError::from(
                    "read_texture_2d: the texture is not owned by this \
                     backend",
                ));
            }
        };
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
        // other backends record).
        Ok(false)
    }

    fn set_camera(&mut self, camera: CameraParams) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.camera = camera;
    }

    fn create_texture_3d(&mut self, desc: &Texture3DDesc) -> Result<TextureView, RenderError> {
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "create_texture_3d: the backend is shut down",
            ));
        }
        desc.validate()?;
        let resource = unsafe {
            self_create_texture_3d(
                self.device,
                self.to_d3d_format(desc.format),
                desc.width,
                desc.height,
                desc.depth,
                0,
                D3D12_RESOURCE_STATE_COMMON,
            )?
        };
        let view = D3d12Texture2D {
            inner: Arc::new(Mutex::new(TextureViewInner {
                width: desc.width,
                height: desc.height,
                depth: desc.depth,
                format: desc.format,
                is_depth: false,
                released: false,
            })),
            resource,
            released: false,
            state: D3D12_RESOURCE_STATE_COMMON,
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

    fn fill_texture_3d(&mut self, view: &TextureView, data: &[u8]) -> Result<(), RenderError> {
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
        // The upload path: the same one-off staging copy the 2D fill
        // follows, with the footprint's depth carrying the volume's
        // layers (the row pitch covers one layer's slice).
        let bytes = bytes_per_pixel(format);
        let row_pitch = d3d::texture_row_pitch(width, bytes);
        let slice_pitch = row_pitch as u64 * height as u64;
        let buffer_size = slice_pitch * depth as u64;
        let staging = unsafe {
            self_create_buffer(
                self.device,
                buffer_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?
        };
        unsafe {
            let mut map: *mut c_void = std::ptr::null_mut();
            let hr = call_slot_i32_4(
                staging as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range { begin: 0, end: 0 } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            );
            if hr != S_OK || map.is_null() {
                return Err(RenderError::from(
                    "fill_texture_3d: the staging buffer's map failed",
                ));
            }
            let map = map as *mut u8;
            let slice = width as usize * height as usize * bytes as usize;
            for layer in 0..depth as usize {
                // The copy per row: the staging buffer's rows sit at the
                // 256-byte pitch (the same per-row rule the 2D fill
                // follows), never contiguously.
                let packed = width as usize * bytes as usize;
                for row in 0..height as usize {
                    let src = data
                        .as_ptr()
                        .add(layer * slice + row * packed);
                    let dst = map
                        .add(layer as usize * slice_pitch as usize + row * row_pitch as usize);
                    std::ptr::copy_nonoverlapping(src, dst, packed);
                }
            }
            call_slot_i32_2(
                staging as ComObject,
                RESOURCE_UNMAP,
                0 as *mut c_void,
                std::ptr::null_mut(),
            );
        }
        let resource = self.textures_3d[index].resource;
        let current_state = self.textures_3d[index].state;
        let to_copy_dest = D3d12ResourceBarrier {
            type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            transition: D3d12ResourceTransitionBarrier {
                resource,
                subresource: 0,
                state_before: current_state,
                state_after: D3D12_RESOURCE_STATE_COPY_DEST,
            },
        };
        let to_shader_read = D3d12ResourceBarrier {
            type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            transition: D3d12ResourceTransitionBarrier {
                resource,
                subresource: 0,
                state_before: D3D12_RESOURCE_STATE_COPY_DEST,
                state_after: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            },
        };
        let dst = D3d12TextureCopyLocation {
            resource,
            type_: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            placed_footprint: D3d12PlacedFootprint {
                offset: 0,
                footprint: D3d12SubresourceFootprint {
                    format: 0,
                    width: 0,
                    height: 0,
                    depth: 0,
                    row_pitch: 0,
                },
            },
        };
        let src = D3d12TextureCopyLocation {
            resource: staging,
            type_: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            placed_footprint: D3d12PlacedFootprint {
                offset: 0,
                footprint: D3d12SubresourceFootprint {
                    format: self.to_d3d_format(format),
                    width,
                    height,
                    depth,
                    row_pitch,
                },
            },
        };
        unsafe {
            if !self.frame_open {
                // The frame's command list opens: the allocator and the
                // list reset (the same rule the 2D fill records).
                let hr = call_slot_i32_0(self.allocator as ComObject, ALLOCATOR_RESET);
                if hr != S_OK {
                    return Err(RenderError::from(
                        "fill_texture_3d: the allocator's reset failed",
                    ));
                }
                let hr = call_slot_i32_2(
                    self.command_list as ComObject,
                    COMMAND_LIST_RESET,
                    self.allocator,
                    self.tri_pipeline,
                );
                if hr != S_OK {
                    return Err(RenderError::from(
                        "fill_texture_3d: the command list's reset failed",
                    ));
                }
            }
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RESOURCE_BARRIER,
                1 as *mut c_void,
                &to_copy_dest as *const D3d12ResourceBarrier as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_6(
                self.command_list as ComObject,
                COMMAND_LIST_COPY_TEXTURE_REGION,
                &dst as *const D3d12TextureCopyLocation as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                &src as *const D3d12TextureCopyLocation as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RESOURCE_BARRIER,
                1 as *mut c_void,
                &to_shader_read as *const D3d12ResourceBarrier as *mut c_void,
                std::ptr::null_mut(),
            );
        }
        self.textures_3d[index].state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        if self.frame_open {
            // The open frame's execution submits the fill's commands; the
            // bookkeeping stays (the same rule the 2D fill records).
            return Ok(());
        }
        // The one-off frame executes: the Close, the ExecuteCommandLists,
        // the queue's Signal and the wait (the same sequence the 2D fill
        // records).
        let hr = unsafe { call_slot_i32_0(self.command_list as ComObject, COMMAND_LIST_CLOSE) };
        if hr != S_OK {
            return Err(RenderError::from(
                "fill_texture_3d: the command list's close failed",
            ));
        }
        let lists = [self.command_list];
        unsafe {
            call_slot_i32_3(
                self.queue as ComObject,
                QUEUE_EXECUTE_COMMAND_LISTS,
                1 as *mut c_void,
                lists.as_ptr() as *const c_void as *mut c_void,
                std::ptr::null_mut(),
            );
            self.fence_value += 1;
            call_slot_i32_4(
                self.queue as ComObject,
                QUEUE_SIGNAL,
                self.fence,
                self.fence_value as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.fence as ComObject,
                FENCE_SET_EVENT_ON_COMPLETION,
                self.fence_value as *mut c_void,
                self.fence_event,
                std::ptr::null_mut(),
            );
            self.loader
                .wait_for_single_object
                .expect("the loader resolved WaitForSingleObject")(
                self.fence_event, 5000
            );
        }
        self.frame_open = false;
        Ok(())
    }

    fn read_texture_3d(&mut self, view: &TextureView, data: &mut [u8]) -> Result<bool, RenderError> {
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
        // The readback: the resource's pixels copy into a one-off
        // READBACK buffer (the same one-off command list the fill's
        // copies follow), with the footprint's depth carrying the
        // volume's layers (the row pitch covers one layer's slice).
        let bytes = bytes_per_pixel(view.inner.lock().unwrap().format) as usize;
        let (width, height, depth, format) = {
            let inner = view.inner.lock().unwrap();
            (inner.width, inner.height, inner.depth, inner.format)
        };
        let row_pitch = d3d::texture_row_pitch(width, bytes as u32);
        let slice_pitch = row_pitch as u64 * height as u64;
        let buffer_size = slice_pitch * depth as u64;
        let readback = unsafe {
            self_create_buffer(
                self.device,
                buffer_size,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?
        };
        let resource = self.textures_3d[index].resource;
        let current_state = self.textures_3d[index].state;
        let to_copy_source = D3d12ResourceBarrier {
            type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            transition: D3d12ResourceTransitionBarrier {
                resource,
                subresource: 0,
                state_before: current_state,
                state_after: D3D12_RESOURCE_STATE_COPY_SOURCE,
            },
        };
        let to_shader_read = D3d12ResourceBarrier {
            type_: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            transition: D3d12ResourceTransitionBarrier {
                resource,
                subresource: 0,
                state_before: D3D12_RESOURCE_STATE_COPY_SOURCE,
                state_after: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            },
        };
        let src = D3d12TextureCopyLocation {
            resource,
            type_: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            placed_footprint: D3d12PlacedFootprint {
                offset: 0,
                footprint: D3d12SubresourceFootprint {
                    format: 0,
                    width: 0,
                    height: 0,
                    depth: 0,
                    row_pitch: 0,
                },
            },
        };
        let dst = D3d12TextureCopyLocation {
            resource: readback,
            type_: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            placed_footprint: D3d12PlacedFootprint {
                offset: 0,
                footprint: D3d12SubresourceFootprint {
                    format: self.to_d3d_format(format),
                    width,
                    height,
                    depth,
                    row_pitch,
                },
            },
        };
        let hr = unsafe { call_slot_i32_0(self.allocator as ComObject, ALLOCATOR_RESET) };
        if hr != S_OK {
            return Err(RenderError::from(
                "read_texture_3d: the allocator's reset failed",
            ));
        }
        let hr = unsafe {
            call_slot_i32_2(
                self.command_list as ComObject,
                COMMAND_LIST_RESET,
                self.allocator,
                self.tri_pipeline,
            )
        };
        if hr != S_OK {
            return Err(RenderError::from(
                "read_texture_3d: the command list's reset failed",
            ));
        }
        unsafe {
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RESOURCE_BARRIER,
                1 as *mut c_void,
                &to_copy_source as *const D3d12ResourceBarrier as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_6(
                self.command_list as ComObject,
                COMMAND_LIST_COPY_TEXTURE_REGION,
                &dst as *const D3d12TextureCopyLocation as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
                &src as *const D3d12TextureCopyLocation as *mut c_void,
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.command_list as ComObject,
                COMMAND_LIST_RESOURCE_BARRIER,
                1 as *mut c_void,
                &to_shader_read as *const D3d12ResourceBarrier as *mut c_void,
                std::ptr::null_mut(),
            );
        }
        self.textures_3d[index].state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let hr = unsafe { call_slot_i32_0(self.command_list as ComObject, COMMAND_LIST_CLOSE) };
        if hr != S_OK {
            return Err(RenderError::from(
                "read_texture_3d: the command list's close failed",
            ));
        }
        let lists = [self.command_list];
        unsafe {
            call_slot_i32_3(
                self.queue as ComObject,
                QUEUE_EXECUTE_COMMAND_LISTS,
                1 as *mut c_void,
                lists.as_ptr() as *const c_void as *mut c_void,
                std::ptr::null_mut(),
            );
            self.fence_value += 1;
            call_slot_i32_4(
                self.queue as ComObject,
                QUEUE_SIGNAL,
                self.fence,
                self.fence_value as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            call_slot_i32_3(
                self.fence as ComObject,
                FENCE_SET_EVENT_ON_COMPLETION,
                self.fence_value as *mut c_void,
                self.fence_event,
                std::ptr::null_mut(),
            );
            self.loader
                .wait_for_single_object
                .expect("the loader resolved WaitForSingleObject")(
                self.fence_event, 5000
            );
        }
        // The mapped READBACK buffer carries the volume's layers (the
        // same row-pitch slice the upload's staging carries).
        let mut map: *mut c_void = std::ptr::null_mut();
        let mapped = unsafe {
            call_slot_i32_4(
                readback as ComObject,
                RESOURCE_MAP,
                0 as *mut c_void,
                &D3d12Range {
                    begin: 0,
                    end: buffer_size,
                } as *const D3d12Range as *mut c_void,
                &mut map as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        if mapped != S_OK || map.is_null() {
            return Err(RenderError::from(
                "read_texture_3d: the readback buffer's map failed",
            ));
        }
        let map = map as *const u8;
        let slice = width as usize * height as usize * bytes;
        let packed = width as usize * bytes;
        unsafe {
            for layer in 0..depth as usize {
                let dst = data.as_mut_ptr().add(layer * slice);
                let src = map.add(layer * slice_pitch as usize);
                for row in 0..height as usize {
                    std::ptr::copy_nonoverlapping(
                        src.add(row * row_pitch as usize),
                        dst.add(row * packed),
                        packed,
                    );
                }
            }
        }
        unsafe {
            call_slot_i32_3(
                readback as ComObject,
                RESOURCE_UNMAP,
                0 as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
        }
        Ok(true)
    }

    /// The M8 light volume: the backend (re)creates the 3D texture the
    /// volume's packed radiance fills (the dims z * 6 layers, the face
    /// `f` of the cell `(x, y, z)` at the layer `z * 6 + f`) and keeps
    /// the uniform block's values the gbuffer pass's Volume block reads.

    fn set_light_volume(&mut self, volume: &LightVolume) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        if volume.dims[0] == 0 || volume.dims[1] == 0 || volume.dims[2] == 0 {
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
        // The old volume's texture releases (the one-volume rule the
        // contract's set_light_volume carries).
        if let Some(old) = self.light_volume.take() {
            self.release_texture_3d(&old);
        }
        self.light_volume = Some(view);
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
        // pre-present rule the other backends record).
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
