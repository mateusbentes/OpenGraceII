//! The Vulkan render module's own entry-point loader.
//!
//! RULE (lesson 3): every constant, structure and function signature the
//! engine uses is declared here, so no generated header and no third-party
//! Vulkan binding joins the tree, and the platform-neutral contract never
//! includes it. Entry points are resolved at runtime through the resolve
//! callback the backend passes in, so this file needs no windowing system.

use std::os::raw::{c_char, c_void};

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

pub type VkInstance = *mut c_void;
pub type VkPhysicalDevice = *mut c_void;
pub type VkDevice = *mut c_void;
pub type VkQueue = *mut c_void;
pub type VkCommandBuffer = *mut c_void;
pub type VkBuffer = *mut c_void;
pub type VkImage = *mut c_void;
pub type VkImageView = *mut c_void;
pub type VkDeviceMemory = *mut c_void;
pub type VkCommandPool = *mut c_void;
pub type VkFence = *mut c_void;
pub type VkSemaphore = *mut c_void;
pub type VkRenderPass = *mut c_void;
pub type VkFramebuffer = *mut c_void;
pub type VkPipeline = *mut c_void;
pub type VkPipelineLayout = *mut c_void;
pub type VkPipelineCache = *mut c_void;
pub type VkShaderModule = *mut c_void;
pub type VkDescriptorSet = *mut c_void;
pub type VkDescriptorSetLayout = *mut c_void;
pub type VkDescriptorPool = *mut c_void;
pub type VkSampler = *mut c_void;

// The spec's VkDeviceSize: every size and offset field of the memory, the
// buffer and the copy structures is 64-bit. A 32-bit field in one of those
// positions shifts every following member and the driver reads past the
// struct's end (the memory-allocate segfault that lesson 3's fix on real
// hardware addressed).
pub type VkDeviceSize = u64;

// ---------------------------------------------------------------------------
// Fixed-size constants
// ---------------------------------------------------------------------------

pub const MAX_EXTENSION_NAME_SIZE: usize = 256;
pub const QUEUE_FAMILY_IGNORED: u32 = u32::MAX;
pub const WHOLE_SIZE: u64 = u64::MAX;

// ---------------------------------------------------------------------------
// Enumerations (the published API's values)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Format {
    R8Unorm = 9,
    R32Sfloat = 100,
    Rg32Sfloat = 103,
    Rgba32Sfloat = 109,
    R8g8b8a8Unorm = 37,
    B8g8r8a8Unorm = 44,
    R32G32B32Sfloat = 106,
    R16g16b16a16Sfloat = 97,
    D24UnormS8Uint = 129,
    D32Sfloat = 126,
    D16Unorm = 124,
    R32Uint = 98,
    R32Sint = 99,
    Rg32Sint = 102,
    Rgba32Sint = 108,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ImageLayout {
    Undefined = 0,
    General = 1,
    ColorAttachmentOptimal = 2,
    DepthStencilAttachmentOptimal = 3,
    DepthStencilReadOnlyOptimal = 4,
    ShaderReadOnlyOptimal = 5,
    TransferSrcOptimal = 6,
    TransferDstOptimal = 7,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ImageAspect {
    Color = 0x1,
    Depth = 0x2,
    Stencil = 0x4,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BufferUsage {
    TransferSrc = 0x1,
    TransferDst = 0x2,
    UniformBuffer = 0x10,
    VertexBuffer = 0x80,
    IndexBuffer = 0x40,
}

impl std::ops::BitOr for BufferUsage {
    type Output = u32;
    fn bitor(self, rhs: Self) -> u32 {
        self as u32 | rhs as u32
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ImageUsage {
    TransferSrc = 0x1,
    TransferDst = 0x2,
    Sampled = 0x4,
    ColorAttachment = 0x10,
    DepthStencilAttachment = 0x20,
}

impl std::ops::BitOr for ImageUsage {
    type Output = u32;
    fn bitor(self, rhs: Self) -> u32 {
        self as u32 | rhs as u32
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MemoryProperty {
    DeviceLocal = 0x1,
    HostVisible = 0x2,
    HostCoherent = 0x4,
}

impl std::ops::BitOr for MemoryProperty {
    type Output = u32;
    fn bitor(self, rhs: Self) -> u32 {
        self as u32 | rhs as u32
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PipelineStage {
    TopOfPipe = 0x1,
    VertexInput = 0x4,
    FragmentShader = 0x80,
    EarlyFragmentTests = 0x100,
    LateFragmentTests = 0x200,
    ColorAttachmentOutput = 0x400,
    Transfer = 0x1000,
    BottomOfPipe = 0x2000,
    AllCommands = 0x10000,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Access {
    ShaderRead = 0x20,
    ShaderWrite = 0x40,
    ColorAttachmentRead = 0x80,
    ColorAttachmentWrite = 0x100,
    DepthStencilAttachmentRead = 0x200,
    DepthStencilAttachmentWrite = 0x400,
    TransferRead = 0x800,
    TransferWrite = 0x1000,
    MemoryRead = 0x8000,
    MemoryWrite = 0x10000,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum StructureType {
    ApplicationInfo = 0,
    InstanceCreateInfo = 1,
    DeviceQueueCreateInfo = 2,
    DeviceCreateInfo = 3,
    SubmitInfo = 4,
    MemoryAllocateInfo = 5,
    MappedMemoryRange = 6,
    FenceCreateInfo = 8,
    SemaphoreCreateInfo = 9,
    BufferCreateInfo = 12,
    ImageCreateInfo = 14,
    ImageViewCreateInfo = 15,
    ShaderModuleCreateInfo = 16,
    PipelineShaderStageCreateInfo = 18,
    PipelineVertexInputStateCreateInfo = 19,
    PipelineInputAssemblyStateCreateInfo = 20,
    PipelineViewportStateCreateInfo = 22,
    PipelineRasterizationStateCreateInfo = 23,
    PipelineMultisampleStateCreateInfo = 24,
    PipelineDepthStencilStateCreateInfo = 25,
    PipelineColorBlendStateCreateInfo = 26,
    PipelineDynamicStateCreateInfo = 27,
    GraphicsPipelineCreateInfo = 28,
    PipelineLayoutCreateInfo = 30,
    SamplerCreateInfo = 31,
    DescriptorSetLayoutCreateInfo = 32,
    DescriptorPoolCreateInfo = 33,
    DescriptorSetAllocateInfo = 34,
    WriteDescriptorSet = 35,
    RenderPassCreateInfo = 38,
    CommandPoolCreateInfo = 39,
    CommandBufferAllocateInfo = 40,
    CommandBufferBeginInfo = 42,
    RenderPassBeginInfo = 43,
    FramebufferCreateInfo = 37,
    BufferMemoryBarrier = 44,
    ImageMemoryBarrier = 45,
    MemoryBarrier = 46,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ShaderStage {
    Vertex = 0x1,
    Fragment = 0x10,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BlendFactor {
    Zero = 0,
    One = 1,
    SrcAlpha = 6,
    OneMinusSrcAlpha = 7,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BlendOp {
    Add = 0,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CullMode {
    None = 0,
    Back = 0x2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FrontFace {
    CounterClockwise = 0,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SampleCount {
    One = 0x1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CompareOp {
    Less = 1,
    Always = 7,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum StencilOp {
    Keep = 0,
    Zero = 1,
    Replace = 2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AttachmentLoadOp {
    Load = 0,
    Clear = 1,
    DontCare = 2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AttachmentStoreOp {
    Store = 0,
    DontCare = 1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PipelineBindPoint {
    Graphics = 0,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DescriptorType {
    CombinedImageSampler = 1,
    UniformBuffer = 6,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IndexType {
    Uint32 = 1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum VertexInputRate {
    Vertex = 0,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CommandBufferUsage {
    OneTimeSubmit = 0x1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CommandPoolCreate {
    ResetCommandBuffer = 0x2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FenceCreate {
    Signaled = 0x1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CommandPoolReset {
    ReleaseResources = 0x1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Result {
    Success = 0,
    NotReady = 2,
    ErrorInitializationFailed = -3,
    ErrorExtensionNotPresent = -7,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Filter {
    Nearest = 0,
    Linear = 1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SamplerAddressMode {
    ClampToEdge = 2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ComponentSwizzle {
    Identity = 0,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CommandBufferLevel {
    Primary = 0,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DynamicState {
    Viewport = 0,
    Scissor = 1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PolygonMode {
    Fill = 0,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SharingMode {
    Exclusive = 0,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BorderColor {
    FloatTransparentBlack = 0,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FormatFeature {
    SampledImage = 0x1,
    DepthStencilAttachment = 0x2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ImageTiling {
    Optimal = 0,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ImageType {
    // The spec's values (the VK_IMAGE_TYPE_* constants): 1D is 0, 2D
    // is 1 and 3D is 2. The earlier Type3D = 3 was invalid — the
    // driver clamped the unknown type to 2D and the 3D texture carried
    // a single layer (the M8 debug's every-copy-lands-in-slice-0 bug).
    Type1D = 0,
    Type2D = 1,
    Type3D = 2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PrimitiveTopology {
    TriangleList = 3,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum StencilFace {
    FrontAndBack = 0x3,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QueueFlags {
    Graphics = 0x1,
    Compute = 0x2,
    Transfer = 0x4,
}

// ---------------------------------------------------------------------------
// Structures (the published API's field layout). Every struct is
// #[repr(C)] and the field order is the published one, so the driver reads
// exactly what the engine writes.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkApplicationInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub p_application_name: *const c_char,
    pub application_version: u32,
    pub p_engine_name: *const c_char,
    pub engine_version: u32,
    pub api_version: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkInstanceCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_application_info: *const c_void,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDeviceQueueCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
    pub queue_count: u32,
    pub p_queue_priorities: *const f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDeviceCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_create_info_count: u32,
    pub p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
    pub p_enabled_features: *const c_void,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkSubmitInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub wait_semaphore_count: u32,
    pub p_wait_semaphores: *const VkSemaphore,
    pub p_wait_dst_stage_mask: *const u32,
    pub command_buffer_count: u32,
    pub p_command_buffers: *const VkCommandBuffer,
    pub signal_semaphore_count: u32,
    pub p_signal_semaphores: *const VkSemaphore,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkMemoryAllocateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub allocation_size: VkDeviceSize,
    pub memory_type_index: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkMappedMemoryRange {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub memory: VkDeviceMemory,
    pub offset: VkDeviceSize,
    pub size: VkDeviceSize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkFenceCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkBufferCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub size: VkDeviceSize,
    pub usage: u32,
    pub sharing_mode: SharingMode,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkImageCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub image_type: ImageType,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: SampleCount,
    pub tiling: ImageTiling,
    pub usage: u32,
    pub sharing_mode: SharingMode,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
    pub initial_layout: ImageLayout,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkComponentMapping {
    pub r: ComponentSwizzle,
    pub g: ComponentSwizzle,
    pub b: ComponentSwizzle,
    pub a: ComponentSwizzle,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkImageSubresourceRange {
    pub aspect_mask: u32,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkImageSubresourceLayers {
    pub aspect_mask: u32,
    pub mip_level: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkOffset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkExtent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkBufferCopy {
    pub src_offset: VkDeviceSize,
    pub dst_offset: VkDeviceSize,
    pub size: VkDeviceSize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkBufferImageCopy {
    pub buffer_offset: VkDeviceSize,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_subresource: VkImageSubresourceLayers,
    pub image_offset: VkOffset3D,
    pub image_extent: VkExtent3D,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkImageViewCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub image: VkImage,
    pub view_type: ImageType,
    pub format: u32,
    pub components: VkComponentMapping,
    pub subresource_range: VkImageSubresourceRange,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkShaderModuleCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub code_size: usize,
    pub p_code: *const u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkViewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkOffset2D {
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkExtent2D {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkRect2D {
    pub offset: VkOffset2D,
    pub extent: VkExtent2D,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineShaderStageCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: ShaderStage,
    pub module: VkShaderModule,
    pub p_name: *const c_char,
    pub p_specialization_info: *const c_void,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkVertexInputBindingDescription {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: VertexInputRate,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkVertexInputAttributeDescription {
    pub location: u32,
    pub binding: u32,
    pub format: u32,
    pub offset: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineVertexInputStateCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub vertex_binding_description_count: u32,
    pub p_vertex_binding_descriptions: *const VkVertexInputBindingDescription,
    pub vertex_attribute_description_count: u32,
    pub p_vertex_attribute_descriptions:
        *const VkVertexInputAttributeDescription,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineInputAssemblyStateCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub topology: PrimitiveTopology,
    pub primitive_restart_enable: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineViewportStateCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub viewport_count: u32,
    pub p_viewports: *const VkViewport,
    pub scissor_count: u32,
    pub p_scissors: *const VkRect2D,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineRasterizationStateCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub depth_clamp_enable: u32,
    pub rasterizer_discard_enable: u32,
    pub polygon_mode: PolygonMode,
    pub cull_mode: u32,
    pub front_face: FrontFace,
    pub depth_bias_enable: u32,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub line_width: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineMultisampleStateCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub rasterization_samples: SampleCount,
    pub sample_shading_enable: u32,
    pub min_sample_shading: f32,
    pub p_sample_mask: *const u32,
    pub alpha_to_coverage_enable: u32,
    pub alpha_to_one_enable: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkStencilOpState {
    pub fail_op: StencilOp,
    pub pass_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub compare_op: CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineDepthStencilStateCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub depth_test_enable: u32,
    pub depth_write_enable: u32,
    pub depth_compare_op: CompareOp,
    pub depth_bounds_test_enable: u32,
    pub stencil_test_enable: u32,
    pub front: VkStencilOpState,
    pub back: VkStencilOpState,
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineColorBlendAttachmentState {
    pub blend_enable: u32,
    pub src_color_blend_factor: BlendFactor,
    pub dst_color_blend_factor: BlendFactor,
    pub color_blend_op: BlendOp,
    pub src_alpha_blend_factor: BlendFactor,
    pub dst_alpha_blend_factor: BlendFactor,
    pub alpha_blend_op: BlendOp,
    pub color_write_mask: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineColorBlendStateCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub logic_op_enable: u32,
    pub logic_op: u32,
    pub attachment_count: u32,
    pub p_attachments: *const VkPipelineColorBlendAttachmentState,
    pub blend_constants: [f32; 4],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineDynamicStateCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub dynamic_state_count: u32,
    pub p_dynamic_states: *const DynamicState,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkGraphicsPipelineCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage_count: u32,
    pub p_stages: *const VkPipelineShaderStageCreateInfo,
    pub p_vertex_input_state: *const VkPipelineVertexInputStateCreateInfo,
    pub p_input_assembly_state:
        *const VkPipelineInputAssemblyStateCreateInfo,
    pub p_tessellation_state: *const c_void,
    pub p_viewport_state: *const VkPipelineViewportStateCreateInfo,
    pub p_rasterization_state: *const VkPipelineRasterizationStateCreateInfo,
    pub p_multisample_state: *const VkPipelineMultisampleStateCreateInfo,
    pub p_depth_stencil_state:
        *const VkPipelineDepthStencilStateCreateInfo,
    pub p_color_blend_state: *const VkPipelineColorBlendStateCreateInfo,
    pub p_dynamic_state: *const VkPipelineDynamicStateCreateInfo,
    pub layout: VkPipelineLayout,
    pub render_pass: VkRenderPass,
    pub subpass: u32,
    pub base_pipeline_handle: VkPipeline,
    pub base_pipeline_index: i32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineLayoutCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub set_layout_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
    pub push_constant_range_count: u32,
    pub p_push_constant_ranges: *const c_void,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkAttachmentDescription {
    pub flags: u32,
    pub format: u32,
    pub samples: SampleCount,
    pub load_op: AttachmentLoadOp,
    pub store_op: AttachmentStoreOp,
    pub stencil_load_op: AttachmentLoadOp,
    pub stencil_store_op: AttachmentStoreOp,
    pub initial_layout: ImageLayout,
    pub final_layout: ImageLayout,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkAttachmentReference {
    pub attachment: u32,
    pub layout: ImageLayout,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkSubpassDescription {
    pub flags: u32,
    pub pipeline_bind_point: PipelineBindPoint,
    pub input_attachment_count: u32,
    pub p_input_attachments: *const VkAttachmentReference,
    pub color_attachment_count: u32,
    pub p_color_attachments: *const VkAttachmentReference,
    pub p_resolve_attachments: *const VkAttachmentReference,
    pub p_depth_stencil_attachment: *const VkAttachmentReference,
    pub preserve_attachment_count: u32,
    pub p_preserve_attachments: *const u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkSubpassDependency {
    pub src_subpass: u32,
    pub dst_subpass: u32,
    pub src_stage_mask: u32,
    pub dst_stage_mask: u32,
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
    pub dependency_flags: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkRenderPassCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub attachment_count: u32,
    pub p_attachments: *const VkAttachmentDescription,
    pub subpass_count: u32,
    pub p_subpasses: *const VkSubpassDescription,
    pub dependency_count: u32,
    pub p_dependencies: *const VkSubpassDependency,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkFramebufferCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub render_pass: VkRenderPass,
    pub attachment_count: u32,
    pub p_attachments: *const VkImageView,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkCommandPoolCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkCommandBufferAllocateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub command_pool: VkCommandPool,
    pub level: CommandBufferLevel,
    pub command_buffer_count: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkCommandBufferBeginInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_inheritance_info: *const c_void,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkClearDepthStencilValue {
    pub depth: f32,
    pub stencil: u32,
}

/// The published layout: VkClearValue is a 16-byte union (the colour's
/// four floats overlap the depth/stencil pair), not a struct — a struct
/// would double the stride every pClearValues array reads.
#[derive(Clone, Copy)]
#[repr(C)]
pub union VkClearValue {
    pub color: [f32; 4],
    pub depth_stencil: VkClearDepthStencilValue,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkRenderPassBeginInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub render_pass: VkRenderPass,
    pub framebuffer: VkFramebuffer,
    pub render_area: VkRect2D,
    pub clear_value_count: u32,
    pub p_clear_values: *const VkClearValue,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: DescriptorType,
    pub descriptor_count: u32,
    /// The stage bitmask (the OR of the stage flags the descriptor is
    /// visible in, the raw Vulkan value).
    pub stage_flags: u32,
    pub p_immutable_samplers: *const VkSampler,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorSetLayoutCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub binding_count: u32,
    pub p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorPoolSize {
    pub descriptor_type: DescriptorType,
    pub descriptor_count: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorPoolCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub max_sets: u32,
    pub pool_size_count: u32,
    pub p_pool_sizes: *const VkDescriptorPoolSize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorSetAllocateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub descriptor_pool: VkDescriptorPool,
    pub descriptor_set_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorBufferInfo {
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub range: VkDeviceSize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorImageInfo {
    pub sampler: VkSampler,
    pub image_view: VkImageView,
    pub image_layout: ImageLayout,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkWriteDescriptorSet {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub dst_set: VkDescriptorSet,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    pub descriptor_count: u32,
    pub descriptor_type: DescriptorType,
    pub p_image_info: *const VkDescriptorImageInfo,
    pub p_buffer_info: *const VkDescriptorBufferInfo,
    pub p_texel_buffer_view: *const c_void,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkSamplerCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mip_map_mode: u32,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub mip_lod_bias: f32,
    pub anisotropy_enable: u32,
    pub max_anisotropy: f32,
    pub compare_enable: u32,
    pub compare_op: CompareOp,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: BorderColor,
    pub unnormalized_coordinates: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkBufferMemoryBarrier {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub size: VkDeviceSize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkImageMemoryBarrier {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
    pub old_layout: ImageLayout,
    pub new_layout: ImageLayout,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub image: VkImage,
    pub subresource_range: VkImageSubresourceRange,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkMemoryBarrier {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPhysicalDeviceLimits {
    pub max_image_dimension1_d: u32,
    pub max_image_dimension2_d: u32,
    pub max_image_dimension3_d: u32,
    pub max_image_dimension_cube: u32,
    pub max_framebuffer_width: u32,
    pub max_framebuffer_height: u32,
    pub max_color_attachments: u32,
    pub max_uniform_buffer_range: u32,
    pub max_push_constants_size: u32,
    pub max_memory_allocation_count: u32,
    pub max_descriptor_set_samplers: u32,
    pub max_descriptor_set_uniform_buffers: u32,
    pub min_uniform_buffer_offset_alignment: u32,
    pub max_per_stage_descriptor_samplers: u32,
    pub max_per_stage_descriptor_uniform_buffers: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPhysicalDeviceProperties {
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub device_name: [c_char; MAX_EXTENSION_NAME_SIZE],
    pub pipeline_cache_uuid: [u8; 16],
    pub limits: VkPhysicalDeviceLimits,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkQueueFamilyProperties {
    pub queue_flags: u32,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: VkExtent2D,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkMemoryType {
    pub property_flags: u32,
    pub heap_index: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkMemoryHeap {
    pub size: u64,
    pub flags: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPhysicalDeviceMemoryProperties {
    pub memory_type_count: u32,
    pub memory_types: [VkMemoryType; 32],
    pub memory_heap_count: u32,
    pub memory_heaps: [VkMemoryHeap; 16],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkMemoryRequirements {
    pub size: VkDeviceSize,
    pub alignment: VkDeviceSize,
    pub memory_type_bits: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkExtensionProperties {
    pub extension_name: [c_char; MAX_EXTENSION_NAME_SIZE],
    pub spec_version: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkFormatProperties {
    pub linear_tiling_features: u32,
    pub optimal_tiling_features: u32,
    pub buffer_features: u32,
}

// ---------------------------------------------------------------------------
// Function-pointer signatures (the published API's symbols)
// ---------------------------------------------------------------------------

pub type ResolveFn = unsafe extern "C" fn(VkInstance, *const c_char)
    -> *mut c_void;

pub type FnGetInstanceProcAddr = unsafe extern "system" fn(
    VkInstance,
    *const c_char,
) -> *mut c_void;
pub type FnGetDeviceProcAddr = unsafe extern "system" fn(
    VkDevice,
    *const c_char,
) -> *mut c_void;
pub type FnEnumerateInstanceExtensionProperties =
    unsafe extern "system" fn(
        *const c_char,
        *mut u32,
        *mut VkExtensionProperties,
    ) -> Result;
pub type FnCreateInstance = unsafe extern "system" fn(
    *const VkInstanceCreateInfo,
    *const c_void,
    *mut VkInstance,
) -> Result;
pub type FnDestroyInstance =
    unsafe extern "system" fn(VkInstance, *const c_void);
pub type FnEnumeratePhysicalDevices = unsafe extern "system" fn(
    VkInstance,
    *mut u32,
    *mut VkPhysicalDevice,
) -> Result;
pub type FnGetPhysicalDeviceProperties = unsafe extern "system" fn(
    VkPhysicalDevice,
    *mut VkPhysicalDeviceProperties,
);
pub type FnGetPhysicalDeviceFormatProperties =
    unsafe extern "system" fn(VkPhysicalDevice, Format, *mut VkFormatProperties);
pub type FnGetPhysicalDeviceFeatures =
    unsafe extern "system" fn(VkPhysicalDevice, *mut c_void);
pub type FnGetPhysicalDeviceMemoryProperties =
    unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties);
pub type FnGetPhysicalDeviceQueueFamilyProperties =
    unsafe extern "system" fn(
        VkPhysicalDevice,
        *mut u32,
        *mut VkQueueFamilyProperties,
    );
pub type FnEnumerateDeviceExtensionProperties =
    unsafe extern "system" fn(
        VkPhysicalDevice,
        *const c_char,
        *mut u32,
        *mut VkExtensionProperties,
    ) -> Result;
pub type FnCreateDevice = unsafe extern "system" fn(
    VkPhysicalDevice,
    *const VkDeviceCreateInfo,
    *const c_void,
    *mut VkDevice,
) -> Result;
pub type FnDestroyDevice =
    unsafe extern "system" fn(VkDevice, *const c_void);
pub type FnGetDeviceQueue = unsafe extern "system" fn(
    VkDevice,
    u32,
    u32,
    *mut VkQueue,
);
pub type FnCreateBuffer = unsafe extern "system" fn(
    VkDevice,
    *const VkBufferCreateInfo,
    *const c_void,
    *mut VkBuffer,
) -> Result;
pub type FnDestroyBuffer =
    unsafe extern "system" fn(VkDevice, VkBuffer, *const c_void);
pub type FnAllocateMemory = unsafe extern "system" fn(
    VkDevice,
    *const VkMemoryAllocateInfo,
    *const c_void,
    *mut VkDeviceMemory,
) -> Result;
pub type FnFreeMemory =
    unsafe extern "system" fn(VkDevice, VkDeviceMemory, *const c_void);
pub type FnGetBufferMemoryRequirements = unsafe extern "system" fn(
    VkDevice,
    VkBuffer,
    *mut VkMemoryRequirements,
);
pub type FnGetImageMemoryRequirements = unsafe extern "system" fn(
    VkDevice,
    VkImage,
    *mut VkMemoryRequirements,
);
pub type FnBindBufferMemory = unsafe extern "system" fn(
    VkDevice,
    VkBuffer,
    VkDeviceMemory,
    VkDeviceSize,
) -> Result;
pub type FnBindImageMemory = unsafe extern "system" fn(
    VkDevice,
    VkImage,
    VkDeviceMemory,
    VkDeviceSize,
) -> Result;
pub type FnMapMemory = unsafe extern "system" fn(
    VkDevice,
    VkDeviceMemory,
    VkDeviceSize,
    VkDeviceSize,
    u32,
    *mut *mut c_void,
) -> Result;
pub type FnUnmapMemory =
    unsafe extern "system" fn(VkDevice, VkDeviceMemory);
pub type FnFlushMappedMemoryRanges = unsafe extern "system" fn(
    VkDevice,
    u32,
    *const VkMappedMemoryRange,
) -> Result;
pub type FnCreateImage = unsafe extern "system" fn(
    VkDevice,
    *const VkImageCreateInfo,
    *const c_void,
    *mut VkImage,
) -> Result;
pub type FnDestroyImage =
    unsafe extern "system" fn(VkDevice, VkImage, *const c_void);
pub type FnCreateImageView = unsafe extern "system" fn(
    VkDevice,
    *const VkImageViewCreateInfo,
    *const c_void,
    *mut VkImageView,
) -> Result;
pub type FnDestroyImageView =
    unsafe extern "system" fn(VkDevice, VkImageView, *const c_void);
pub type FnCreateShaderModule = unsafe extern "system" fn(
    VkDevice,
    *const VkShaderModuleCreateInfo,
    *const c_void,
    *mut VkShaderModule,
) -> Result;
pub type FnDestroyShaderModule =
    unsafe extern "system" fn(VkDevice, VkShaderModule, *const c_void);
pub type FnCreateCommandPool = unsafe extern "system" fn(
    VkDevice,
    *const VkCommandPoolCreateInfo,
    *const c_void,
    *mut VkCommandPool,
) -> Result;
pub type FnDestroyCommandPool =
    unsafe extern "system" fn(VkDevice, VkCommandPool, *const c_void);
pub type FnAllocateCommandBuffers = unsafe extern "system" fn(
    VkDevice,
    *const VkCommandBufferAllocateInfo,
    *mut VkCommandBuffer,
) -> Result;
pub type FnFreeCommandBuffers = unsafe extern "system" fn(
    VkDevice,
    VkCommandPool,
    u32,
    *const VkCommandBuffer,
);
pub type FnCreateFence = unsafe extern "system" fn(
    VkDevice,
    *const VkFenceCreateInfo,
    *const c_void,
    *mut VkFence,
) -> Result;
pub type FnDestroyFence =
    unsafe extern "system" fn(VkDevice, VkFence, *const c_void);
pub type FnWaitForFences = unsafe extern "system" fn(
    VkDevice,
    u32,
    *const VkFence,
    u32,
    u64,
) -> Result;
pub type FnResetFences =
    unsafe extern "system" fn(VkDevice, u32, *const VkFence) -> Result;
pub type FnBeginCommandBuffer = unsafe extern "system" fn(
    VkCommandBuffer,
    *const VkCommandBufferBeginInfo,
) -> Result;
pub type FnEndCommandBuffer =
    unsafe extern "system" fn(VkCommandBuffer) -> Result;
pub type FnResetCommandBuffer =
    unsafe extern "system" fn(VkCommandBuffer, u32) -> Result;
pub type FnQueueSubmit = unsafe extern "system" fn(
    VkQueue,
    u32,
    *const VkSubmitInfo,
    VkFence,
) -> Result;
pub type FnQueueWaitIdle =
    unsafe extern "system" fn(VkQueue) -> Result;
pub type FnDeviceWaitIdle =
    unsafe extern "system" fn(VkDevice) -> Result;
pub type FnCmdBeginRenderPass = unsafe extern "system" fn(
    VkCommandBuffer,
    *const VkRenderPassBeginInfo,
    u32,
);
pub type FnCmdEndRenderPass =
    unsafe extern "system" fn(VkCommandBuffer);
pub type FnCmdSetViewport = unsafe extern "system" fn(
    VkCommandBuffer,
    u32,
    u32,
    *const VkViewport,
);
pub type FnCmdSetScissor =
    unsafe extern "system" fn(VkCommandBuffer, u32, u32, *const VkRect2D);
pub type FnCmdDrawIndexed = unsafe extern "system" fn(
    VkCommandBuffer,
    u32,
    u32,
    u32,
    i32,
    u32,
);
pub type FnCmdDraw = unsafe extern "system" fn(
    VkCommandBuffer,
    u32,
    u32,
    u32,
    u32,
);
pub type FnCmdBindPipeline = unsafe extern "system" fn(
    VkCommandBuffer,
    PipelineBindPoint,
    VkPipeline,
);
pub type FnCmdBindVertexBuffers = unsafe extern "system" fn(
    VkCommandBuffer,
    u32,
    u32,
    *const VkBuffer,
    *const VkDeviceSize,
);
pub type FnCmdBindIndexBuffer = unsafe extern "system" fn(
    VkCommandBuffer,
    VkBuffer,
    VkDeviceSize,
    IndexType,
);
pub type FnCmdBindDescriptorSets = unsafe extern "system" fn(
    VkCommandBuffer,
    PipelineBindPoint,
    VkPipelineLayout,
    u32,
    u32,
    *const VkDescriptorSet,
    u32,
    *const u32,
);
pub type FnCmdPushConstants = unsafe extern "system" fn(
    VkCommandBuffer,
    VkPipelineLayout,
    ShaderStage,
    u32,
    u32,
    *const c_void,
);
pub type FnCmdPipelineBarrier = unsafe extern "system" fn(
    VkCommandBuffer,
    PipelineStage,
    PipelineStage,
    u32,
    u32,
    *const VkMemoryBarrier,
    u32,
    *const VkBufferMemoryBarrier,
    u32,
    *const VkImageMemoryBarrier,
);
pub type FnCmdCopyBuffer = unsafe extern "system" fn(
    VkCommandBuffer,
    VkBuffer,
    VkBuffer,
    u32,
    *const VkBufferCopy,
);
pub type FnCmdCopyBufferToImage = unsafe extern "system" fn(
    VkCommandBuffer,
    VkBuffer,
    VkImage,
    ImageLayout,
    u32,
    *const VkBufferImageCopy,
);
pub type FnCmdCopyImageToBuffer = unsafe extern "system" fn(
    VkCommandBuffer,
    VkImage,
    ImageLayout,
    VkBuffer,
    u32,
    *const VkBufferImageCopy,
);
pub type FnCmdClearAttachments = unsafe extern "system" fn(
    VkCommandBuffer,
    u32,
    *const c_void,
    u32,
    *const VkRect2D,
);
pub type FnCmdClearDepthStencilImage = unsafe extern "system" fn(
    VkCommandBuffer,
    VkImage,
    ImageLayout,
    *const VkClearDepthStencilValue,
    u32,
    *const VkImageSubresourceRange,
);
pub type FnCmdFillBuffer = unsafe extern "system" fn(
    VkCommandBuffer,
    VkBuffer,
    VkDeviceSize,
    VkDeviceSize,
    u32,
);
pub type FnCreateRenderPass = unsafe extern "system" fn(
    VkDevice,
    *const VkRenderPassCreateInfo,
    *const c_void,
    *mut VkRenderPass,
) -> Result;
pub type FnDestroyRenderPass =
    unsafe extern "system" fn(VkDevice, VkRenderPass, *const c_void);
pub type FnCreateFramebuffer = unsafe extern "system" fn(
    VkDevice,
    *const VkFramebufferCreateInfo,
    *const c_void,
    *mut VkFramebuffer,
) -> Result;
pub type FnDestroyFramebuffer =
    unsafe extern "system" fn(VkDevice, VkFramebuffer, *const c_void);
pub type FnCreatePipelineLayout = unsafe extern "system" fn(
    VkDevice,
    *const VkPipelineLayoutCreateInfo,
    *const c_void,
    *mut VkPipelineLayout,
) -> Result;
pub type FnDestroyPipelineLayout =
    unsafe extern "system" fn(VkDevice, VkPipelineLayout, *const c_void);
pub type FnCreateGraphicsPipelines = unsafe extern "system" fn(
    VkDevice,
    VkPipelineCache,
    u32,
    *const VkGraphicsPipelineCreateInfo,
    *const c_void,
    *mut VkPipeline,
) -> Result;
pub type FnDestroyPipeline =
    unsafe extern "system" fn(VkDevice, VkPipeline, *const c_void);
pub type FnCreateDescriptorSetLayout = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorSetLayoutCreateInfo,
    *const c_void,
    *mut VkDescriptorSetLayout,
) -> Result;
pub type FnDestroyDescriptorSetLayout =
    unsafe extern "system" fn(VkDevice, VkDescriptorSetLayout, *const c_void);
pub type FnCreateDescriptorPool = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorPoolCreateInfo,
    *const c_void,
    *mut VkDescriptorPool,
) -> Result;
pub type FnDestroyDescriptorPool =
    unsafe extern "system" fn(VkDevice, VkDescriptorPool, *const c_void);
pub type FnAllocateDescriptorSets = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorSetAllocateInfo,
    *mut VkDescriptorSet,
) -> Result;
pub type FnResetDescriptorPool =
    unsafe extern "system" fn(VkDevice, VkDescriptorPool, u32) -> Result;
pub type FnUpdateDescriptorSets = unsafe extern "system" fn(
    VkDevice,
    u32,
    *const VkWriteDescriptorSet,
    u32,
    *const c_void,
);
pub type FnCreateSampler = unsafe extern "system" fn(
    VkDevice,
    *const VkSamplerCreateInfo,
    *const c_void,
    *mut VkSampler,
) -> Result;
pub type FnDestroySampler =
    unsafe extern "system" fn(VkDevice, VkSampler, *const c_void);

// ---------------------------------------------------------------------------
// The loader
// ---------------------------------------------------------------------------

/// Minimal runtime loader for the Vulkan entry points the engine uses.
///
/// The backend resolves symbols through a resolve callback (the platform
/// passes the OS's `vkGetInstanceProcAddr` in), so this file needs no
/// windowing system. Every pointer is checked at load time: the first
/// symbol that cannot be resolved is reported by name.
pub struct VkLoader {
    pub vk_get_instance_proc_addr: Option<FnGetInstanceProcAddr>,
    pub vk_enumerate_instance_extension_properties:
        Option<FnEnumerateInstanceExtensionProperties>,
    pub vk_create_instance: Option<FnCreateInstance>,
    pub vk_destroy_instance: Option<FnDestroyInstance>,
    pub vk_enumerate_physical_devices: Option<FnEnumeratePhysicalDevices>,
    pub vk_get_physical_device_properties:
        Option<FnGetPhysicalDeviceProperties>,
    pub vk_get_physical_device_format_properties:
        Option<FnGetPhysicalDeviceFormatProperties>,
    pub vk_get_physical_device_features:
        Option<FnGetPhysicalDeviceFeatures>,
    pub vk_get_physical_device_memory_properties:
        Option<FnGetPhysicalDeviceMemoryProperties>,
    pub vk_get_physical_device_queue_family_properties:
        Option<FnGetPhysicalDeviceQueueFamilyProperties>,
    pub vk_enumerate_device_extension_properties:
        Option<FnEnumerateDeviceExtensionProperties>,
    pub vk_create_device: Option<FnCreateDevice>,
    pub vk_destroy_device: Option<FnDestroyDevice>,
    pub vk_get_device_queue: Option<FnGetDeviceQueue>,
    pub vk_get_device_proc_addr: Option<FnGetDeviceProcAddr>,
    pub vk_create_buffer: Option<FnCreateBuffer>,
    pub vk_destroy_buffer: Option<FnDestroyBuffer>,
    pub vk_allocate_memory: Option<FnAllocateMemory>,
    pub vk_free_memory: Option<FnFreeMemory>,
    pub vk_get_buffer_memory_requirements:
        Option<FnGetBufferMemoryRequirements>,
    pub vk_get_image_memory_requirements:
        Option<FnGetImageMemoryRequirements>,
    pub vk_bind_buffer_memory: Option<FnBindBufferMemory>,
    pub vk_bind_image_memory: Option<FnBindImageMemory>,
    pub vk_map_memory: Option<FnMapMemory>,
    pub vk_unmap_memory: Option<FnUnmapMemory>,
    pub vk_flush_mapped_memory_ranges: Option<FnFlushMappedMemoryRanges>,
    pub vk_create_image: Option<FnCreateImage>,
    pub vk_destroy_image: Option<FnDestroyImage>,
    pub vk_create_image_view: Option<FnCreateImageView>,
    pub vk_destroy_image_view: Option<FnDestroyImageView>,
    pub vk_create_shader_module: Option<FnCreateShaderModule>,
    pub vk_destroy_shader_module: Option<FnDestroyShaderModule>,
    pub vk_create_command_pool: Option<FnCreateCommandPool>,
    pub vk_destroy_command_pool: Option<FnDestroyCommandPool>,
    pub vk_allocate_command_buffers: Option<FnAllocateCommandBuffers>,
    pub vk_free_command_buffers: Option<FnFreeCommandBuffers>,
    pub vk_create_fence: Option<FnCreateFence>,
    pub vk_destroy_fence: Option<FnDestroyFence>,
    pub vk_wait_for_fences: Option<FnWaitForFences>,
    pub vk_reset_fences: Option<FnResetFences>,
    pub vk_begin_command_buffer: Option<FnBeginCommandBuffer>,
    pub vk_end_command_buffer: Option<FnEndCommandBuffer>,
    pub vk_reset_command_buffer: Option<FnResetCommandBuffer>,
    pub vk_queue_submit: Option<FnQueueSubmit>,
    pub vk_queue_wait_idle: Option<FnQueueWaitIdle>,
    pub vk_device_wait_idle: Option<FnDeviceWaitIdle>,
    pub vk_cmd_begin_render_pass: Option<FnCmdBeginRenderPass>,
    pub vk_cmd_end_render_pass: Option<FnCmdEndRenderPass>,
    pub vk_cmd_set_viewport: Option<FnCmdSetViewport>,
    pub vk_cmd_set_scissor: Option<FnCmdSetScissor>,
    pub vk_cmd_draw_indexed: Option<FnCmdDrawIndexed>,
    pub vk_cmd_draw: Option<FnCmdDraw>,
    pub vk_cmd_bind_pipeline: Option<FnCmdBindPipeline>,
    pub vk_cmd_bind_vertex_buffers: Option<FnCmdBindVertexBuffers>,
    pub vk_cmd_bind_index_buffer: Option<FnCmdBindIndexBuffer>,
    pub vk_cmd_bind_descriptor_sets: Option<FnCmdBindDescriptorSets>,
    pub vk_cmd_push_constants: Option<FnCmdPushConstants>,
    pub vk_cmd_pipeline_barrier: Option<FnCmdPipelineBarrier>,
    pub vk_cmd_copy_buffer: Option<FnCmdCopyBuffer>,
    pub vk_cmd_copy_buffer_to_image: Option<FnCmdCopyBufferToImage>,
    pub vk_cmd_copy_image_to_buffer: Option<FnCmdCopyImageToBuffer>,
    pub vk_cmd_clear_attachments: Option<FnCmdClearAttachments>,
    pub vk_cmd_clear_depth_stencil_image:
        Option<FnCmdClearDepthStencilImage>,
    pub vk_cmd_fill_buffer: Option<FnCmdFillBuffer>,
    pub vk_create_render_pass: Option<FnCreateRenderPass>,
    pub vk_destroy_render_pass: Option<FnDestroyRenderPass>,
    pub vk_create_framebuffer: Option<FnCreateFramebuffer>,
    pub vk_destroy_framebuffer: Option<FnDestroyFramebuffer>,
    pub vk_create_pipeline_layout: Option<FnCreatePipelineLayout>,
    pub vk_destroy_pipeline_layout: Option<FnDestroyPipelineLayout>,
    pub vk_create_graphics_pipelines: Option<FnCreateGraphicsPipelines>,
    pub vk_destroy_pipeline: Option<FnDestroyPipeline>,
    pub vk_create_descriptor_set_layout:
        Option<FnCreateDescriptorSetLayout>,
    pub vk_destroy_descriptor_set_layout:
        Option<FnDestroyDescriptorSetLayout>,
    pub vk_create_descriptor_pool: Option<FnCreateDescriptorPool>,
    pub vk_destroy_descriptor_pool: Option<FnDestroyDescriptorPool>,
    pub vk_allocate_descriptor_sets: Option<FnAllocateDescriptorSets>,
    pub vk_reset_descriptor_pool: Option<FnResetDescriptorPool>,
    pub vk_update_descriptor_sets: Option<FnUpdateDescriptorSets>,
    pub vk_create_sampler: Option<FnCreateSampler>,
    pub vk_destroy_sampler: Option<FnDestroySampler>,
}

impl VkLoader {
    pub fn new() -> Self {
        VkLoader {
            vk_get_instance_proc_addr: None,
            vk_enumerate_instance_extension_properties: None,
            vk_create_instance: None,
            vk_destroy_instance: None,
            vk_enumerate_physical_devices: None,
            vk_get_physical_device_properties: None,
            vk_get_physical_device_format_properties: None,
            vk_get_physical_device_features: None,
            vk_get_physical_device_memory_properties: None,
            vk_get_physical_device_queue_family_properties: None,
            vk_enumerate_device_extension_properties: None,
            vk_create_device: None,
            vk_destroy_device: None,
            vk_get_device_queue: None,
            vk_get_device_proc_addr: None,
            vk_create_buffer: None,
            vk_destroy_buffer: None,
            vk_allocate_memory: None,
            vk_free_memory: None,
            vk_get_buffer_memory_requirements: None,
            vk_get_image_memory_requirements: None,
            vk_bind_buffer_memory: None,
            vk_bind_image_memory: None,
            vk_map_memory: None,
            vk_unmap_memory: None,
            vk_flush_mapped_memory_ranges: None,
            vk_create_image: None,
            vk_destroy_image: None,
            vk_create_image_view: None,
            vk_destroy_image_view: None,
            vk_create_shader_module: None,
            vk_destroy_shader_module: None,
            vk_create_command_pool: None,
            vk_destroy_command_pool: None,
            vk_allocate_command_buffers: None,
            vk_free_command_buffers: None,
            vk_create_fence: None,
            vk_destroy_fence: None,
            vk_wait_for_fences: None,
            vk_reset_fences: None,
            vk_begin_command_buffer: None,
            vk_end_command_buffer: None,
            vk_reset_command_buffer: None,
            vk_queue_submit: None,
            vk_queue_wait_idle: None,
            vk_device_wait_idle: None,
            vk_cmd_begin_render_pass: None,
            vk_cmd_end_render_pass: None,
            vk_cmd_set_viewport: None,
            vk_cmd_set_scissor: None,
            vk_cmd_draw_indexed: None,
            vk_cmd_draw: None,
            vk_cmd_bind_pipeline: None,
            vk_cmd_bind_vertex_buffers: None,
            vk_cmd_bind_index_buffer: None,
            vk_cmd_bind_descriptor_sets: None,
            vk_cmd_push_constants: None,
            vk_cmd_pipeline_barrier: None,
            vk_cmd_copy_buffer: None,
            vk_cmd_copy_buffer_to_image: None,
            vk_cmd_copy_image_to_buffer: None,
            vk_cmd_clear_attachments: None,
            vk_cmd_clear_depth_stencil_image: None,
            vk_cmd_fill_buffer: None,
            vk_create_render_pass: None,
            vk_destroy_render_pass: None,
            vk_create_framebuffer: None,
            vk_destroy_framebuffer: None,
            vk_create_pipeline_layout: None,
            vk_destroy_pipeline_layout: None,
            vk_create_graphics_pipelines: None,
            vk_destroy_pipeline: None,
            vk_create_descriptor_set_layout: None,
            vk_destroy_descriptor_set_layout: None,
            vk_create_descriptor_pool: None,
            vk_destroy_descriptor_pool: None,
            vk_allocate_descriptor_sets: None,
            vk_reset_descriptor_pool: None,
            vk_update_descriptor_sets: None,
            vk_create_sampler: None,
            vk_destroy_sampler: None,
        }
    }

    /// Resolves the loader-global entry points (the ones callable before an
    /// instance exists) from the given callback. Returns false and sets
    /// `missing` to the first name that could not be resolved.
    pub fn load(&mut self, resolve: ResolveFn, missing: &mut String) -> bool {
        macro_rules! global {
            ($field:ident, $name:literal) => {
                let name = concat!($name, "\0");
                let ptr = unsafe {
                    resolve(std::ptr::null_mut(), name.as_ptr() as *const _)
                };
                if ptr.is_null() {
                    *missing = $name.to_string();
                    return false;
                }
                self.$field = Some(unsafe {
                    std::mem::transmute::<*mut c_void, _>(ptr)
                });
            };
        }
        global!(
            vk_get_instance_proc_addr,
            "vkGetInstanceProcAddr"
        );
        global!(
            vk_enumerate_instance_extension_properties,
            "vkEnumerateInstanceExtensionProperties"
        );
        global!(vk_create_instance, "vkCreateInstance");
        true
    }

    /// Resolves the instance-level entry points (the ones callable only
    /// with a real instance) by calling the previously loaded
    /// `vkGetInstanceProcAddr` on this instance directly. Returns false
    /// and sets `missing` to the first name that could not be resolved.
    pub fn load_instance(&mut self, instance: VkInstance, missing: &mut String) -> bool {
        let get_proc = match self.vk_get_instance_proc_addr {
            Some(f) => f,
            None => {
                *missing = "vkGetInstanceProcAddr".to_string();
                return false;
            }
        };
        macro_rules! instance_fn {
            ($field:ident, $name:literal) => {
                let name = concat!($name, "\0");
                let ptr = unsafe { get_proc(instance, name.as_ptr() as *const _) };
                if ptr.is_null() {
                    *missing = $name.to_string();
                    return false;
                }
                self.$field = Some(unsafe {
                    std::mem::transmute::<*mut c_void, _>(ptr)
                });
            };
        }
        instance_fn!(vk_destroy_instance, "vkDestroyInstance");
        instance_fn!(
            vk_enumerate_physical_devices,
            "vkEnumeratePhysicalDevices"
        );
        instance_fn!(
            vk_get_physical_device_properties,
            "vkGetPhysicalDeviceProperties"
        );
        instance_fn!(
            vk_get_physical_device_format_properties,
            "vkGetPhysicalDeviceFormatProperties"
        );
        instance_fn!(
            vk_get_physical_device_features,
            "vkGetPhysicalDeviceFeatures"
        );
        instance_fn!(
            vk_get_physical_device_memory_properties,
            "vkGetPhysicalDeviceMemoryProperties"
        );
        instance_fn!(
            vk_get_physical_device_queue_family_properties,
            "vkGetPhysicalDeviceQueueFamilyProperties"
        );
        instance_fn!(
            vk_enumerate_device_extension_properties,
            "vkEnumerateDeviceExtensionProperties"
        );
        instance_fn!(vk_create_device, "vkCreateDevice");
        instance_fn!(vk_get_device_proc_addr, "vkGetDeviceProcAddr");
        instance_fn!(vk_destroy_device, "vkDestroyDevice");
        instance_fn!(vk_get_device_queue, "vkGetDeviceQueue");
        true
    }

    /// Resolves every device-level entry point by calling the previously
    /// loaded `vkGetDeviceProcAddr` on this device directly. Returns false
    /// and sets `missing` to the first name that could not be resolved.
    pub fn load_device(&mut self, device: VkDevice, missing: &mut String) -> bool {
        let get_proc = match self.vk_get_device_proc_addr {
            Some(f) => f,
            None => {
                *missing = "vkGetDeviceProcAddr".to_string();
                return false;
            }
        };
        macro_rules! device {
            ($field:ident, $name:literal) => {
                let name = concat!($name, "\0");
                let ptr = unsafe {
                    get_proc(device, name.as_ptr() as *const _)
                };
                if ptr.is_null() {
                    *missing = $name.to_string();
                    return false;
                }
                self.$field = Some(unsafe {
                    std::mem::transmute::<*mut c_void, _>(ptr)
                });
            };
        }
        device!(vk_create_buffer, "vkCreateBuffer");
        device!(vk_destroy_buffer, "vkDestroyBuffer");
        device!(vk_allocate_memory, "vkAllocateMemory");
        device!(vk_free_memory, "vkFreeMemory");
        device!(
            vk_get_buffer_memory_requirements,
            "vkGetBufferMemoryRequirements"
        );
        device!(
            vk_get_image_memory_requirements,
            "vkGetImageMemoryRequirements"
        );
        device!(vk_bind_buffer_memory, "vkBindBufferMemory");
        device!(vk_bind_image_memory, "vkBindImageMemory");
        device!(vk_map_memory, "vkMapMemory");
        device!(vk_unmap_memory, "vkUnmapMemory");
        device!(
            vk_flush_mapped_memory_ranges,
            "vkFlushMappedMemoryRanges"
        );
        device!(vk_create_image, "vkCreateImage");
        device!(vk_destroy_image, "vkDestroyImage");
        device!(vk_create_image_view, "vkCreateImageView");
        device!(vk_destroy_image_view, "vkDestroyImageView");
        device!(vk_create_shader_module, "vkCreateShaderModule");
        device!(vk_destroy_shader_module, "vkDestroyShaderModule");
        device!(vk_create_command_pool, "vkCreateCommandPool");
        device!(vk_destroy_command_pool, "vkDestroyCommandPool");
        device!(
            vk_allocate_command_buffers,
            "vkAllocateCommandBuffers"
        );
        device!(vk_free_command_buffers, "vkFreeCommandBuffers");
        device!(vk_create_fence, "vkCreateFence");
        device!(vk_destroy_fence, "vkDestroyFence");
        device!(vk_wait_for_fences, "vkWaitForFences");
        device!(vk_reset_fences, "vkResetFences");
        device!(vk_begin_command_buffer, "vkBeginCommandBuffer");
        device!(vk_end_command_buffer, "vkEndCommandBuffer");
        device!(vk_reset_command_buffer, "vkResetCommandBuffer");
        device!(vk_queue_submit, "vkQueueSubmit");
        device!(vk_queue_wait_idle, "vkQueueWaitIdle");
        device!(vk_device_wait_idle, "vkDeviceWaitIdle");
        device!(vk_cmd_begin_render_pass, "vkCmdBeginRenderPass");
        device!(vk_cmd_end_render_pass, "vkCmdEndRenderPass");
        device!(vk_cmd_set_viewport, "vkCmdSetViewport");
        device!(vk_cmd_set_scissor, "vkCmdSetScissor");
        device!(vk_cmd_draw_indexed, "vkCmdDrawIndexed");
        device!(vk_cmd_draw, "vkCmdDraw");
        device!(vk_cmd_bind_pipeline, "vkCmdBindPipeline");
        device!(vk_cmd_bind_vertex_buffers, "vkCmdBindVertexBuffers");
        device!(vk_cmd_bind_index_buffer, "vkCmdBindIndexBuffer");
        device!(vk_cmd_bind_descriptor_sets, "vkCmdBindDescriptorSets");
        device!(vk_cmd_push_constants, "vkCmdPushConstants");
        device!(vk_cmd_pipeline_barrier, "vkCmdPipelineBarrier");
        device!(vk_cmd_copy_buffer, "vkCmdCopyBuffer");
        device!(
            vk_cmd_copy_buffer_to_image,
            "vkCmdCopyBufferToImage"
        );
        device!(
            vk_cmd_copy_image_to_buffer,
            "vkCmdCopyImageToBuffer"
        );
        device!(vk_cmd_clear_attachments, "vkCmdClearAttachments");
        device!(
            vk_cmd_clear_depth_stencil_image,
            "vkCmdClearDepthStencilImage"
        );
        device!(vk_cmd_fill_buffer, "vkCmdFillBuffer");
        device!(vk_create_render_pass, "vkCreateRenderPass");
        device!(vk_destroy_render_pass, "vkDestroyRenderPass");
        device!(vk_create_framebuffer, "vkCreateFramebuffer");
        device!(vk_destroy_framebuffer, "vkDestroyFramebuffer");
        device!(vk_create_pipeline_layout, "vkCreatePipelineLayout");
        device!(vk_destroy_pipeline_layout, "vkDestroyPipelineLayout");
        device!(
            vk_create_graphics_pipelines,
            "vkCreateGraphicsPipelines"
        );
        device!(vk_destroy_pipeline, "vkDestroyPipeline");
        device!(
            vk_create_descriptor_set_layout,
            "vkCreateDescriptorSetLayout"
        );
        device!(
            vk_destroy_descriptor_set_layout,
            "vkDestroyDescriptorSetLayout"
        );
        device!(vk_create_descriptor_pool, "vkCreateDescriptorPool");
        device!(
            vk_destroy_descriptor_pool,
            "vkDestroyDescriptorPool"
        );
        device!(
            vk_allocate_descriptor_sets,
            "vkAllocateDescriptorSets"
        );
        device!(vk_reset_descriptor_pool, "vkResetDescriptorPool");
        device!(vk_update_descriptor_sets, "vkUpdateDescriptorSets");
        device!(vk_create_sampler, "vkCreateSampler");
        device!(vk_destroy_sampler, "vkDestroySampler");
        true
    }
}
