//! The engine's own render contract.
//!
//! The contract is technique-agnostic: generic render-pass composition
//! instead of technique-named methods, so a deferred path can reuse the same
//! primitives. Backends implement the `RenderBackend` trait in their own
//! modules; the contract never contains graphics-API types.
//!
//! The protocol rules are the same as OpenGrace's contract: mutating calls
//! made before a backend is initialised or after shutdown are protocol
//! violations and are counted through `violationCount()`; a repeated
//! shutdown is a violation too but must stay safe.

pub mod null;
#[cfg(feature = "vulkan")]
pub mod vulkan;
pub use null::{NullEvent, NullRenderBackend};
#[cfg(feature = "vulkan")]
pub use vulkan::{VulkanBackend, VulkanBackendConfig};

use std::sync::{Arc, Mutex};
// Re-exported so the backend modules (and the engine's other modules) reach
// the contract's math types through the contract.
pub use crate::math::{Mat4, Vec3};

/// Straight (non-premultiplied) RGBA colour, 8 bits per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClearColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

/// Buffer selection for clear(), as a platform-neutral bitmask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClearBuffer {
    pub color: bool,
    pub depth: bool,
    pub stencil: bool,
}

impl ClearBuffer {
    pub const COLOR_DEPTH_STENCIL: ClearBuffer = ClearBuffer {
        color: true,
        depth: true,
        stencil: true,
    };
}

/// One pixel sample produced by readback().
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadbackSample {
    pub x: u32,
    pub y: u32,
    pub color: ClearColor,
    pub depth: f32,
    pub stencil: i32,
}

/// Project-owned pixel format enumeration. This is deliberately NOT a
/// graphics-API enum: the backend maps these values onto its own API's
/// constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8 bits per channel, RGBA colour attachment (linear values, no display
    /// encoding)
    Rgba8,
    /// 16-bit float per channel, RGBA colour attachment (HDR)
    Rgba16F,
    /// 8-bit single channel
    R8,
    /// 24-bit depth + 8-bit stencil attachment
    Depth24Stencil8,
}

/// Bytes one texel of the format occupies in memory. The same count the
/// backends use for a volume slice.
pub fn bytes_per_pixel(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::Rgba8 => 4,
        PixelFormat::Rgba16F => 8,
        PixelFormat::R8 => 1,
        PixelFormat::Depth24Stencil8 => 4,
    }
}

/// Hard cap on colour attachments per render target. Enforced, not silently
/// applied.
pub const MAX_COLOR_ATTACHMENTS: u32 = 4;

/// Attachment index selecting the depth attachment in
/// `renderTargetTexture()`.
pub const DEPTH_ATTACHMENT: u32 = 0xFFFF_FFFF;

/// Description of a render target. A depth-only target (colorAttachmentCount
/// == 0) is what a shadow pass needs.
#[derive(Clone, Debug)]
pub struct RenderTargetDesc {
    pub width: u32,
    pub height: u32,
    pub color_attachment_count: u32,
    pub color_formats: [PixelFormat; MAX_COLOR_ATTACHMENTS as usize],
    pub depth_attachment: bool,
    pub depth_format: PixelFormat,
}

impl Default for RenderTargetDesc {
    fn default() -> Self {
        RenderTargetDesc {
            width: 0,
            height: 0,
            color_attachment_count: 1,
            color_formats: [PixelFormat::Rgba8; MAX_COLOR_ATTACHMENTS as usize],
            depth_attachment: true,
            depth_format: PixelFormat::Depth24Stencil8,
        }
    }
}

impl RenderTargetDesc {
    /// Throws `RenderError` on an invalid or incomplete desc: zero size, more
    /// than MAX_COLOR_ATTACHMENTS colour attachments, or neither a colour nor
    /// a depth attachment.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.width == 0 || self.height == 0 {
            return Err(RenderError::from(
                "RenderTargetDesc: width and height must be positive",
            ));
        }
        if self.color_attachment_count > MAX_COLOR_ATTACHMENTS {
            return Err(RenderError::from(format!(
                "RenderTargetDesc: at most {} colour attachments, got {}",
                MAX_COLOR_ATTACHMENTS, self.color_attachment_count
            )));
        }
        if self.color_attachment_count == 0 && !self.depth_attachment {
            return Err(RenderError::from(
                "RenderTargetDesc: neither a colour nor a depth attachment",
            ));
        }
        Ok(())
    }
}

/// Description of a standalone 2D texture: a resource the application
/// creates, fills, samples and reads back.
#[derive(Clone, Copy, Debug)]
pub struct Texture2DDesc {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
}

impl Default for Texture2DDesc {
    fn default() -> Self {
        Texture2DDesc {
            width: 0,
            height: 0,
            format: PixelFormat::Rgba8,
        }
    }
}

impl Texture2DDesc {
    /// Throws `RenderError` when a dimension is zero, or when the format is
    /// the 2D depth/stencil format.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.width == 0 || self.height == 0 {
            return Err(RenderError::from(
                "Texture2DDesc: width and height must be positive",
            ));
        }
        if self.format == PixelFormat::Depth24Stencil8 {
            return Err(RenderError::from(
                "Texture2DDesc: the depth/stencil format is not a colour texture",
            ));
        }
        Ok(())
    }
}

/// Description of a 3D texture: a volume the contract can create, fill and
/// sample. Dimensions are in texels and each of them must be positive.
#[derive(Clone, Copy, Debug)]
pub struct Texture3DDesc {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: PixelFormat,
}

impl Default for Texture3DDesc {
    fn default() -> Self {
        Texture3DDesc {
            width: 0,
            height: 0,
            depth: 0,
            format: PixelFormat::Rgba16F,
        }
    }
}

impl Texture3DDesc {
    /// Throws `RenderError` when a dimension is zero, or when the format is
    /// the 2D depth/stencil format (the same rejection Texture2DDesc makes:
    /// a depth/stencil format is not a colour volume).
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.width == 0 || self.height == 0 || self.depth == 0 {
            return Err(RenderError::from(
                "Texture3DDesc: width, height and depth must be positive",
            ));
        }
        if self.format == PixelFormat::Depth24Stencil8 {
            return Err(RenderError::from(
                "Texture3DDesc: the depth/stencil format is not a colour volume",
            ));
        }
        Ok(())
    }
}

/// Description of a layer of a volume, as a render target. A layer target
/// borrows the volume; it does not own it.
#[derive(Clone, Debug)]
pub struct VolumeLayerDesc {
    pub volume: Texture3DDesc,
    pub layer: u32,
}

impl VolumeLayerDesc {
    pub fn validate(&self) -> Result<(), RenderError> {
        self.volume.validate()?;
        if self.layer >= self.volume.depth {
            return Err(RenderError::from(format!(
                "VolumeLayerDesc: layer {} out of range (depth {})",
                self.layer, self.volume.depth
            )));
        }
        Ok(())
    }
}

/// The error type of the render contract. It carries the message that the
/// caller sees.
#[derive(Debug, Clone)]
pub struct RenderError {
    pub message: String,
}

impl RenderError {
    pub fn from(message: impl Into<String>) -> Self {
        RenderError {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RenderError {}

/// A shared handle to a render target. The handle object stays valid until
/// backend shutdown even after release, so that use-after-release can be
/// detected safely as a protocol violation.
#[derive(Clone)]
pub struct RenderTarget {
    pub(crate) inner: Arc<Mutex<RenderTargetInner>>,
}

impl RenderTarget {
    pub fn width(&self) -> u32 {
        self.inner.lock().unwrap().width
    }
    pub fn height(&self) -> u32 {
        self.inner.lock().unwrap().height
    }
    pub fn color_attachment_count(&self) -> u32 {
        self.inner.lock().unwrap().color_attachment_count
    }
    pub fn has_color_attachment(&self) -> bool {
        self.inner.lock().unwrap().color_attachment_count > 0
    }
    pub fn has_depth_attachment(&self) -> bool {
        self.inner.lock().unwrap().depth_attachment
    }
}

/// The backend-owned state of a render target handle.
pub struct RenderTargetInner {
    pub width: u32,
    pub height: u32,
    pub color_attachment_count: u32,
    pub depth_attachment: bool,
    pub released: bool,
}

/// A shared handle to one attachment of a render target, for sampling it as a
/// texture (for example the depth attachment of a shadow map).
#[derive(Clone)]
pub struct TextureView {
    pub(crate) inner: Arc<Mutex<TextureViewInner>>,
}

impl TextureView {
    pub fn width(&self) -> u32 {
        self.inner.lock().unwrap().width
    }
    pub fn height(&self) -> u32 {
        self.inner.lock().unwrap().height
    }
    pub fn depth(&self) -> u32 {
        self.inner.lock().unwrap().depth
    }
    pub fn format(&self) -> PixelFormat {
        self.inner.lock().unwrap().format
    }
    pub fn is_depth(&self) -> bool {
        self.inner.lock().unwrap().is_depth
    }
}

/// The backend-owned state of a texture-view handle.
pub struct TextureViewInner {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: PixelFormat,
    pub is_depth: bool,
    pub released: bool,
}

/// A shared handle to a GPU mesh. Upload is explicit via `upload()`; nothing
/// reaches the GPU until upload().
#[derive(Clone)]
pub struct GpuMesh {
    pub(crate) inner: Arc<Mutex<GpuMeshInner>>,
}

impl GpuMesh {
    pub fn vertex_count(&self) -> u32 {
        self.inner.lock().unwrap().vertex_count
    }
    pub fn index_count(&self) -> u32 {
        self.inner.lock().unwrap().index_count
    }
}

/// The backend-owned state of a GPU mesh handle.
pub struct GpuMeshInner {
    pub vertex_count: u32,
    pub index_count: u32,
    pub uploaded: bool,
}

/// A shared handle to a shader program. The backend owns the source (the
/// shader library rule); the handle is what draw() receives.
#[derive(Clone)]
pub struct ShaderProgram {
    pub(crate) inner: Arc<Mutex<ShaderProgramInner>>,
}

impl ShaderProgram {
    pub fn id(&self) -> u32 {
        self.inner.lock().unwrap().id
    }
}

/// The backend-owned state of a shader program handle.
pub struct ShaderProgramInner {
    pub id: u32,
    pub compiled: bool,
}

/// Camera parameters, API-neutral. The default is the backend's documented
/// diagnostic camera: eye (0,0,1) looking at the origin, up +Y, fovY 90
/// degrees.
#[derive(Clone, Copy, Debug)]
pub struct CameraParams {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_rad: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}
impl Default for CameraParams {
    fn default() -> Self {
        CameraParams {
            eye: Vec3::new(0.0, 0.0, 1.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            fov_y_rad: std::f32::consts::PI / 2.0,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        }
    }
}
impl CameraParams {
    /// Throws `RenderError` on out-of-range or degenerate values.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.fov_y_rad <= 0.0 || self.fov_y_rad >= std::f32::consts::PI {
            return Err(RenderError::from(
                "CameraParams: fovY must be in (0, pi)",
            ));
        }
        if self.aspect <= 0.0 {
            return Err(RenderError::from("CameraParams: aspect must be > 0"));
        }
        if self.near <= 0.0 || self.far <= self.near {
            return Err(RenderError::from(
                "CameraParams: near must be > 0 and far must be > near",
            ));
        }
        Ok(())
    }
}

/// The API-neutral render contract. Backends implement this trait in their own
/// modules; the contract never contains graphics-API types.
pub trait RenderBackend {
    /// Human-readable backend description, for example "null (headless)".
    fn describe(&self) -> String;

    /// Sets the colour used by subsequent clear() calls.
    fn set_clear_color(&mut self, color: ClearColor);

    /// The colour currently used by clear().
    fn clear_color(&self) -> ClearColor;

    /// Enables or disables depth testing for subsequent passes.
    fn set_depth_test_enabled(&mut self, enabled: bool);

    /// Clears the frame target with the current clear colour.
    fn clear(&mut self);

    /// Frame target size in pixels.
    fn width(&self) -> u32;
    fn height(&self) -> u32;

    /// Releases backend resources. Exactly one successful call per backend
    /// lifetime; calls after shutdown are protocol violations.
    fn shutdown(&mut self);

    /// Number of protocol violations observed so far.
    fn violation_count(&self) -> u64;

    /// Compiles and links a shader program for the given API-neutral program
    /// identity. The backend owns the source (the shader library rule).
    fn create_shader_program(&mut self, id: u32) -> Result<ShaderProgram, RenderError>;

    /// Creates a GPU mesh handle. Upload is explicit via `upload()` on the
    /// backend.
    fn create_gpu_mesh(&mut self) -> Result<GpuMesh, RenderError>;

    /// Uploads the mesh: validates it and uploads the vertex and index data.
    fn upload_mesh(&mut self, mesh: &GpuMesh, layout: &VertexLayout, vertices: &[u8],
                   indices: &[u32]) -> Result<(), RenderError>;

    /// Activates the program, sets a full-frame viewport and draws the mesh
    /// with an indexed draw.
    fn draw(&mut self, mesh: &GpuMesh, program: &ShaderProgram);

    /// Submits a lit draw (mesh + transform + surface + sun + dynamic lights)
    /// for drawing with the backend's forward lighting pass.
    fn submit(&mut self, submission: &DrawSubmission);

    // -- Render passes and targets -------------------------------------------
    /// Creates a render target. Throws `RenderError` on an invalid or
    /// incomplete desc.
    fn create_render_target(&mut self, desc: &RenderTargetDesc)
        -> Result<RenderTarget, RenderError>;

    /// Begins a render pass on a target; None targets the default framebuffer.
    fn begin_render_pass(&mut self, target: Option<&RenderTarget>);

    /// Ends the current render pass (back to the default framebuffer).
    fn end_render_pass(&mut self);

    /// Releases a target. Use after release is a protocol violation.
    fn release_render_target(&mut self, target: &RenderTarget);

    /// Attachment texture of a target, for sampling. None + violation on
    /// unknown/released/feedback-bound targets.
    fn render_target_texture(&self, target: &RenderTarget) -> Option<TextureView>;

    /// Attachment texture by index.
    fn render_target_texture_at(&self, target: &RenderTarget, attachment: u32)
        -> Option<TextureView>;

    /// Binds an attachment texture for sampling on a texture unit.
    fn bind_texture(&mut self, view: Option<&TextureView>, unit: u32);

    // -- 2D textures ----------------------------------------------------------
    /// Creates a standalone 2D texture and returns its view for sampling.
    fn create_texture_2d(&mut self, desc: &Texture2DDesc)
        -> Result<TextureView, RenderError>;

    /// Releases a texture created by create_texture_2d.
    fn release_texture_2d(&mut self, view: &TextureView);

    /// Writes the texture's one layer: width * height * bytesPerPixel(format)
    /// bytes in row-major order, rows top to bottom, no row padding.
    fn fill_texture_2d(&mut self, view: &TextureView, data: &[u8])
        -> Result<(), RenderError>;

    /// Reads the texture's one layer back into data, in the same order and
    /// with the same size rule as fill_texture_2d.
    fn read_texture_2d(&self, view: &TextureView, data: &mut [u8])
        -> Result<bool, RenderError>;

    // -- Camera and path selection -------------------------------------------
    /// Sets the camera used by submit()/draw() (view-projection).
    fn set_camera(&mut self, camera: CameraParams);

    /// The camera currently used by submit()/draw().
    fn camera(&self) -> CameraParams;

    /// Presents the completed frame: captures any queued readback and swaps
    /// the back buffer.
    fn swap_buffers(&mut self);

    /// Queues a pixel to be captured by the next readback().
    fn queue_readback(&mut self, x: u32, y: u32);

    /// Captures the centre pixel and all queued pixels from the current read
    /// buffer.
    fn readback(&mut self);

    /// Samples captured by the last readback(), in queue order.
    fn last_readback(&self) -> &[ReadbackSample];

    /// Whether a centre-pixel sample has been captured.
    fn has_center_pixel(&self) -> bool;
    fn center_pixel(&self) -> ClearColor;
    fn center_depth(&self) -> f32;
    fn center_stencil(&self) -> i32;
}

/// Vertex attribute semantics understood by the render module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexSemantic {
    Position,
    Normal,
    TexCoord0,
    Color,
}

/// Component type of a vertex attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentType {
    Float32,
    Uint8Normalized,
}

/// One vertex attribute inside a vertex.
#[derive(Clone, Copy, Debug)]
pub struct VertexAttribute {
    pub semantic: VertexSemantic,
    pub component_type: ComponentType,
    pub components: u32,
    pub offset: u32,
    pub normalized: bool,
}

/// Full vertex layout: a stride plus the attributes inside each vertex.
#[derive(Clone, Debug)]
pub struct VertexLayout {
    pub stride: u32,
    pub attributes: Vec<VertexAttribute>,
}

impl Default for VertexLayout {
    fn default() -> Self {
        VertexLayout {
            stride: 0,
            attributes: Vec::new(),
        }
    }
}

/// A submission to the forward lighting pass. Plain data, no graphics-API
/// types: the same shape the contract draws with.
#[derive(Clone)]
pub struct DrawSubmission {
    pub mesh: GpuMesh,
    pub model: Mat4,
    pub material: Material,
    pub sun: Option<DirectionalLight>,
    pub lights: Vec<DynamicLight>,
}

/// The material's surface parameters, validated. The contract's values, the
/// same as OpenGrace's SurfaceParams.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceParams {
    pub albedo: Vec3,
    pub specular: Vec3,
    pub shininess: f32,
}

impl SurfaceParams {
    /// Throws `RenderError` on out-of-range or degenerate values.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.albedo.x < 0.0 || self.albedo.y < 0.0 || self.albedo.z < 0.0 {
            return Err(RenderError::from(
                "SurfaceParams: albedo components must be >= 0",
            ));
        }
        if self.specular.x < 0.0 || self.specular.y < 0.0 || self.specular.z < 0.0 {
            return Err(RenderError::from(
                "SurfaceParams: specular components must be >= 0",
            ));
        }
        if self.shininess < 0.0 {
            return Err(RenderError::from(
                "SurfaceParams: shininess must be >= 0",
            ));
        }
        Ok(())
    }
}

/// The material's shader plus its parameter set. A value is the engine's
/// uniform value, or a sampler binding.
#[derive(Clone)]
pub struct Material {
    pub surface: SurfaceParams,
    pub shader: Option<ShaderProgram>,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            surface: SurfaceParams {
                albedo: Vec3::new(1.0, 1.0, 1.0),
                specular: Vec3::new(0.0, 0.0, 0.0),
                shininess: 0.0,
            },
            shader: None,
        }
    }
}

/// Directional light (the sun): direction the light travels in, colour and
/// intensity.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl DirectionalLight {
    /// Throws `RenderError` on out-of-range or degenerate values.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.direction.length() == 0.0 {
            return Err(RenderError::from(
                "DirectionalLight: direction must be non-zero",
            ));
        }
        if self.color.x < 0.0 || self.color.y < 0.0 || self.color.z < 0.0 {
            return Err(RenderError::from(
                "DirectionalLight: colour components must be >= 0",
            ));
        }
        if self.intensity < 0.0 {
            return Err(RenderError::from(
                "DirectionalLight: intensity must be >= 0",
            ));
        }
        Ok(())
    }
}

/// Point light with a windowed inverse-square attenuation model.
#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
}

impl PointLight {
    /// Throws `RenderError` on out-of-range or degenerate values.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.color.x < 0.0 || self.color.y < 0.0 || self.color.z < 0.0 {
            return Err(RenderError::from(
                "PointLight: colour components must be >= 0",
            ));
        }
        if self.intensity < 0.0 {
            return Err(RenderError::from(
                "PointLight: intensity must be >= 0",
            ));
        }
        if self.radius <= 0.0 {
            return Err(RenderError::from("PointLight: radius must be > 0"));
        }
        Ok(())
    }
}

/// Spot light with a smoothstep cone falloff.
#[derive(Clone, Copy, Debug)]
pub struct SpotLight {
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub outer_angle_rad: f32,
    pub inner_angle_rad: f32,
}

impl SpotLight {
    /// Throws `RenderError` on out-of-range or degenerate values.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.color.x < 0.0 || self.color.y < 0.0 || self.color.z < 0.0 {
            return Err(RenderError::from(
                "SpotLight: colour components must be >= 0",
            ));
        }
        if self.intensity < 0.0 {
            return Err(RenderError::from(
                "SpotLight: intensity must be >= 0",
            ));
        }
        if self.range <= 0.0 {
            return Err(RenderError::from("SpotLight: range must be > 0"));
        }
        if self.outer_angle_rad <= 0.0 || self.outer_angle_rad > std::f32::consts::PI {
            return Err(RenderError::from(
                "SpotLight: outer angle must be in (0, pi]",
            ));
        }
        if self.inner_angle_rad <= 0.0 || self.inner_angle_rad >= self.outer_angle_rad {
            return Err(RenderError::from(
                "SpotLight: inner angle must be in (0, outer angle)",
            ));
        }
        Ok(())
    }
}

/// A dynamic light: point or spot.
#[derive(Clone, Copy, Debug)]
pub enum DynamicLight {
    Point(PointLight),
    Spot(SpotLight),
}

impl DynamicLight {
    /// Throws `RenderError` on out-of-range or degenerate values.
    pub fn validate(&self) -> Result<(), RenderError> {
        match self {
            DynamicLight::Point(p) => p.validate(),
            DynamicLight::Spot(s) => s.validate(),
        }
    }
}
