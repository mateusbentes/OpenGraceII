//! The null render backend: a deterministic, headless implementation of the
//! render contract.
//!
//! The null backend records every call in order (an event log) and counts
//! protocol violations. It is the reference the contract checks run against
//! on every machine — the same role OpenGrace's null backend played.

use super::*;

/// The null backend's event log entry.
#[derive(Clone, Debug, PartialEq)]
pub enum NullEvent {
    SetClearColor(ClearColor),
    SetDepthTestEnabled(bool),
    Clear(ClearColor),
    CreateShaderProgram(String),
    CreateGpuMesh,
    UploadMesh,
    Draw,
    Submit,
    CreateRenderTarget,
    BeginRenderPass,
    EndRenderPass,
    SubmitLightingPass,
    SubmitPostChain,
    ReleaseRenderTarget,
    BindTexture,
    CreateTexture2D,
    ReleaseTexture2D,
    FillTexture2D,
    ReadTexture2D,
    CreateTexture3D,
    ReleaseTexture3D,
    FillTexture3D,
    ReadTexture3D,
    SetLightVolume,
    SetCamera,
    Shutdown,
    Violation(&'static str),
}

/// The null backend's own texture view: plain data, one byte vector per
/// texture, so a fill and a read round-trip the same values with no graphics
/// API.
pub struct NullTexture2D {
    pub inner: Arc<Mutex<TextureViewInner>>,
    pub bytes: Vec<u8>,
    pub released: bool,
}

/// The null backend's own 3D texture view: plain data, one byte vector
/// per texture, the same rule the 2D textures carry.
pub struct NullTexture3D {
    pub inner: Arc<Mutex<TextureViewInner>>,
    pub bytes: Vec<u8>,
    pub released: bool,
}

/// The null backend's own render target: records its desc and release state.
pub struct NullRenderTarget {
    pub inner: Arc<Mutex<RenderTargetInner>>,
    pub views: Vec<NullTexture2D>,
    pub depth_view: Option<NullTexture2D>,
    pub released: bool,
}

/// The headless render backend.
pub struct NullRenderBackend {
    width: u32,
    height: u32,
    clear_color: ClearColor,
    depth_test_enabled: bool,
    camera: CameraParams,
    bound_target: Option<Arc<Mutex<RenderTargetInner>>>,
    shutdown: bool,
    violations: u64,
    events: Vec<NullEvent>,
    targets: Vec<NullRenderTarget>,
    textures_2d: Vec<NullTexture2D>,
    textures_3d: Vec<NullTexture3D>,
    light_volume: Option<NullTexture3D>,
    next_mesh_id: usize,
    readback_queue: Vec<(u32, u32)>,
    last_readback: Vec<ReadbackSample>,
    has_center: bool,
    center: ClearColor,
    center_depth: f32,
    center_stencil: i32,
}

impl NullRenderBackend {
    /// Throws `RenderError` when width or height is zero.
    pub fn new(width: u32, height: u32) -> Result<Self, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::from(
                "NullRenderBackend: width and height must be positive",
            ));
        }
        Ok(NullRenderBackend {
            width,
            height,
            clear_color: ClearColor {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            depth_test_enabled: true,
            camera: CameraParams {
                eye: Vec3::new(0.0, 0.0, 1.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                up: Vec3::new(0.0, 1.0, 0.0),
                fov_y_rad: std::f32::consts::PI / 2.0,
                aspect: 1.0,
                near: 0.1,
                far: 100.0,
            },
            bound_target: None,
            shutdown: false,
            violations: 0,
            events: Vec::new(),
            targets: Vec::new(),
            textures_2d: Vec::new(),
            textures_3d: Vec::new(),
            light_volume: None,
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
        })
    }

    fn record(&mut self, event: NullEvent) {
        self.events.push(event);
    }

    fn require_ready(&mut self, what: &'static str) -> bool {
        if self.shutdown {
            self.violations += 1;
            self.events.push(NullEvent::Violation(what));
            false
        } else {
            true
        }
    }

    /// The recorded call sequence, in order.
    pub fn events(&self) -> &[NullEvent] {
        &self.events
    }

    fn find_target_index(&self, target: &RenderTarget) -> Option<usize> {
        self.targets
            .iter()
            .position(|t| Arc::ptr_eq(&t.inner, &target.inner))
    }

    fn find_texture_index(&self, view: &TextureView) -> Option<usize> {
        self.textures_2d
            .iter()
            .position(|t| Arc::ptr_eq(&t.inner, &view.inner))
    }

    fn find_texture3d_index(&self, view: &TextureView) -> Option<usize> {
        self.textures_3d
            .iter()
            .position(|t| Arc::ptr_eq(&t.inner, &view.inner))
    }
}

impl RenderBackend for NullRenderBackend {
    fn describe(&self) -> String {
        "null (headless)".to_string()
    }

    fn set_clear_color(&mut self, color: ClearColor) {
        self.record(NullEvent::SetClearColor(color));
        if self.require_ready("setClearColor") {
            self.clear_color = color;
        }
    }

    fn clear_color(&self) -> ClearColor {
        self.clear_color
    }

    fn set_depth_test_enabled(&mut self, enabled: bool) {
        self.record(NullEvent::SetDepthTestEnabled(enabled));
        if self.require_ready("setDepthTestEnabled") {
            self.depth_test_enabled = enabled;
        }
    }

    fn clear(&mut self) {
        self.record(NullEvent::Clear(self.clear_color));
        if self.require_ready("clear") {
            // The clear colour is modelled into the bound target's colour
            // attachment views, the same way the readback path models the
            // clear colour into the centre pixel.
            if let Some(index) = self
                .bound_target
                .as_ref()
                .and_then(|t| self.targets.iter().position(|x| Arc::ptr_eq(&x.inner, t)))
            {
                if let Some(view) = self.targets[index].views.first_mut() {
                    let bpp = bytes_per_pixel(view.inner.lock().unwrap().format) as usize;
                    for i in (0..view.bytes.len()).step_by(bpp) {
                        view.bytes[i] = self.clear_color.r;
                        if bpp > 1 {
                            view.bytes[i + 1] = self.clear_color.g;
                        }
                        if bpp > 2 {
                            view.bytes[i + 2] = self.clear_color.b;
                        }
                        if bpp > 3 {
                            view.bytes[i + 3] = self.clear_color.a;
                        }
                    }
                }
            }
        }
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn shutdown(&mut self) {
        self.record(NullEvent::Shutdown);
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
        self.record(NullEvent::CreateShaderProgram(format!("id {}", id)));
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
        self.record(NullEvent::CreateGpuMesh);
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
        self.record(NullEvent::UploadMesh);
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
                        ComponentType::Float32 => 4,
                        ComponentType::Uint8Normalized => 1,
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
        self.record(NullEvent::Draw);
        if self.require_ready("draw") {
            let inner = mesh.inner.lock().unwrap();
            if !inner.uploaded {
                self.violations += 1;
            }
        }
    }

    fn submit(&mut self, submission: &DrawSubmission) {
        self.record(NullEvent::Submit);
        if self.require_ready("submit") {
            if submission.sun.is_none() && submission.lights.is_empty() {
                // The contract permits a submission with no lights; the
                // null backend records the event and draws.
            }
            for light in &submission.lights {
                if light.validate().is_err() {
                    self.violations += 1;
                }
            }
        }
    }

    fn submit_lighting_pass(
        &mut self,
        submission: &DrawSubmission,
        gbuffer: &RenderTarget,
    ) {
        self.record(NullEvent::SubmitLightingPass);
        if self.require_ready("submit_lighting_pass") {
            // The G-buffer must be a target this backend owns and has not
            // released: sampling a foreign or released target is a protocol
            // violation (the same rule the target handles follow).
            match self.find_target_index(gbuffer) {
                Some(index) => {
                    if self.targets[index].released {
                        self.violations += 1;
                    }
                }
                None => {
                    self.violations += 1;
                }
            }
            for light in &submission.lights {
                if light.validate().is_err() {
                    self.violations += 1;
                }
            }
            let _ = &submission.sun;
            let _ = &submission.mesh;
        }
    }
    fn submit_post_chain(
        &mut self,
        submission: &DrawSubmission,
        hdr_target: &RenderTarget,
        config: &PostChainConfig,
    ) {
        self.record(NullEvent::SubmitPostChain);
        let _ = config;
        if self.require_ready("submit_post_chain") {
            match self.find_target_index(hdr_target) {
                Some(index) => {
                    if self.targets[index].released {
                        self.violations += 1;
                    }
                }
                None => {
                    self.violations += 1;
                }
            }
            let _ = submission;
        }
    }

    fn create_render_target(
        &mut self,
        desc: &RenderTargetDesc,
    ) -> Result<RenderTarget, RenderError> {
        self.record(NullEvent::CreateRenderTarget);
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "create_render_target: the backend is shut down",
            ));
        }
        desc.validate()?;
        let mut views = Vec::new();
        for i in 0..desc.color_attachment_count as usize {
            let bpp = bytes_per_pixel(desc.color_formats[i]) as usize;
            views.push(NullTexture2D {
                inner: Arc::new(Mutex::new(TextureViewInner {
                    width: desc.width,
                    height: desc.height,
                    depth: 1,
                    format: desc.color_formats[i],
                    is_depth: false,
                    released: false,
                })),
                bytes: vec![0; (desc.width as usize) * (desc.height as usize) * bpp],
                released: false,
            });
        }
        let depth_view = if desc.depth_attachment {
            let bpp = bytes_per_pixel(desc.depth_format) as usize;
            Some(NullTexture2D {
                inner: Arc::new(Mutex::new(TextureViewInner {
                    width: desc.width,
                    height: desc.height,
                    depth: 1,
                    format: desc.depth_format,
                    is_depth: true,
                    released: false,
                })),
                bytes: vec![0; (desc.width as usize) * (desc.height as usize) * bpp],
                released: false,
            })
        } else {
            None
        };
        let target = NullRenderTarget {
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
            inner: target.inner.clone(),
        };
        self.targets.push(target);
        Ok(handle)
    }

    fn begin_render_pass(&mut self, target: Option<&RenderTarget>) {
        self.record(NullEvent::BeginRenderPass);
        if self.require_ready("begin_render_pass") {
            match target {
                None => {
                    self.bound_target = None;
                }
                Some(t) => {
                    match self.find_target_index(t) {
                        Some(index) => {
                            if self.targets[index].released {
                                self.violations += 1;
                            } else {
                                self.bound_target = Some(t.inner.clone());
                            }
                        }
                        None => {
                            self.violations += 1;
                        }
                    }
                }
            }
        }
    }

    fn end_render_pass(&mut self) {
        self.record(NullEvent::EndRenderPass);
        if self.require_ready("end_render_pass") {
            self.bound_target = None;
        }
    }

    fn release_render_target(&mut self, target: &RenderTarget) {
        self.record(NullEvent::ReleaseRenderTarget);
        if self.require_ready("release_render_target") {
            match self.find_target_index(target) {
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
    }

    fn render_target_texture(&self, target: &RenderTarget) -> Option<TextureView> {
        match self.find_target_index(target) {
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
        match self.find_target_index(target) {
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
                if attachment < t.views.len() as u32 {
                    Some(TextureView {
                        inner: t.views[attachment as usize].inner.clone(),
                    })
                } else {
                    None
                }
            }
            None => None,
        }
    }

    fn bind_texture(&mut self, _view: Option<&TextureView>, _unit: u32) {
        self.record(NullEvent::BindTexture);
        if self.require_ready("bind_texture") {
            // The null backend records the binding; the null model has no
            // sampler state to validate.
        }
    }

    fn create_texture_2d(&mut self, desc: &Texture2DDesc) -> Result<TextureView, RenderError> {
        self.record(NullEvent::CreateTexture2D);
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
        let bpp = bytes_per_pixel(desc.format) as usize;
        let texture = NullTexture2D {
            inner: Arc::new(Mutex::new(TextureViewInner {
                width: desc.width,
                height: desc.height,
                depth: 1,
                format: desc.format,
                is_depth: false,
                released: false,
            })),
            bytes: vec![0; (desc.width as usize) * (desc.height as usize) * bpp],
            released: false,
        };
        let handle = TextureView {
            inner: texture.inner.clone(),
        };
        self.textures_2d.push(texture);
        Ok(handle)
    }

    fn release_texture_2d(&mut self, view: &TextureView) {
        self.record(NullEvent::ReleaseTexture2D);
        if self.require_ready("release_texture_2d") {
            match self.find_texture_index(view) {
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
    }

    fn fill_texture_2d(&mut self, view: &TextureView, data: &[u8]) -> Result<(), RenderError> {
        self.record(NullEvent::FillTexture2D);
        if self.shutdown {
            self.violations += 1;
            return Err(RenderError::from(
                "fill_texture_2d: the backend is shut down",
            ));
        }
        let index = match self.find_texture_index(view) {
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
        let expected =
            (inner.width as usize) * (inner.height as usize) * bytes_per_pixel(inner.format) as usize;
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "fill_texture_2d: the byte count is {} but the texture is {}",
                data.len(),
                expected
            )));
        }
        self.textures_2d[index].bytes = data.to_vec();
        Ok(())
    }

    fn read_texture_2d(
        &self,
        view: &TextureView,
        data: &mut [u8],
    ) -> Result<bool, RenderError> {
        let index = match self.find_texture_index(view) {
            Some(i) => i,
            None => {
                return Err(RenderError::from(
                    "read_texture_2d: the texture is not owned by this backend",
                ));
            }
        };
        if self.textures_2d[index].released {
            return Err(RenderError::from(
                "read_texture_2d: the texture is released",
            ));
        }
        let inner = view.inner.lock().unwrap();
        let expected =
            (inner.width as usize) * (inner.height as usize) * bytes_per_pixel(inner.format) as usize;
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "read_texture_2d: the byte count is {} but the texture is {}",
                data.len(),
                expected
            )));
        }
        data.copy_from_slice(&self.textures_2d[index].bytes);
        Ok(true)
    }

    fn create_texture_3d(&mut self, desc: &Texture3DDesc)
        -> Result<TextureView, RenderError> {
        self.record(NullEvent::CreateTexture3D);
        if let Err(_error) = desc.validate() {
            self.violations += 1;
            return Err(RenderError::from(
                "create_texture_3d: the desc is invalid",
            ));
        }
        let count = (desc.width as usize) * (desc.height as usize)
            * (desc.depth as usize) * bytes_per_pixel(desc.format) as usize;
        let inner = Arc::new(Mutex::new(TextureViewInner {
            width: desc.width,
            height: desc.height,
            depth: desc.depth,
            format: desc.format,
            is_depth: false,
            released: false,
        }));
        let texture = NullTexture3D {
            inner: inner.clone(),
            bytes: vec![0u8; count],
            released: false,
        };
        self.textures_3d.push(texture);
        Ok(TextureView { inner })
    }

    fn release_texture_3d(&mut self, view: &TextureView) {
        self.record(NullEvent::ReleaseTexture3D);
        let index = match self.find_texture3d_index(view) {
            Some(i) => i,
            None => {
                self.violations += 1;
                return;
            }
        };
        if self.textures_3d[index].released {
            self.violations += 1;
            return;
        }
        self.textures_3d[index].released = true;
        view.inner.lock().unwrap().released = true;
    }

    fn fill_texture_3d(&mut self, view: &TextureView, data: &[u8])
        -> Result<(), RenderError> {
        self.record(NullEvent::FillTexture3D);
        let index = match self.find_texture3d_index(view) {
            Some(i) => i,
            None => {
                return Err(RenderError::from(
                    "fill_texture_3d: the texture is not owned by this backend",
                ));
            }
        };
        if self.textures_3d[index].released {
            return Err(RenderError::from(
                "fill_texture_3d: the texture is released",
            ));
        }
        let inner = view.inner.lock().unwrap();
        let expected = (inner.width as usize) * (inner.height as usize)
            * (inner.depth as usize) * bytes_per_pixel(inner.format) as usize;
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "fill_texture_3d: the byte count is {} but the texture is {}",
                data.len(),
                expected
            )));
        }
        self.textures_3d[index].bytes = data.to_vec();
        Ok(())
    }

    fn read_texture_3d(
        &mut self,
        view: &TextureView,
        data: &mut [u8],
    ) -> Result<bool, RenderError> {
        let index = match self.find_texture3d_index(view) {
            Some(i) => i,
            None => {
                return Err(RenderError::from(
                    "read_texture_3d: the texture is not owned by this backend",
                ));
            }
        };
        if self.textures_3d[index].released {
            return Err(RenderError::from(
                "read_texture_3d: the texture is released",
            ));
        }
        let inner = view.inner.lock().unwrap();
        let expected = (inner.width as usize) * (inner.height as usize)
            * (inner.depth as usize) * bytes_per_pixel(inner.format) as usize;
        drop(inner);
        if data.len() != expected {
            return Err(RenderError::from(format!(
                "read_texture_3d: the byte count is {} but the texture is {}",
                data.len(),
                expected
            )));
        }
        data.copy_from_slice(&self.textures_3d[index].bytes);
        Ok(true)
    }

    fn set_light_volume(&mut self, volume: &LightVolume) {
        self.record(NullEvent::SetLightVolume);
        if self.require_ready("set_light_volume") {
            if volume.dims[0] == 0
                || volume.dims[1] == 0
                || volume.dims[2] == 0
                || volume.cell_size <= 0.0
            {
                self.violations += 1;
                return;
            }
            // The volume's own 3D texture: the same plain-data path the
            // 2D textures carry (the bytes carry the voxels' RGBA16F
            // halves, the depth layers `z * 6 + f`).
            let width = volume.dims[0];
            let height = volume.dims[1];
            let depth = volume.dims[2] * 6;
            let bytes = width as usize
                * height as usize
                * depth as usize
                * bytes_per_pixel(PixelFormat::Rgba16F) as usize;
            let texture = NullTexture3D {
                inner: Arc::new(Mutex::new(TextureViewInner {
                    width,
                    height,
                    depth,
                    format: PixelFormat::Rgba16F,
                    is_depth: false,
                    released: false,
                })),
                bytes: volume.pack_texture_3d(),
                released: false,
            };
            let _ = bytes;
            self.light_volume = Some(texture);
        }
    }

    fn set_camera(&mut self, camera: CameraParams) {
        self.record(NullEvent::SetCamera);
        if self.require_ready("set_camera") {
            if let Err(_error) = camera.validate() {
                self.violations += 1;
            } else {
                self.camera = camera;
            }
        }
    }

    fn camera(&self) -> CameraParams {
        self.camera
    }

    fn swap_buffers(&mut self) {
        if self.shutdown {
            self.violations += 1;
        }
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
        // The null backend's documented value semantics: every pixel reports
        // exactly the current clear colour (depth 1.0, stencil 0).
        self.last_readback.clear();
        let mut queue = std::mem::take(&mut self.readback_queue);
        if queue.is_empty() {
            self.has_center = true;
            self.center = self.clear_color;
            self.center_depth = 1.0;
            self.center_stencil = 0;
        }
        for (x, y) in queue.drain(..) {
            self.last_readback.push(ReadbackSample {
                x,
                y,
                color: self.clear_color,
                depth: 1.0,
                stencil: 0,
            });
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
