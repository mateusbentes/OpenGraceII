//! The milestone 8 (OpenGL) integration example: the light volume on the post
//! chain — the CPU builds the volume (the point light injected, the
//! propagation steps run), the CPU packs it into the Rgba16F texture
//! the `set_light_volume` call carries, and the post chain's g-buffer
//! pass adds the indirect term (the albedo times the volume's sampled
//! radiance) to the direct shading before the composite pass reads.
//!
//! The run reports one line per sample and exits 0 when every check
//! holds: the centre's tonemapped value matches the reference's ACES
//! evaluation of the direct shading plus the indirect term the CPU
//! reference's `sample` carries, the corner carries the clear colour,
//! and the protocol stays clean (zero violations).
use og2_core::lightvolume::LightVolume;
use og2_core::mesh::{cube, pnu_layout, PNU_STRIDE};
use og2_core::post::tonemap_channel;
use og2_core::render::opengl::{OpenGlBackend, OpenGlBackendConfig};
use og2_core::render::{
    CameraParams, ClearBuffer, ClearColor, DirectionalLight, DrawSubmission,
    DynamicLight, Material, PixelFormat, PointLight, PostChainConfig,
    RenderBackend, RenderTargetDesc, Texture2DDesc, Texture3DDesc,
};
use og2_core::math::{Mat4, Vec3};
fn main() {
    let config = OpenGlBackendConfig {
        width: 320,
        height: 240,
        draw_triangle: false,
        triangle_color: ClearColor { r: 0, g: 0, b: 0, a: 255 },
        readback_center: false,
        clear_buffers: ClearBuffer::COLOR_DEPTH_STENCIL,
        clear_depth: 1.0,
        clear_stencil: 0,
    };
    let mut backend = match OpenGlBackend::new(config) {
        Ok(b) => b,
        Err(e) => { println!("SKIP: {}", e); std::process::exit(77); }
    };
    println!("describe: {}", backend.describe());
    // The clear colour: black (the chain's HDR background the composite
    // passes through — the checks' expected values carry it).
    backend.set_clear_color(ClearColor { r: 0, g: 0, b: 0, a: 255 });
    // The texture: a flat mid-grey (the 180 the M5/M6 runs carry).
    let texture = backend
        .create_texture_2d(&Texture2DDesc {
            width: 4,
            height: 4,
            format: PixelFormat::Rgba8,
        })
        .expect("create_texture_2d");
    let texels = vec![180u8, 180, 180, 255].repeat(16);
    backend.fill_texture_2d(&texture, &texels).expect("fill_texture_2d");
    backend.bind_texture(Some(&texture), 0);
    backend.set_camera(CameraParams {
        eye: Vec3::new(0.0, 0.0, 2.0),
        target: Vec3::new(0.0, 0.0, 0.0),
        up: Vec3::new(0.0, 1.0, 0.0),
        fov_y_rad: std::f32::consts::FRAC_PI_2,
        aspect: 320.0 / 240.0,
        near: 0.1,
        far: 100.0,
    });
    let mesh_data = cube(0.5);
    let layout = pnu_layout();
    assert_eq!(layout.stride, PNU_STRIDE);
    let mesh = backend.create_gpu_mesh().expect("create_gpu_mesh");
    backend
        .upload_mesh(&mesh, &layout, &mesh_data.vertices, &mesh_data.indices)
        .expect("upload_mesh");
    let _program = backend.create_shader_program(1).expect("program");
    // The HDR target: the two Rgba16F attachments (the radiance and the
    // world position) plus the depth attachment.
    let hdr_target = backend
        .create_render_target(&RenderTargetDesc {
            width: 320,
            height: 240,
            color_attachment_count: 2,
            color_formats: [PixelFormat::Rgba16F, PixelFormat::Rgba16F, PixelFormat::Rgba8, PixelFormat::Rgba8],
            depth_attachment: true,
            depth_format: PixelFormat::Depth24Stencil8,
        })
        .expect("create_render_target");
    // The lights the M5/M6 runs carry: the sun at 0.5 full-facing plus
    // the point light's 0.5 at the front face's centre distance.
    let submission = DrawSubmission {
        mesh: mesh.clone(),
        model: Mat4::IDENTITY,
        material: Material::default(),
        sun: Some(DirectionalLight {
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 0.5,
        }),
        lights: vec![DynamicLight::Point(PointLight {
            position: Vec3::new(0.0, 0.0, 1.0),
            radius: 5.0,
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 0.5,
        })],
    };
    // The volume: the CPU reference builds the same grid the shader
    // samples — the point light injected at the same position, the
    // propagation steps run, and the packed Rgba16F texture the
    // `set_light_volume` call carries.
    let mut volume = LightVolume::new(1.0, [-2.0, -2.0, -2.0], [4, 4, 4]);
    volume.inject_point_light(
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        0.5,
        5.0,
    );
    volume.propagate(1);
    let volume_texture = backend
        .create_texture_3d(&Texture3DDesc {
            width: 4,
            height: 4,
            depth: 24,
            format: PixelFormat::Rgba16F,
        })
        .expect("create_texture_3d");
    let volume_data = volume.pack_texture_3d();
    backend
        .fill_texture_3d(&volume_texture, &volume_data)
        .expect("fill_texture_3d");
    backend.set_light_volume(&volume);
    // The post chain: the tonemapping on (the chain's default effect);
    // the bloom, the DOF and the SSAO off (the deterministic
    // single-effect checks keep the readback's expected values simple).
    let post_config = PostChainConfig {
        tonemapping: true,
        bloom: false,
        depth_of_field: false,
        ssao: false,
        anti_aliasing: false,
        indirect_light: true,
        indirect_intensity: 0.5,
        ..PostChainConfig::default()
    };
    backend.clear();
    backend.submit_post_chain(&submission, &hdr_target, &post_config);
    backend.queue_readback(160, 120);
    backend.queue_readback(4, 4);
    backend.swap_buffers();
    let samples: Vec<_> = backend.last_readback().to_vec();
    for sample in &samples {
        println!(
            "  ({}, {}) = {:?} depth {:.4}",
            sample.x, sample.y, sample.color, sample.depth
        );
    }
    println!("violations: {}", backend.violation_count());
    backend.release_render_target(&hdr_target);
    backend.shutdown();
    // The checks: the centre's tonemapped grey matches the reference's
    // evaluation — the direct shading's radiance the M6 run pins (the
    // 145/255 the sun's full-facing diffuse plus the point light's
    // contribution carry) plus the indirect term (the albedo's 180/255
    // times the volume's sampled radiance times the intensity) through
    // the ACES curve and the gamma; the corner carries the clear
    // colour; the protocol stays clean.
    let centre = samples
        .iter()
        .find(|s| s.x == 160 && s.y == 120)
        .expect("the centre sample");
    let corner = samples
        .iter()
        .find(|s| s.x == 4 && s.y == 4)
        .expect("the corner sample");
    // The CPU reference: the direct radiance plus the indirect term
    // (the volume's sample at the front face's centre — the world
    // position (0, 0, 0.25) and the normal (0, 0, 1) the camera's
    // front face carries — the 0.5 the example's indirect intensity
    // carries is the default's value).
    let albedo = 180.0 / 255.0;
    let direct = 145.0 / 255.0;
    let indirect = volume.sample([0.0, 0.0, 0.25], [0.0, 0.0, 1.0]);
    let indirect_term = albedo * indirect[0] * 0.5;
    let expected = tonemap_channel(direct + indirect_term);
    let ok = samples.len() >= 2
        && centre.color.r == expected
        && centre.color.g == expected
        && centre.color.b == expected
        && centre.depth == 1.0
        && corner.color.r == 0
        && corner.color.g == 0
        && corner.color.b == 0
        && corner.depth == 1.0
        && backend.violation_count() == 0;
    println!("expected centre: {}", expected);
    if ok {
        println!("OK");
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
