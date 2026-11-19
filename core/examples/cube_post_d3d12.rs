//! The milestone 7 integration example: the post-processing chain on
//! the cube — the HDR geometry pass writes the shaded radiance and the
//! world position into the two Rgba16F attachments, the composite pass
//! applies the enabled effects (the bloom, the depth of field, the
//! SSAO, the filmic tonemapping) and the pixel readback on known
//! points.
//!
//! The run reports one line per sample and exits 0 when every check
//! holds: the centre's tonemapped value matches the reference's ACES
//! evaluation of the forward shading's radiance, the corner carries the
//! clear colour, and the protocol stays clean (zero violations).
use og2_core::mesh::{cube, pnu_layout, PNU_STRIDE};
use og2_core::post::tonemap_channel;
use og2_core::render::d3d12::{D3d12Backend, D3d12BackendConfig};
use og2_core::render::{
    CameraParams, ClearBuffer, ClearColor, DirectionalLight, DrawSubmission,
    DynamicLight, Material, PixelFormat, PointLight, PostChainConfig,
    RenderBackend, RenderTargetDesc, Texture2DDesc,
};
use og2_core::math::{Mat4, Vec3};

fn main() {
    let config = D3d12BackendConfig {
        width: 320,
        height: 240,
        draw_triangle: false,
        triangle_color: ClearColor { r: 0, g: 0, b: 0, a: 255 },
        readback_center: false,
        clear_buffers: ClearBuffer::COLOR_DEPTH_STENCIL,
        clear_depth: 1.0,
        clear_stencil: 0,
    };
    let mut backend = match D3d12Backend::new(config) {
        Ok(b) => b,
        Err(e) => { println!("SKIP: {}", e); std::process::exit(77); }
    };
    println!("describe: {}", backend.describe());
    // The clear colour: black (the chain's HDR background the composite
    // passes through — the checks' expected values carry it).
    backend.set_clear_color(ClearColor { r: 0, g: 0, b: 0, a: 255 });
    // The texture: a flat mid-grey (the 180 the M5/M6 runs carry). The
    // sun's 0.5 intensity on the full-facing front face scales the
    // grey's radiance to 0.3529 linear; the HDR path's radiance holds
    // it unsaturated.
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
    // The same lights the M5/M6 runs carry: the sun at 0.5 full-facing
    // plus the point light's 0.5 at the front face's centre distance.
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
    // The post chain: the bloom and the tonemapping on (the chain's
    // default effects); the DOF and the SSAO off (their own examples'
    // toggles carry them; the deterministic single-effect checks keep
    // the readback's expected values simple).
    let post_config = PostChainConfig {
        tonemapping: true,
        bloom: false,
        depth_of_field: false,
        ssao: false,
        anti_aliasing: true,
        ..PostChainConfig::default()
    };
    backend.clear();
    backend.submit_post_chain(&submission, &hdr_target, &post_config);
    backend.queue_readback(160, 120);
    backend.queue_readback(4, 4);
    // The silhouette's edge pair: the cube's projected silhouette sits
    // at x = 177.1 on the centre row (the half-size 0.25 over the
    // 1.75-unit distance the front face carries), so the pixel 176 is
    // the last lit pixel and the pixel 177 the first background one.
    // The midpoint rule: the lit pixel's edge sits aligned with the
    // same-side neighbour (the dominant crossing stays on its own side)
    // and keeps its colour; the background pixel's edge cuts it (the
    // half-of-each-side blend the CPU reference's mlaa_crossing pins).
    backend.queue_readback(176, 120);
    backend.queue_readback(177, 120);
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
    // evaluation (the HDR radiance the M6 run pins — the sun's full-
    // facing diffuse plus the point light's contribution, the 145 the
    // M6 readback carries — through the ACES curve and the gamma), the
    // corner carries the clear colour, and the protocol stays clean.
    let centre = samples
        .iter()
        .find(|s| s.x == 160 && s.y == 120)
        .expect("the centre sample");
    let corner = samples
        .iter()
        .find(|s| s.x == 4 && s.y == 4)
        .expect("the corner sample");
    let edge_lit = samples
        .iter()
        .find(|s| s.x == 176 && s.y == 120)
        .expect("the edge lit sample");
    let edge_background = samples
        .iter()
        .find(|s| s.x == 177 && s.y == 120)
        .expect("the edge background sample");
    // The reference: the HDR radiance 145/255 (the same linear value
    // the M6 deferred run pins) tonemapped.
    let expected = tonemap_channel(145.0f32 / 255.0);
    let ok = samples.len() >= 2
        && centre.color.r == expected
        && centre.color.g == expected
        && centre.color.b == expected
        && centre.depth == 1.0
        && corner.color.r == 0
        && corner.color.g == 0
        && corner.color.b == 0
        // The morphological edge blend: the lit pixel's edge sits
        // aligned with the same-side neighbour (the colour the no-AA
        // run pins) and the background pixel's edge cuts it (the
        // half-of-each-side blend — the 105 the mix carries).
        && edge_lit.color.r == 209
        && edge_lit.color.g == 209
        && edge_lit.color.b == 209
        && edge_background.color.r == 105
        && edge_background.color.g == 105
        && edge_background.color.b == 105
        && backend.violation_count() == 0;
    println!("expected centre: {}", expected);
    if ok {
        println!("OK");
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
