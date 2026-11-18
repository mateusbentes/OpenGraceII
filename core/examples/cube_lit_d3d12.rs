//! The milestone 5 integration example: the lit cube drawn through the
//! generic path with a directional sun, the pixel readback on known
//! points, on the Direct3D 12 backend. The same checks the other
//! backends' cube_lit runs (the same honest pattern: the generic path,
//! not a special-case shortcut). Without the platform's d3d12.dll the
//! loader reports that cleanly and the run skips (exit 77, the same
//! skip the triangle smoke follows).
use og2_core::mesh::{cube, pnu_layout, PNU_STRIDE};
use og2_core::render::d3d12::{D3d12Backend, D3d12BackendConfig};
use og2_core::render::{
    CameraParams, ClearBuffer, ClearColor, DirectionalLight, DrawSubmission,
    Material, PixelFormat, RenderBackend, Texture2DDesc,
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
    backend.set_clear_color(ClearColor { r: 30, g: 40, b: 50, a: 255 });
    // The texture: a flat mid-grey. With a flat albedo the shading's
    // differences across the faces come from the lighting alone.
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
    // The generic path draws without a named program (the submission
    // carries the mesh; the lit pipeline the upload ensured serves it).
    let _program = backend.create_shader_program(1).expect("program");
    // The sun shines towards -Z (the cube's front face's normal points
    // at +Z, so the front face's NdotL is 1.0). The ambient-free Blinn-
    // Phong makes the back face's sun term zero.
    let submission = DrawSubmission {
        mesh: mesh.clone(),
        model: Mat4::IDENTITY,
        material: Material::default(),
        sun: Some(DirectionalLight {
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 0.5,
        }),
        lights: Vec::new(),
    };
    // The lit submission is the frame's only draw (the unlit draw()
    // call the M4 smokes make would paint the cube first and the lit
    // draw's equal-depth LESS test would lose the pixels).
    backend.clear();
    backend.submit(&submission);
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
    backend.shutdown();
    let centre = samples
        .iter()
        .find(|s| s.x == 160 && s.y == 120)
        .expect("the centre sample");
    let corner = samples
        .iter()
        .find(|s| s.x == 4 && s.y == 4)
        .expect("the corner sample");
    let ok = samples.len() >= 2
        && centre.color.r > 60
        && centre.color.r < 140
        && (centre.color.r as i32 - centre.color.g as i32).abs() < 30
        && centre.depth < 0.99
        && corner.color.r == 30
        && corner.color.g == 40
        && corner.color.b == 50
        && backend.violation_count() == 0;
    if ok {
        println!("OK");
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
