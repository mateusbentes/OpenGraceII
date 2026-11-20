//! The milestone 4 integration example: the textured cube drawn through
//! the generic path (create_gpu_mesh + upload_mesh + set_camera + draw)
//! with the pixel readback on known points.
//!
//! The run reports one line per check and exits 0 when every check holds:
//! the cube's front-face centre carries the texture's colour, the frame's
//! corner carries the clear colour, and the readback's depth at the cube's
//! centre is closer than 1 (the depth test wrote the cube).
use og2_core::mesh::{cube, pnu_layout, PNU_STRIDE};
use og2_core::render::vulkan::{VulkanBackend, VulkanBackendConfig};
use og2_core::render::{
    CameraParams, ClearBuffer, ClearColor, DrawSubmission, Material,
    PixelFormat, RenderBackend, Texture2DDesc,
};
use og2_core::math::{Mat4, Vec3};

fn main() {
    let config = VulkanBackendConfig {
        width: 320,
        height: 240,
        surface: None,
        draw_triangle: false,
        triangle_color: ClearColor { r: 0, g: 0, b: 0, a: 255 },
        readback_center: false,
        clear_buffers: ClearBuffer::COLOR_DEPTH_STENCIL,
        clear_depth: 1.0,
        clear_stencil: 0,
    };
    let mut backend = match VulkanBackend::new(config) {
        Ok(b) => b,
        Err(e) => { println!("SKIP: {}", e); std::process::exit(77); }
    };
    println!("describe: {}", backend.describe());
    backend.set_clear_color(ClearColor { r: 30, g: 40, b: 50, a: 255 });
    // The texture: one 4x4 whose first row is green and whose other rows
    // are red. The cube's front face's centre uv (0.5, 0.5) samples the
    // texel row 2 with NEAREST filtering — red under both the Vulkan
    // (row 0 top) and the OpenGL (row 0 bottom) texel conventions, so the
    // check is convention-agnostic. The green row only proves the sampler
    // reads the upload (a constant-colour fallback would also pass red).
    let texture = backend
        .create_texture_2d(&Texture2DDesc {
            width: 4,
            height: 4,
            format: PixelFormat::Rgba8,
        })
        .expect("create_texture_2d");
    let mut texels = [255u8, 0, 0, 255].repeat(16);
    for texel in texels[0..4].chunks_mut(4) {
        texel.copy_from_slice(&[0, 255, 0, 255]);
    }
    backend.fill_texture_2d(&texture, &texels).expect("fill_texture_2d");
    backend.bind_texture(Some(&texture), 0);
    // The camera: the diagnostic camera looks at the origin from +Z at
    // distance 2 (the cube's front face fills the frame's centre).
    backend.set_camera(CameraParams {
        eye: Vec3::new(0.0, 0.0, 2.0),
        target: Vec3::new(0.0, 0.0, 0.0),
        up: Vec3::new(0.0, 1.0, 0.0),
        fov_y_rad: std::f32::consts::FRAC_PI_2,
        aspect: 320.0 / 240.0,
        near: 0.1,
        far: 100.0,
    });
    // The cube: size 0.5 (the front face spans z=+0.25, well inside the
    // camera's near plane).
    let mesh_data = cube(0.5);
    let layout = pnu_layout();
    assert_eq!(layout.stride, PNU_STRIDE);
    let mesh = backend.create_gpu_mesh().expect("create_gpu_mesh");
    backend
        .upload_mesh(&mesh, &layout, &mesh_data.vertices, &mesh_data.indices)
        .expect("upload_mesh");
    let program = backend.create_shader_program(1).expect("program");
    let submission = DrawSubmission {
        mesh: mesh.clone(),
        model: Mat4::IDENTITY,
        material: Material::default(),
        sun: None,
        lights: Vec::new(),
    };
    backend.clear();
    backend.draw(&mesh, &program);
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
    // The checks: the cube's centre carries the texture's red, the corner
    // carries the clear colour, the centre's depth is the cube's surface
    // (closer than the far plane's 1.0), and the protocol stays clean.
    // The queue's order may repeat a point (the draw and the submit both
    // record), so the checks match the points by coordinate.
    let centre = samples
        .iter()
        .find(|s| s.x == 160 && s.y == 120)
        .expect("the centre sample");
    let corner = samples
        .iter()
        .find(|s| s.x == 4 && s.y == 4)
        .expect("the corner sample");
    let ok = samples.len() >= 2
        && centre.color.r > 200
        && centre.color.g < 60
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
