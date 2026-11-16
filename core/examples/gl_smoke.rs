use og2_core::render::opengl::{OpenGlBackend, OpenGlBackendConfig};
use og2_core::render::{ClearColor, RenderBackend};

fn main() {
    let config = OpenGlBackendConfig {
        width: 320,
        height: 240,
        draw_triangle: true,
        triangle_color: ClearColor { r: 200, g: 40, b: 40, a: 255 },
        readback_center: true,
        clear_buffers: og2_core::render::ClearBuffer::COLOR_DEPTH_STENCIL,
        clear_depth: 1.0,
        clear_stencil: 0,
    };
    let mut backend = match OpenGlBackend::new(config) {
        Ok(b) => b,
        Err(e) => { println!("SKIP: {}", e); std::process::exit(77); }
    };
    println!("describe: {}", backend.describe());
    backend.set_clear_color(ClearColor { r: 0, g: 0, b: 0, a: 255 });
    backend.clear();
    backend.swap_buffers();
    println!("has_center: {}", backend.has_center_pixel());
    println!("center: {:?}", backend.center_pixel());
    println!("center_depth: {}", backend.center_depth());
    println!("center_stencil: {}", backend.center_stencil());
    println!("violations: {}", backend.violation_count());
    backend.shutdown();
    let ok = backend.has_center_pixel()
        && backend.center_depth() == 1.0
        && backend.center_stencil() == 0
        && backend.violation_count() == 0;
    if ok {
        println!("OK");
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
