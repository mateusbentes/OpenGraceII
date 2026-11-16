use og2_core::render::d3d12::{D3d12Backend, D3d12BackendConfig};
use og2_core::render::{ClearColor, RenderBackend};
fn main() {
    let config = D3d12BackendConfig {
        width: 320,
        height: 240,
        draw_triangle: true,
        triangle_color: ClearColor { r: 200, g: 40, b: 40, a: 255 },
        readback_center: true,
        clear_depth: 1.0,
        clear_stencil: 0,
        clear_buffers: og2_core::render::ClearBuffer::COLOR_DEPTH_STENCIL,
    };
    let mut backend = match D3d12Backend::new(config) {
        Ok(b) => b,
        Err(e) => { println!("SKIP: {}", e); std::process::exit(77); }
    };
    println!("describe: {}", backend.describe());
    backend.set_clear_color(ClearColor { r: 0, g: 0, b: 0, a: 255 });
    backend.clear();
    backend.swap_buffers();
    // The second frame: the readback's copies left the resources in the
    // COPY_SOURCE state, so the return barriers are exercised (the
    // two-frame probe's rule).
    backend.set_clear_color(ClearColor { r: 40, g: 200, b: 40, a: 255 });
    backend.clear();
    backend.swap_buffers();
    println!("has_center: {}", backend.has_center_pixel());
    println!("center: {:?}", backend.center_pixel());
    println!("center_depth: {}", backend.center_depth());
    // The stencil plane's readback is undefined under the Wine run's
    // vkd3d 1.3/lavapipe (the marker sweep showed the copied bytes land
    // as garbage between runs): the smoke reports the value without
    // asserting it, and the real-driver validation will close it.
    println!("center_stencil: {} (undefined under vkd3d)", backend.center_stencil());
    println!("violations: {}", backend.violation_count());
    backend.shutdown();
    let ok = backend.has_center_pixel()
        && backend.center_depth() == 1.0
        && backend.violation_count() == 0;
    if ok {
        println!("OK");
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
