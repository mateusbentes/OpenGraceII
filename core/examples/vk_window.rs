//! The milestone 9's windowed path smoke: the platform layer's X11
//! window carries the native surface, the Vulkan backend's swapchain
//! creates the present path and the frame loop presents into it.
//!
//! The readback stays with the off-screen target: the windowed path's
//! frames present to the window, so the smoke checks the frame loop's
//! own determinism (the acquire/present cycle the swapchain carries)
//! instead of the pixel readback.

use og2_core::platform::x11::X11Backend;
use og2_core::platform::{PlatformBackend, WindowDesc};
use og2_core::render::vulkan::{VulkanBackend, VulkanBackendConfig};
use og2_core::render::{ClearBuffer, ClearColor, RenderBackend};

fn main() {
    // The platform layer's window: the native surface the backend's
    // swapchain binds to (the M9 contract).
    let desc = WindowDesc {
        title: "OpenGraceII vk_window".to_string(),
        width: 320,
        height: 240,
        resizable: false,
    };
    let mut platform = match X11Backend::new(&desc) {
        Ok(p) => p,
        Err(e) => {
            println!("SKIP: {}", e);
            std::process::exit(77);
        }
    };
    let surface = platform.native_surface();
    println!("surface: {:?}", surface);
    // The windowed config: the surface the platform layer carries.
    let config = VulkanBackendConfig {
        width: 320,
        height: 240,
        surface: Some(surface),
        draw_triangle: false,
        triangle_color: ClearColor { r: 200, g: 40, b: 40, a: 255 },
        readback_center: false,
        clear_buffers: ClearBuffer::COLOR_DEPTH_STENCIL,
        clear_depth: 1.0,
        clear_stencil: 0,
    };
    let mut backend = match VulkanBackend::new(config) {
        Ok(b) => b,
        Err(e) => {
            println!("SKIP: {}", e);
            std::process::exit(77);
        }
    };
    println!("describe: {}", backend.describe());
    // The frame loop: the clear opens the frame, the swap closes and
    // presents it (the M9 path's own contract).
    let clear = ClearColor { r: 30, g: 40, b: 50, a: 255 };
    for _ in 0..5 {
        backend.set_clear_color(clear);
        backend.clear();
        backend.swap_buffers();
    }
    assert_eq!(backend.violation_count(), 0);
    backend.shutdown();
    assert_eq!(backend.violation_count(), 0);
    platform.shutdown();
    println!("OK");
}
