//! The render contract's tests: the null backend's deterministic
//! lifecycle and the Vulkan backend's hardware-when-available smoke
//! (the same shape the OpenGrace test_vulkan_render.cpp follows).

#[cfg(feature = "vulkan")]
use og2_core::render::vulkan::VulkanBackend;
#[cfg(feature = "vulkan")]
use og2_core::render::VulkanBackendConfig;
use og2_core::render::{
    ClearBuffer, ClearColor, NullRenderBackend, RenderBackend,
};
#[cfg(feature = "metal")]
use og2_core::render::metal::{MetalBackend, MetalBackendConfig};
#[cfg(feature = "d3d12")]
use og2_core::render::d3d12::{D3d12Backend, D3d12BackendConfig};

const CLEAR_COLOR: ClearColor = ClearColor {
    r: 30,
    g: 40,
    b: 50,
    a: 255,
};
// The backends' smokes carry the triangle colour; the featureless
// build compiles none of them, so the constant is gated the same way.
#[cfg(any(feature = "vulkan", feature = "metal", feature = "d3d12"))]
const TRIANGLE_COLOR: ClearColor = ClearColor {
    r: 200,
    g: 40,
    b: 40,
    a: 255,
};

#[test]
fn null_backend_lifecycle_and_readback() {
    let mut backend = NullRenderBackend::new(320, 240)
        .expect("NullRenderBackend::new");
    backend.set_clear_color(CLEAR_COLOR);
    assert_eq!(backend.clear_color(), CLEAR_COLOR);
    backend.clear();
    // The null backend's readback models the clear colour: every pixel
    // reports exactly the current clear colour (depth 1.0, stencil 0).
    backend.swap_buffers();
    assert_eq!(backend.has_center_pixel(), true);
    assert_eq!(backend.center_pixel(), CLEAR_COLOR);
    assert_eq!(backend.center_depth(), 1.0);
    assert_eq!(backend.center_stencil(), 0);
    assert_eq!(backend.violation_count(), 0);
    backend.shutdown();
    assert_eq!(backend.violation_count(), 0);
}

#[test]
fn null_backend_queues_and_reads_back() {
    let mut backend = NullRenderBackend::new(320, 240)
        .expect("NullRenderBackend::new");
    backend.set_clear_color(CLEAR_COLOR);
    backend.clear();
    backend.queue_readback(10, 20);
    backend.queue_readback(30, 40);
    backend.readback();
    let samples = backend.last_readback();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].x, 10);
    assert_eq!(samples[0].y, 20);
    assert_eq!(samples[0].color, CLEAR_COLOR);
    assert_eq!(samples[0].depth, 1.0);
    assert_eq!(samples[0].stencil, 0);
    backend.swap_buffers();
    assert_eq!(backend.violation_count(), 0);
    backend.shutdown();
}

#[test]
fn null_backend_rejects_after_shutdown() {
    let mut backend = NullRenderBackend::new(320, 240)
        .expect("NullRenderBackend::new");
    backend.shutdown();
    // Calls after shutdown are protocol violations, and the clear
    // colour is no longer settable.
    backend.set_clear_color(CLEAR_COLOR);
    backend.clear();
    backend.swap_buffers();
    // swap_buffers counts one violation and its readback() counts one
    // more, so the three calls after shutdown produce four.
    assert_eq!(backend.violation_count(), 4);
    backend.shutdown();
    assert_eq!(backend.violation_count(), 5);
}

#[test]
fn null_backend_clear_buffers() {
    let mut backend = NullRenderBackend::new(320, 240)
        .expect("NullRenderBackend::new");
    backend.set_clear_color(CLEAR_COLOR);
    backend.clear();
    backend.clear();
    backend.swap_buffers();
    assert_eq!(backend.clear_color(), CLEAR_COLOR);
    assert_eq!(backend.center_pixel(), CLEAR_COLOR);
    assert_eq!(backend.violation_count(), 0);
    backend.shutdown();
}

#[test]
fn null_backend_describe() {
    let backend =
        NullRenderBackend::new(320, 240).expect("NullRenderBackend::new");
    assert_eq!(backend.describe(), "null (headless)");
    assert_eq!(backend.width(), 320);
    assert_eq!(backend.height(), 240);
    assert_eq!(backend.violation_count(), 0);
}

#[test]
fn null_backend_default_config_fields() {
    assert_eq!(ClearBuffer::COLOR_DEPTH_STENCIL.color, true);
    assert_eq!(ClearBuffer::COLOR_DEPTH_STENCIL.depth, true);
    assert_eq!(ClearBuffer::COLOR_DEPTH_STENCIL.stencil, true);
    #[cfg(feature = "vulkan")]
    {
        let config = VulkanBackendConfig::default();
        assert_eq!(config.clear_buffers, ClearBuffer::COLOR_DEPTH_STENCIL);
        assert_eq!(config.clear_depth, 1.0);
        assert_eq!(config.clear_stencil, 0);
    }
}

/// The Vulkan backend's smoke: the backend initialises on the platform's
/// loader (the sandbox's lavapipe counts), renders the diagnostic
/// triangle and reads the centre pixel back. The test skips when the
/// loader cannot open the platform's library (the same skip the OpenGrace
/// test_vulkan_render.cpp follows).
#[test]
#[cfg(feature = "vulkan")]
fn vulkan_backend_smoke() {
    let config = VulkanBackendConfig {
        width: 320,
        height: 240,
        draw_triangle: true,
        triangle_color: TRIANGLE_COLOR,
        readback_center: true,
        clear_buffers: ClearBuffer::COLOR_DEPTH_STENCIL,
        clear_depth: 0.25,
        clear_stencil: 7,
    };
    let mut backend = match VulkanBackend::new(config) {
        Ok(b) => b,
        Err(e) => {
            // The platform has no Vulkan loader (the Mac and the iOS
            // builds compile the backend but never run it): the skip is
            // a return, never a process exit (an exit would kill the
            // whole test binary, the defect the macOS suite run caught).
            eprintln!("skip: {}", e);
            return;
        }
    };
    backend.set_clear_color(CLEAR_COLOR);
    backend.clear();
    backend.swap_buffers();
    assert_eq!(backend.has_center_pixel(), true);
    let centre = backend.center_pixel();
    eprintln!("vulkan centre: {:?}", centre);
    // The clear depth/stencil values reach the framebuffer (the same
    // check the C++ test_vulkan_render.cpp runs on the depth word).
    assert!((backend.center_depth() - 0.25).abs() < 0.001);
    assert_eq!(backend.center_stencil(), 7);
    assert_eq!(backend.violation_count(), 0);
    backend.shutdown();
    assert_eq!(backend.violation_count(), 0);
}

/// The Metal backend's smoke: the backend initialises on the platform's
/// loader (a Mac counts; the loader reports cleanly where the framework
/// is absent, so the test skips on Linux). The same skip the Vulkan
/// smoke follows.
#[test]
#[cfg(feature = "metal")]
fn metal_backend_smoke() {
    let config = MetalBackendConfig {
        width: 320,
        height: 240,
        draw_triangle: true,
        triangle_color: TRIANGLE_COLOR,
        readback_center: true,
        clear_buffers: ClearBuffer::COLOR_DEPTH_STENCIL,
        clear_depth: 0.25,
        clear_stencil: 7,
    };
    let mut backend = match MetalBackend::new(config) {
        Ok(b) => b,
        Err(e) => {
            // The platform has no Metal framework: the loader reports it
            // cleanly and the test skips (the same skip the Vulkan smoke
            // follows; the hardware run needs a Mac).
            eprintln!("skip: {}", e);
            return;
        }
    };
    backend.set_clear_color(CLEAR_COLOR);
    backend.clear();
    backend.swap_buffers();
    assert_eq!(backend.has_center_pixel(), true);
    assert!((backend.center_depth() - 0.25).abs() < 0.001);
    assert_eq!(backend.center_stencil(), 7);
    assert_eq!(backend.violation_count(), 0);
    backend.shutdown();
    assert_eq!(backend.violation_count(), 0);
}

/// The Direct3D 12 backend's smoke: the backend initialises on the
/// platform's d3d12.dll (the sandbox validates the path under Wine's
/// vkd3d; a Windows hardware run stays with the user). The loader
/// reports cleanly where the DLL is absent, so the test skips — the
/// same skip the Metal smoke follows.
#[test]
#[cfg(feature = "d3d12")]
fn d3d12_backend_smoke() {
    let config = D3d12BackendConfig {
        width: 320,
        height: 240,
        draw_triangle: true,
        triangle_color: TRIANGLE_COLOR,
        readback_center: true,
        clear_buffers: ClearBuffer::COLOR_DEPTH_STENCIL,
        clear_depth: 0.25,
        clear_stencil: 7,
    };
    let mut backend = match D3d12Backend::new(config) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: {}", e);
            return;
        }
    };
    backend.set_clear_color(CLEAR_COLOR);
    backend.clear();
    backend.swap_buffers();
    assert_eq!(backend.has_center_pixel(), true);
    assert!((backend.center_depth() - 0.25).abs() < 0.001);
    assert_eq!(backend.center_stencil(), 7);
    assert_eq!(backend.violation_count(), 0);
    backend.shutdown();
    assert_eq!(backend.violation_count(), 0);
}
