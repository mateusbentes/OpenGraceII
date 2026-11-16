//! The Direct3D 12 render backend: the render contract's Windows
//! implementation, behind the platform-neutral `RenderBackend` trait.
//!
//! The backend is headless (the same rule the Vulkan, the OpenGL and the
//! Metal backends follow): the frame target is an off-screen colour
//! texture plus a depth/stencil texture, no swap chain and no window. The
//! loader resolves the published entry points at runtime.

pub mod d3d12_loader;
pub mod shader_library;
pub mod backend;

pub use backend::{D3d12Backend, D3d12BackendConfig};
