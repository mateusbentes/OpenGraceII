//! The Metal render backend (macOS/iOS): the engine's own loader, own
//! code, behind the platform-neutral render contract.

pub mod mtl_loader;
pub mod shader_library;
pub mod backend;

pub use backend::{MetalBackend, MetalBackendConfig};
