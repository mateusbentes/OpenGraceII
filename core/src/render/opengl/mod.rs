//! The OpenGL render backend.
//!
//! RULE (lesson 3): the engine's own loader, own code, no third-party
//! binding. The loader is its own file so the platform-neutral contract
//! never includes it; the backend consumes the contract through the
//! `RenderBackend` trait in `super::super`.
pub mod gl_loader;
pub mod backend;
pub mod shader_library;
pub use backend::{OpenGlBackend, OpenGlBackendConfig};
