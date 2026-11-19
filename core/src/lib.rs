//! OpenGraceII core.
//!
//! The engine's own foundation: object model, resource manager, threading,
//! math, curves and the profiler. The module layout follows the architecture
//! document (docs/grace-architecture.md section 4); each module is
//! introduced as its milestone lands, not before.

pub mod math;
pub mod render;
pub mod mesh;
pub mod lighting;
pub mod post;
