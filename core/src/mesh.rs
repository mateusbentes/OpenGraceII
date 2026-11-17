//! The engine's own CPU-side mesh format.
//!
//! Milestone 4: a mesh is plain data — a declared vertex layout plus the
//! vertex bytes and the indices — so the render backends can consume it
//! through the same upload path (no per-mesh special cases). The builders
//! produce the diagnostic geometry the validation suite draws.

use crate::math::Aabb;
use crate::render::{ComponentType, VertexAttribute, VertexLayout, VertexSemantic};

/// The engine's standard vertex: position, normal, one UV set.
/// 8 float32s, 32 bytes.
pub const PNU_STRIDE: u32 = 32;

/// The vertex layout matching the `PNU` builder vertex: position (offset
/// 0), normal (offset 12), texcoord0 (offset 24).
pub fn pnu_layout() -> VertexLayout {
    VertexLayout {
        stride: PNU_STRIDE,
        attributes: vec![
            VertexAttribute {
                semantic: VertexSemantic::Position,
                component_type: ComponentType::Float32,
                components: 3,
                offset: 0,
                normalized: false,
            },
            VertexAttribute {
                semantic: VertexSemantic::Normal,
                component_type: ComponentType::Float32,
                components: 3,
                offset: 12,
                normalized: false,
            },
            VertexAttribute {
                semantic: VertexSemantic::TexCoord0,
                component_type: ComponentType::Float32,
                components: 2,
                offset: 24,
                normalized: false,
            },
        ],
    }
}

/// A CPU mesh: the vertex bytes, the indices and the layout that
/// describes the vertex bytes.
#[derive(Clone, Debug)]
pub struct MeshData {
    pub layout: VertexLayout,
    pub vertices: Vec<u8>,
    pub indices: Vec<u32>,
}

impl MeshData {
    /// The vertex count (the byte length must be a stride multiple; the
    /// builders guarantee it, the upload validates it).
    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32 / self.layout.stride
    }

    /// The AABB of the position attribute (the standard layout's first
    /// attribute reads three float32s at its offset).
    pub fn aabb(&self) -> Aabb {
        let position = self
            .layout
            .attributes
            .iter()
            .find(|a| a.semantic == VertexSemantic::Position)
            .expect("the layout carries a position attribute");
        let stride = self.layout.stride as usize;
        let offset = position.offset as usize;
        let mut points = Vec::with_capacity(self.vertex_count() as usize);
        for v in 0..self.vertex_count() as usize {
            let base = v * stride + offset;
            let x = f32::from_le_bytes(self.vertices[base..base + 4].try_into().unwrap());
            let y = f32::from_le_bytes(
                self.vertices[base + 4..base + 8].try_into().unwrap(),
            );
            let z = f32::from_le_bytes(
                self.vertices[base + 8..base + 12].try_into().unwrap(),
            );
            points.push(crate::math::Vec3::new(x, y, z));
        }
        Aabb::from_points(&points)
    }
}

/// Writes one f32 into `out` (little-endian, the platform's uniform
/// convention).
fn push_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Writes one PNU vertex.
fn push_pnu_vertex(out: &mut Vec<u8>, p: [f32; 3], n: [f32; 3], uv: [f32; 2]) {
    for v in p {
        push_f32(out, v);
    }
    for v in n {
        push_f32(out, v);
    }
    for v in uv {
        push_f32(out, v);
    }
}

/// An axis-aligned unit cube centred at the origin (the extents are
/// `size`), 24 vertices (4 per face so the normals and the UVs stay
/// per-face), 36 indices. The winding is counter-clockwise viewed from
/// outside (the front-face convention the backends share).
pub fn cube(size: f32) -> MeshData {
    let h = size * 0.5;
    // Each face: the four corners, the normal, in CCW order from outside.
    let faces: [[([f32; 3], [f32; 3], [f32; 2]); 4]; 6] = [
        // +Z (front)
        [
            ([-h, -h, h], [0.0, 0.0, 1.0], [0.0, 0.0]),
            ([h, -h, h], [0.0, 0.0, 1.0], [1.0, 0.0]),
            ([h, h, h], [0.0, 0.0, 1.0], [1.0, 1.0]),
            ([-h, h, h], [0.0, 0.0, 1.0], [0.0, 1.0]),
        ],
        // -Z (back)
        [
            ([h, -h, -h], [0.0, 0.0, -1.0], [0.0, 0.0]),
            ([-h, -h, -h], [0.0, 0.0, -1.0], [1.0, 0.0]),
            ([-h, h, -h], [0.0, 0.0, -1.0], [1.0, 1.0]),
            ([h, h, -h], [0.0, 0.0, -1.0], [0.0, 1.0]),
        ],
        // +X (right)
        [
            ([h, -h, h], [1.0, 0.0, 0.0], [0.0, 0.0]),
            ([h, -h, -h], [1.0, 0.0, 0.0], [1.0, 0.0]),
            ([h, h, -h], [1.0, 0.0, 0.0], [1.0, 1.0]),
            ([h, h, h], [1.0, 0.0, 0.0], [0.0, 1.0]),
        ],
        // -X (left)
        [
            ([-h, -h, -h], [-1.0, 0.0, 0.0], [0.0, 0.0]),
            ([-h, -h, h], [-1.0, 0.0, 0.0], [1.0, 0.0]),
            ([-h, h, h], [-1.0, 0.0, 0.0], [1.0, 1.0]),
            ([-h, h, -h], [-1.0, 0.0, 0.0], [0.0, 1.0]),
        ],
        // +Y (top)
        [
            ([-h, h, h], [0.0, 1.0, 0.0], [0.0, 0.0]),
            ([h, h, h], [0.0, 1.0, 0.0], [1.0, 0.0]),
            ([h, h, -h], [0.0, 1.0, 0.0], [1.0, 1.0]),
            ([-h, h, -h], [0.0, 1.0, 0.0], [0.0, 1.0]),
        ],
        // -Y (bottom)
        [
            ([-h, -h, -h], [0.0, -1.0, 0.0], [0.0, 0.0]),
            ([h, -h, -h], [0.0, -1.0, 0.0], [1.0, 0.0]),
            ([h, -h, h], [0.0, -1.0, 0.0], [1.0, 1.0]),
            ([-h, -h, h], [0.0, -1.0, 0.0], [0.0, 1.0]),
        ],
    ];
    let mut vertices = Vec::with_capacity(24 * PNU_STRIDE as usize);
    let mut indices = Vec::with_capacity(36);
    for face in &faces {
        let base = (vertices.len() as u32) / PNU_STRIDE;
        for (p, n, uv) in face {
            push_pnu_vertex(&mut vertices, *p, *n, *uv);
        }
        for i in [0u32, 1, 2, 0, 2, 3] {
            indices.push(base + i);
        }
    }
    MeshData {
        layout: pnu_layout(),
        vertices,
        indices,
    }
}

/// A unit quad in the XY plane facing +Z, 4 vertices, 6 indices (the
/// simplest textured surface; the smoke tests draw it for the texture
/// path's checks).
pub fn quad() -> MeshData {
    let mut vertices = Vec::with_capacity(4 * PNU_STRIDE as usize);
    push_pnu_vertex(&mut vertices, [-0.5, -0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]);
    push_pnu_vertex(&mut vertices, [0.5, -0.5, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]);
    push_pnu_vertex(&mut vertices, [0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0]);
    push_pnu_vertex(&mut vertices, [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]);
    MeshData {
        layout: pnu_layout(),
        vertices,
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    #[test]
    fn pnu_layout_offsets_and_stride() {
        let layout = pnu_layout();
        assert_eq!(layout.stride, 32);
        assert_eq!(layout.attributes.len(), 3);
        assert_eq!(layout.attributes[0].semantic, VertexSemantic::Position);
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[1].semantic, VertexSemantic::Normal);
        assert_eq!(layout.attributes[1].offset, 12);
        assert_eq!(layout.attributes[2].semantic, VertexSemantic::TexCoord0);
        assert_eq!(layout.attributes[2].offset, 24);
        // The attributes fit inside the stride: the last attribute's end
        // is the stride itself.
        let last = layout.attributes.last().unwrap();
        assert!(last.offset + last.components * 4 <= layout.stride);
    }

    #[test]
    fn cube_shape_and_indexing() {
        let cube = cube(1.0);
        assert_eq!(cube.vertex_count(), 24);
        assert_eq!(cube.indices.len(), 36);
        assert_eq!(cube.layout.stride, PNU_STRIDE);
        // The indices stay inside the vertex range and the six faces use
        // the same two triangles' pattern.
        for index in &cube.indices {
            assert!(*index < 24);
        }
        for face in 0..6 {
            let base = face * 6;
            let pattern = [0u32, 1, 2, 0, 2, 3];
            for (k, expected) in pattern.iter().enumerate() {
                assert_eq!(cube.indices[base + k], face as u32 * 4 + expected);
            }
        }
    }

    #[test]
    fn cube_aabb_matches_the_size() {
        let cube = cube(2.0);
        let box_ = cube.aabb();
        assert!((box_.min.x + 1.0).abs() < 1e-6 && (box_.min.y + 1.0).abs() < 1e-6);
        assert!((box_.max.x - 1.0).abs() < 1e-6 && (box_.max.z - 1.0).abs() < 1e-6);
        assert!((box_.centre().x.abs()) < 1e-6);
        assert!((box_.extent().x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cube_normals_are_unit_and_outward() {
        let cube = cube(1.0);
        let stride = cube.layout.stride as usize;
        let offset = 12usize; // the normal attribute's offset
        let mut seen_normals = Vec::new();
        for v in 0..cube.vertex_count() as usize {
            let base = v * stride + offset;
            let nx = f32::from_le_bytes(cube.vertices[base..base + 4].try_into().unwrap());
            let ny =
                f32::from_le_bytes(cube.vertices[base + 4..base + 8].try_into().unwrap());
            let nz = f32::from_le_bytes(
                cube.vertices[base + 8..base + 12].try_into().unwrap(),
            );
            let n = Vec3::new(nx, ny, nz);
            assert!((n.length() - 1.0).abs() < 1e-6, "the normal is unit");
            seen_normals.push(n);
        }
        // Each of the six axis directions appears as a normal.
        for axis in [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ] {
            assert!(
                seen_normals.iter().any(|n| *n == axis),
                "the face normal {:?} appears",
                axis
            );
        }
    }

    #[test]
    fn quad_shape() {
        let quad = quad();
        assert_eq!(quad.vertex_count(), 4);
        assert_eq!(quad.indices, vec![0, 1, 2, 0, 2, 3]);
        let box_ = quad.aabb();
        assert!((box_.min.x + 0.5).abs() < 1e-6);
        assert!((box_.max.x - 0.5).abs() < 1e-6);
    }
}
