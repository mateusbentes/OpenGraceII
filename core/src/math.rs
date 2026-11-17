//! The engine's own math types.
//!
//! Introduced in milestone 1 as the smallest stable surface the later
//! milestones (render foundation, physics, lighting) can build on. The
//! implementations are intentionally straightforward: correctness first,
//! profiled optimisation later.

use std::ops::{Add, Mul, Neg, Sub};

/// A two-component vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

/// A three-component vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A four-component vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };
    pub const ONE: Vec2 = Vec2 { x: 1.0, y: 1.0 };

    pub fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    pub fn dot(self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalised(self) -> Vec2 {
        let len = self.length();
        if len > 0.0 {
            Vec2::new(self.x / len, self.y / len)
        } else {
            Vec2::ZERO
        }
    }
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Vec3 = Vec3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    pub const X: Vec3 = Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const Y: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const Z: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalised(self) -> Vec3 {
        let len = self.length();
        if len > 0.0 {
            Vec3::new(self.x / len, self.y / len, self.z / len)
        } else {
            Vec3::ZERO
        }
    }
}

impl Vec4 {
    pub const ZERO: Vec4 = Vec4 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };
    pub const ONE: Vec4 = Vec4 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
        w: 1.0,
    };

    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
        Vec4 { x, y, z, w }
    }

    pub fn dot(self, other: Vec4) -> f32 {
        self.x * other.x
            + self.y * other.y
            + self.z * other.z
            + self.w * other.w
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x + other.x, self.y + other.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x - other.x, self.y - other.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, s: f32) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }
}

impl Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2::new(-self.x, -self.y)
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Add for Vec4 {
    type Output = Vec4;
    fn add(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )
    }
}

impl Sub for Vec4 {
    type Output = Vec4;
    fn sub(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        )
    }
}

impl Mul<f32> for Vec4 {
    type Output = Vec4;
    fn mul(self, s: f32) -> Vec4 {
        Vec4::new(self.x * s, self.y * s, self.z * s, self.w * s)
    }
}

impl Neg for Vec4 {
    type Output = Vec4;
    fn neg(self) -> Vec4 {
        Vec4::new(-self.x, -self.y, -self.z, -self.w)
    }
}

/// A 4x4 matrix in the engine's memory layout: `m[c][r]` is the element
/// at column c, row r, so the flat memory can be handed to the graphics
/// APIs' uniform blocks as-is (the same `cols[c][r]` the engine's
/// column-major convention follows).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// Column-major elements: `m[c][r]` is the element at column c,
    /// row r.
    pub m: [[f32; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4 {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// Matrix product: `(a * b)(c, r) = sum_k a(k, r) * b(c, k)`.
    pub fn multiply(self, other: Mat4) -> Mat4 {
        let mut out = Mat4 { m: [[0.0; 4]; 4] };
        for c in 0..4 {
            for r in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.m[k][r] * other.m[c][k];
                }
                out.m[c][r] = sum;
            }
        }
        out
    }

    /// Matrix-vector product treating v as a column vector.
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        let m = &self.m;
        Vec3::new(
            m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0],
            m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1],
            m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z + m[3][2],
        )
    }

    /// Right-handed view matrix ("look at"): the camera sits at eye and
    /// looks at target, with up as the approximate up direction. Maps
    /// world space to view space where the camera looks down -Z. A point
    /// at target ends up at view coordinates (0, 0, -distance, 1).
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let forward = (target - eye).normalised();
        let right_raw = forward.cross(up);
        let right = right_raw.normalised();
        let real_up = right.cross(forward);
        let mut m = Mat4::IDENTITY;
        m.m[0][0] = right.x;
        m.m[1][0] = right.y;
        m.m[2][0] = right.z;
        m.m[0][1] = real_up.x;
        m.m[1][1] = real_up.y;
        m.m[2][1] = real_up.z;
        m.m[0][2] = -forward.x;
        m.m[1][2] = -forward.y;
        m.m[2][2] = -forward.z;
        m.m[3][0] = -right.dot(eye);
        m.m[3][1] = -real_up.dot(eye);
        m.m[3][2] = forward.dot(eye);
        m
    }

    /// Perspective projection matrix in the OpenGL clip convention: after
    /// the perspective divide, x/y are in [-1, 1] and z (depth) is in
    /// [-1, 1], with -1 at the near plane and +1 at the far plane.
    /// `fov_y` is the vertical field of view in radians; `aspect` is
    /// width / height; `near` and `far` are positive distances from the
    /// eye with `near < far`.
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fov_y * 0.5).tan();
        let mut m = Mat4 { m: [[0.0; 4]; 4] };
        m.m[0][0] = f / aspect;
        m.m[1][1] = f;
        m.m[2][2] = (far + near) / (near - far);
        m.m[3][2] = (2.0 * far * near) / (near - far);
        m.m[2][3] = -1.0;
        m
    }
}

/// A quaternion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Quat = Quat {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Quat {
        Quat { x, y, z, w }
    }

    pub fn normalised(self) -> Quat {
        let len = (self.x * self.x
            + self.y * self.y
            + self.z * self.z
            + self.w * self.w)
            .sqrt();
        if len > 0.0 {
            Quat::new(self.x / len, self.y / len, self.z / len, self.w / len)
        } else {
            Quat::IDENTITY
        }
    }

    /// The unit quaternion rotating by `angle` radians around `axis`
    /// (the axis need not be normalised; a zero axis yields identity).
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Quat {
        let len = axis.length();
        if len == 0.0 {
            return Quat::IDENTITY;
        }
        let s = (angle * 0.5).sin() / len;
        Quat::new(
            axis.x * s,
            axis.y * s,
            axis.z * s,
            (angle * 0.5).cos(),
        )
    }

    /// Hamilton product: the rotation `self` followed by `other`.
    /// `a.multiply(b)` applies `a` first, then `b`.
    pub fn multiply(self, other: Quat) -> Quat {
        Quat::new(
            self.w * other.x + self.x * other.w + self.y * other.z
                - self.z * other.y,
            self.w * other.y + self.y * other.w + self.z * other.x
                - self.x * other.z,
            self.w * other.z + self.z * other.w + self.x * other.y
                - self.y * other.x,
            self.w * other.w - self.x * other.x - self.y * other.y
                - self.z * other.z,
        )
    }

    /// Rotates `v` by this quaternion (`q * v * q*`).
    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let uv = u.cross(v);
        let uuv = u.cross(uv);
        v + uv * (2.0 * self.w) + uuv * 2.0
    }

    /// The 3x3 rotation as a 4x4 matrix (the rotation part only; use
    /// `Transform::to_mat4` for the full position/rotation/scale chain).
    pub fn to_mat4(self) -> Mat4 {
        let q = self.normalised();
        let x2 = q.x + q.x;
        let y2 = q.y + q.y;
        let z2 = q.z + q.z;
        let xx = q.x * x2;
        let xy = q.x * y2;
        let xz = q.x * z2;
        let yy = q.y * y2;
        let yz = q.y * z2;
        let zz = q.z * z2;
        let wx = q.w * x2;
        let wy = q.w * y2;
        let wz = q.w * z2;
        let mut m = Mat4::IDENTITY;
        m.m[0][0] = 1.0 - (yy + zz);
        m.m[1][0] = xy - wz;
        m.m[2][0] = xz + wy;
        m.m[0][1] = xy + wz;
        m.m[1][1] = 1.0 - (xx + zz);
        m.m[2][1] = yz - wx;
        m.m[0][2] = xz - wy;
        m.m[1][2] = yz + wx;
        m.m[2][2] = 1.0 - (xx + yy);
        m
    }

    /// Spherical interpolation between two unit quaternions. The shortest
    /// path is taken automatically (a negative dot negates the target).
    /// Coplanar (nearly antiparallel) inputs fall back to normalised lerp.
    pub fn slerp(self, target: Quat, t: f32) -> Quat {
        let mut end = target;
        let mut cos_half = self.x * end.x
            + self.y * end.y
            + self.z * end.z
            + self.w * end.w;
        if cos_half < 0.0 {
            end = Quat::new(-end.x, -end.y, -end.z, -end.w);
            cos_half = -cos_half;
        }
        let (a, b) = if cos_half > 0.9995 {
            (1.0 - t, t)
        } else {
            let half = cos_half.acos();
            let sin_half = half.sin();
            (((half * (1.0 - t)).sin()) / sin_half, ((half * t).sin()) / sin_half)
        };
        Quat::new(
            self.x * a + end.x * b,
            self.y * a + end.y * b,
            self.z * a + end.z * b,
            self.w * a + end.w * b,
        )
        .normalised()
    }

    /// The rotation quaternion of a T*R*S matrix (the matrix must be
    /// orthonormal after the scales are divided out; the standard
    /// Shepperd's method picks the largest component for stability).
    pub fn from_mat4(m: Mat4) -> Quat {
        let trace = m.m[0][0] + m.m[1][1] + m.m[2][2];
        let (x, y, z, w) = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            (
                (m.m[2][1] - m.m[1][2]) / s,
                (m.m[0][2] - m.m[2][0]) / s,
                (m.m[1][0] - m.m[0][1]) / s,
                s * 0.25,
            )
        } else if m.m[0][0] > m.m[1][1] && m.m[0][0] > m.m[2][2] {
            let s = (1.0 + m.m[0][0] - m.m[1][1] - m.m[2][2]).sqrt() * 2.0;
            (
                s * 0.25,
                (m.m[0][1] + m.m[1][0]) / s,
                (m.m[0][2] + m.m[2][0]) / s,
                (m.m[2][1] - m.m[1][2]) / s,
            )
        } else if m.m[1][1] > m.m[2][2] {
            let s = (1.0 + m.m[1][1] - m.m[0][0] - m.m[2][2]).sqrt() * 2.0;
            (
                (m.m[0][1] + m.m[1][0]) / s,
                s * 0.25,
                (m.m[1][2] + m.m[2][1]) / s,
                (m.m[0][2] - m.m[2][0]) / s,
            )
        } else {
            let s = (1.0 + m.m[2][2] - m.m[0][0] - m.m[1][1]).sqrt() * 2.0;
            (
                (m.m[0][2] + m.m[2][0]) / s,
                (m.m[1][2] + m.m[2][1]) / s,
                s * 0.25,
                (m.m[1][0] - m.m[0][1]) / s,
            )
        };
        Quat::new(x, y, z, w).normalised()
    }
}

/// A plane in the `normal . p + distance = 0` form; the negative
/// half-space is where `normal . p + distance < 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub normal: Vec3,
    pub distance: f32,
}

impl Plane {
    /// The plane through `p` with the given normal (normalised on use).
    pub fn from_point_normal(p: Vec3, normal: Vec3) -> Plane {
        let n = normal.normalised();
        Plane { normal: n, distance: -n.dot(p) }
    }
}

/// The six planes of a view-projection matrix, extracted in the OpenGL
/// clip convention (the convention Mat4::perspective documents). The
/// planes' normals face outward from the frustum's volume (the
/// Gribb-Hartmann extraction; each plane is a row combination of the
/// matrix, normalised).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    pub fn from_view_projection(vp: Mat4) -> Frustum {
        let m = vp.m;
        // row(r) = (m[0][r], m[1][r], m[2][r], m[3][r]) in the
        // column-major layout.
        let row = |r: usize| [m[0][r], m[1][r], m[2][r], m[3][r]];
        let r3 = row(3);
        let combinations: [(usize, f32); 6] = [
            (0, 1.0),  // left:   row3 + row0
            (0, -1.0), // right:  row3 - row0
            (1, 1.0),  // bottom: row3 + row1
            (1, -1.0), // top:    row3 - row1
            (2, 1.0),  // near:   row3 + row2
            (2, -1.0), // far:    row3 - row2
        ];
        let mut planes = [Plane { normal: Vec3::ZERO, distance: 0.0 }; 6];
        for (i, (r, sign)) in combinations.into_iter().enumerate() {
            let rr = row(r);
            let n = Vec3::new(
                r3[0] + sign * rr[0],
                r3[1] + sign * rr[1],
                r3[2] + sign * rr[2],
            );
            let d = r3[3] + sign * rr[3];
            let len = n.length();
            planes[i] = if len > 0.0 {
                Plane { normal: n * (1.0 / len), distance: d / len }
            } else {
                Plane { normal: n, distance: d }
            };
        }
        Frustum { planes }
    }

    /// Whether a sphere is entirely outside any plane (conservative; a
    /// sphere touching a plane counts as inside).
    pub fn sphere_outside(self, centre: Vec3, radius: f32) -> bool {
        self.planes
            .iter()
            .any(|p| p.normal.dot(centre) + p.distance < -radius)
    }
}

/// An axis-aligned bounding box. The empty box (min = +inf, max = -inf)
/// is the neutral element of `unioned`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: Vec3 {
            x: f32::INFINITY,
            y: f32::INFINITY,
            z: f32::INFINITY,
        },
        max: Vec3 {
            x: f32::NEG_INFINITY,
            y: f32::NEG_INFINITY,
            z: f32::NEG_INFINITY,
        },
    };

    pub fn new(min: Vec3, max: Vec3) -> Aabb {
        Aabb { min, max }
    }

    /// The box containing every point in `points` (an empty slice yields
    /// the empty box).
    pub fn from_points(points: &[Vec3]) -> Aabb {
        let mut box_ = Aabb::EMPTY;
        for p in points {
            box_.min = Vec3::new(
                box_.min.x.min(p.x),
                box_.min.y.min(p.y),
                box_.min.z.min(p.z),
            );
            box_.max = Vec3::new(
                box_.max.x.max(p.x),
                box_.max.y.max(p.y),
                box_.max.z.max(p.z),
            );
        }
        box_
    }

    pub fn centre(self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    pub fn extent(self) -> Vec3 {
        Vec3::new(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5,
        )
    }

    /// The box containing both inputs.
    pub fn unioned(self, other: Aabb) -> Aabb {
        Aabb::new(
            Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        )
    }

    /// Grows the box to contain `point`.
    pub fn expanded_to(self, point: Vec3) -> Aabb {
        self.unioned(Aabb::new(point, point))
    }

    /// Whether the box is entirely outside any of the frustum's planes
    /// (the conservative culling test: the box's positive vertex — the
    /// corner furthest along each plane's normal — is the only corner
    /// that can be inside that plane's half-space).
    pub fn outside_frustum(self, frustum: &Frustum) -> bool {
        frustum.planes.iter().any(|plane| {
            let px = if plane.normal.x >= 0.0 { self.max.x } else { self.min.x };
            let py = if plane.normal.y >= 0.0 { self.max.y } else { self.min.y };
            let pz = if plane.normal.z >= 0.0 { self.max.z } else { self.min.z };
            plane.normal.dot(Vec3::new(px, py, pz)) + plane.distance < 0.0
        })
    }
}

/// A position/rotation/scale node: the scene graph's transform. The
/// composition order matches the scene conventions: `compose(parent,
/// child)` places the child inside the parent's space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        position: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
        rotation: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
        scale: Vec3 { x: 1.0, y: 1.0, z: 1.0 },
    };

    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Transform {
        Transform { position, rotation, scale }
    }

    /// The local transform as a matrix: T * R * S.
    pub fn to_mat4(self) -> Mat4 {
        let r = self.rotation.to_mat4();
        let mut m = r;
        m.m[0][0] *= self.scale.x;
        m.m[0][1] *= self.scale.y;
        m.m[0][2] *= self.scale.z;
        m.m[1][0] *= self.scale.x;
        m.m[1][1] *= self.scale.y;
        m.m[1][2] *= self.scale.z;
        m.m[2][0] *= self.scale.x;
        m.m[2][1] *= self.scale.y;
        m.m[2][2] *= self.scale.z;
        m.m[3][0] = self.position.x;
        m.m[3][1] = self.position.y;
        m.m[3][2] = self.position.z;
        m
    }

    /// The world transform of `child` placed inside `self` (the parent):
    /// the parent's matrix times the child's matrix.
    pub fn compose(parent: Transform, child: Transform) -> Transform {
        let m = parent.to_mat4().multiply(child.to_mat4());
        Transform::from_mat4(m)
    }

    /// Decomposes an affine T*R*S matrix back into the transform.
    pub fn from_mat4(m: Mat4) -> Transform {
        let position = Vec3::new(m.m[3][0], m.m[3][1], m.m[3][2]);
        let sx = Vec3::new(m.m[0][0], m.m[0][1], m.m[0][2]).length();
        let sy = Vec3::new(m.m[1][0], m.m[1][1], m.m[1][2]).length();
        let sz = Vec3::new(m.m[2][0], m.m[2][1], m.m[2][2]).length();
        let mut rot = m;
        rot.m[0][0] /= sx;
        rot.m[0][1] /= sx;
        rot.m[0][2] /= sx;
        rot.m[1][0] /= sy;
        rot.m[1][1] /= sy;
        rot.m[1][2] /= sy;
        rot.m[2][0] /= sz;
        rot.m[2][1] /= sz;
        rot.m[2][2] /= sz;
        let rotation = Quat::from_mat4(rot);
        Transform { position, rotation, scale: Vec3::new(sx, sy, sz) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec2_arithmetic() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a + b, Vec2::new(4.0, 6.0));
        assert_eq!(b - a, Vec2::new(2.0, 2.0));
        assert_eq!(a * 2.0, Vec2::new(2.0, 4.0));
        assert_eq!(-a, Vec2::new(-1.0, -2.0));
    }

    #[test]
    fn vec2_dot_length_normalise() {
        let a = Vec2::new(3.0, 4.0);
        assert_eq!(a.dot(a), 25.0);
        assert_eq!(a.length(), 5.0);
        assert_eq!(a.normalised(), Vec2::new(0.6, 0.8));
        assert_eq!(Vec2::ZERO.normalised(), Vec2::ZERO);
    }

    #[test]
    fn vec3_cross() {
        let x = Vec3::X;
        let y = Vec3::Y;
        assert_eq!(x.cross(y), Vec3::Z);
        assert_eq!(y.cross(x), -Vec3::Z);
    }

    #[test]
    fn vec3_normalise() {
        let a = Vec3::new(2.0, 0.0, 0.0);
        assert_eq!(a.normalised(), Vec3::X);
    }

    #[test]
    fn vec4_dot() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a.dot(a), 30.0);
    }

    #[test]
    fn mat4_identity() {
        let i = Mat4::IDENTITY;
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(i.transform_point(p), p);
    }

    #[test]
    fn mat4_multiply() {
        let i = Mat4::IDENTITY;
        let a = Mat4 {
            m: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        assert_eq!(a.multiply(i), a);
        assert_eq!(i.multiply(a), a);
    }

    #[test]
    fn mat4_translation() {
        let mut t = Mat4::IDENTITY;
        t.m[3][0] = 10.0;
        t.m[3][1] = 20.0;
        t.m[3][2] = 30.0;
        let p = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(t.transform_point(p), Vec3::new(11.0, 21.0, 31.0));
    }

    #[test]
    fn quat_identity_normalised() {
        let q = Quat::IDENTITY;
        assert_eq!(q.normalised(), q);
    }

    #[test]
    fn quat_normalise_unnormalised() {
        let q = Quat::new(0.0, 0.0, 0.0, 2.0);
        assert_eq!(q.normalised(), Quat::IDENTITY);
    }

    #[test]
    fn quat_axis_angle_rotates_the_axis() {
        // A 90-degree turn around Y maps +Z onto +X.
        let q = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let rotated = q.rotate_vec3(Vec3::Z);
        assert!((rotated - Vec3::X).length() < 1e-5);
    }

    #[test]
    fn quat_multiply_composes_rotations() {
        let a = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let b = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let composed = a.multiply(b);
        let rotated = composed.rotate_vec3(Vec3::Z);
        // Two 90-degree turns make 180 degrees: +Z onto -Z.
        assert!((rotated + Vec3::Z).length() < 1e-5);
    }

    #[test]
    fn quat_to_mat4_matches_rotate() {
        let q = Quat::from_axis_angle(
            Vec3::new(0.5, 1.0, -0.25),
            1.234,
        );
        let v = Vec3::new(3.0, -2.0, 0.75);
        let via_quat = q.rotate_vec3(v);
        let m = q.to_mat4();
        let via_mat = m.transform_point(v);
        assert!((via_quat - via_mat).length() < 1e-5);
    }

    #[test]
    fn quat_slerp_endpoints_and_halfway() {
        let a = Quat::IDENTITY;
        let b = Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI);
        let t0 = a.slerp(b, 0.0);
        let t1 = a.slerp(b, 1.0);
        let half = a.slerp(b, 0.5);
        assert!((t0.w - 1.0).abs() < 1e-5);
        // b is the 180-degree turn (w = 0); slerp's t=1 lands on it.
        assert!((t1.w - b.w).abs() < 1e-5);
        // The halfway rotation is a 90-degree turn around Y in either
        // direction (both are valid halfway rotations of a 180-degree
        // turn): +Z onto ±X.
        let rotated = half.rotate_vec3(Vec3::Z);
        assert!((rotated.length() - 1.0).abs() < 1e-5);
        assert!(rotated.y.abs() < 1e-5);
        assert!(rotated.z.abs() < 1e-5);
        assert!((rotated.x.abs() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn aabb_points_and_union() {
        let empty = Aabb::from_points(&[]);
        assert_eq!(empty.unioned(Aabb::new(Vec3::ZERO, Vec3::ONE)), Aabb::new(Vec3::ZERO, Vec3::ONE));
        let box_ = Aabb::from_points(&[
            Vec3::new(-1.0, 2.0, 0.5),
            Vec3::new(3.0, -2.0, 4.0),
        ]);
        assert_eq!(box_.min, Vec3::new(-1.0, -2.0, 0.5));
        assert_eq!(box_.max, Vec3::new(3.0, 2.0, 4.0));
        assert!((box_.centre() - Vec3::new(1.0, 0.0, 2.25)).length() < 1e-6);
        assert!((box_.extent() - Vec3::new(2.0, 2.0, 1.75)).length() < 1e-6);
    }

    #[test]
    fn frustum_contains_the_view_volume() {
        let vp = Mat4::perspective(1.0, 1.0, 0.1, 100.0)
            .multiply(Mat4::look_at(
                Vec3::new(0.0, 0.0, 5.0),
                Vec3::ZERO,
                Vec3::Y,
            ));
        let frustum = Frustum::from_view_projection(vp);
        // A sphere at the world origin (inside the view volume) is not
        // outside any plane.
        assert!(!frustum.sphere_outside(Vec3::ZERO, 0.1));
        // A sphere far behind the camera is outside.
        assert!(frustum.sphere_outside(Vec3::new(0.0, 0.0, 6.0), 0.1));
    }

    #[test]
    fn aabb_culling_against_frustum() {
        let vp = Mat4::perspective(1.0, 1.0, 0.1, 100.0)
            .multiply(Mat4::look_at(
                Vec3::new(0.0, 0.0, 5.0),
                Vec3::ZERO,
                Vec3::Y,
            ));
        let frustum = Frustum::from_view_projection(vp);
        let inside = Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
        let behind = Aabb::new(
            Vec3::new(-0.5, -0.5, 50.0),
            Vec3::new(0.5, 0.5, 51.0),
        );
        assert!(!inside.outside_frustum(&frustum));
        assert!(behind.outside_frustum(&frustum));
    }

    #[test]
    fn transform_roundtrip_and_compose() {
        let t = Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_axis_angle(Vec3::Y, 0.7),
            Vec3::ONE,
        );
        let m = t.to_mat4();
        let back = Transform::from_mat4(m);
        assert!((back.position - t.position).length() < 1e-5);
        // The composed parent/child transform equals the matrices'
        // product.
        let parent = Transform::new(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::from_axis_angle(Vec3::X, 0.3),
            Vec3::ONE,
        );
        let child = Transform::new(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        );
        let world = Transform::compose(parent, child);
        let via_matrices =
            parent.to_mat4().multiply(child.to_mat4()).transform_point(Vec3::ZERO);
        assert!((world.position - via_matrices).length() < 1e-5);
    }
}
