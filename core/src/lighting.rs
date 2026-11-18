//! The engine's own forward-lighting model: the pure shading functions the
//! backends' shaders mirror (the same Blinn-Phong equations, one reference
//! implementation the tests carry and the shaders translate).
//!
//! The module is deliberately free of graphics-API types: the byte layout
//! of the lights' uniform block (the M4c rule) lives in each backend's own
//! mapping, but the shading math is the contract's.

use crate::math::Vec3;
use crate::render::{DirectionalLight, PointLight};

/// One point light's attenuation at the given distance (the windowed
/// inverse-square model the contract's `PointLight` documents: the light
/// falls to zero at the radius, and the window keeps the denominator
/// positive inside it).
pub fn point_attenuation(light: &PointLight, distance: f32) -> f32 {
    if distance >= light.radius || distance < 0.0 {
        return 0.0;
    }
    let window = 1.0 - (distance / light.radius).powi(2);
    window * window / (1.0 + distance * distance)
}

/// The spot light's cone falloff at the given angle between the light's
/// forward direction and the direction to the surface (the smoothstep
/// between the inner and the outer cone angles the contract documents).
pub fn spot_falloff(
    outer_angle_rad: f32,
    inner_angle_rad: f32,
    cos_angle: f32,
) -> f32 {
    let cos_outer = outer_angle_rad.cos();
    let cos_inner = inner_angle_rad.cos();
    if cos_angle <= cos_outer {
        return 0.0;
    }
    if cos_angle >= cos_inner {
        return 1.0;
    }
    let t = (cos_angle - cos_outer) / (cos_inner - cos_outer);
    t * t * (3.0 - 2.0 * t)
}

/// The Blinn-Phong reflection for one light: the lambert diffuse plus the
/// Blinn half-vector specular, the light's colour and intensity scaled.
/// `normal` and `to_light` must be unit length; `to_eye` points from the
/// surface towards the viewer.
pub fn blinn_phong(
    light_color: Vec3,
    intensity: f32,
    attenuation: f32,
    normal: Vec3,
    to_light: Vec3,
    to_eye: Vec3,
    shininess: f32,
    specular: Vec3,
) -> Vec3 {
    let diffuse = normal.dot(to_light).max(0.0);
    let half = (to_light + to_eye).normalised();
    let spec_mag = half.dot(normal).max(0.0).powf(shininess.max(0.0));
    let spec = if shininess > 0.0 && diffuse > 0.0 {
        spec_mag
    } else {
        0.0
    };
    Vec3::new(
        light_color.x * intensity * attenuation * (diffuse + spec * specular.x),
        light_color.y * intensity * attenuation * (diffuse + spec * specular.y),
        light_color.z * intensity * attenuation * (diffuse + spec * specular.z),
    )
}

/// The sun's contribution: the directional light has no position and no
/// attenuation (the direction the light travels in is the negated
/// to-light vector).
pub fn sun_contribution(
    sun: &DirectionalLight,
    normal: Vec3,
    to_eye: Vec3,
    shininess: f32,
    specular: Vec3,
) -> Vec3 {
    let to_light = Vec3::new(
        -sun.direction.x,
        -sun.direction.y,
        -sun.direction.z,
    )
    .normalised();
    blinn_phong(
        sun.color,
        sun.intensity,
        1.0,
        normal,
        to_light,
        to_eye,
        shininess,
        specular,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn light() -> PointLight {
        PointLight {
            position: Vec3::new(0.0, 0.0, 0.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 1.0,
            radius: 10.0,
        }
    }

    #[test]
    fn point_attenuation_is_one_at_the_origin_and_zero_at_the_radius() {
        assert!((point_attenuation(&light(), 0.0) - 1.0).abs() < 1e-6);
        assert_eq!(point_attenuation(&light(), 10.0), 0.0);
        assert_eq!(point_attenuation(&light(), 11.0), 0.0);
    }

    #[test]
    fn point_attenuation_falls_with_distance_and_stays_positive() {
        let near = point_attenuation(&light(), 1.0);
        let far = point_attenuation(&light(), 5.0);
        assert!(near > far && far > 0.0);
    }

    #[test]
    fn spot_falloff_is_one_inside_the_inner_cone_and_zero_outside() {
        let outer = 0.6;
        let inner = 0.3;
        assert!((spot_falloff(outer, inner, 1.0) - 1.0).abs() < 1e-6);
        assert_eq!(spot_falloff(outer, inner, (outer + 0.01).cos()), 0.0);
    }

    #[test]
    fn spot_falloff_is_smooth_between_the_cones() {
        let outer: f32 = 0.6;
        let inner: f32 = 0.3;
        // The falloff's interpolation runs in the cosine space (the
        // smoothstep's t = (cos - cos_outer) / (cos_inner - cos_outer)),
        // so the exact midpoint sits at the cosines' midpoint.
        let mid_cos = (outer.cos() + inner.cos()) / 2.0;
        let mid = spot_falloff(outer, inner, mid_cos);
        assert!(mid > 0.0 && mid < 1.0);
        // The smoothstep's midpoint is exactly 0.5.
        assert!((mid - 0.5).abs() < 1e-6);
    }

    #[test]
    fn blinn_phong_full_facing_surface_carries_the_full_diffuse() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let to_light = Vec3::new(0.0, 0.0, 1.0);
        let to_eye = Vec3::new(0.0, 0.0, 1.0);
        let out = blinn_phong(
            Vec3::new(1.0, 1.0, 1.0),
            1.0,
            1.0,
            normal,
            to_light,
            to_eye,
            0.0,
            Vec3::new(0.0, 0.0, 0.0),
        );
        assert!((out.x - 1.0).abs() < 1e-6);
        assert!((out.y - 1.0).abs() < 1e-6);
        assert!((out.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn blinn_phong_grazing_surface_carries_no_diffuse() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let to_light = Vec3::new(1.0, 0.0, 0.0);
        let to_eye = Vec3::new(0.0, 0.0, 1.0);
        let out = blinn_phong(
            Vec3::new(1.0, 1.0, 1.0),
            1.0,
            1.0,
            normal,
            to_light,
            to_eye,
            0.0,
            Vec3::new(0.0, 0.0, 0.0),
        );
        assert!(out.x.abs() < 1e-6);
    }

    #[test]
    fn blinn_phong_specular_peaks_at_the_half_vector() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        // 45 degrees between the to-light and the to-eye directions: the
        // half vector sits exactly on the normal, the specular's peak.
        let to_light = Vec3::new(1.0, 0.0, 1.0).normalised();
        let to_eye = Vec3::new(-1.0, 0.0, 1.0).normalised();
        let out = blinn_phong(
            Vec3::new(1.0, 1.0, 1.0),
            1.0,
            1.0,
            normal,
            to_light,
            to_eye,
            32.0,
            Vec3::new(1.0, 1.0, 1.0),
        );
        // Diffuse = cos(45) ~ 0.7071, specular = 1.0: the sum carries
        // both.
        let expected = 0.70710678 + 1.0;
        assert!((out.x - expected).abs() < 1e-4);
    }

    #[test]
    fn sun_contribution_negates_the_direction() {
        let sun = DirectionalLight {
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::new(1.0, 0.0, 0.0),
            intensity: 1.0,
        };
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let to_eye = Vec3::new(0.0, 0.0, 1.0);
        let out = sun_contribution(&sun, normal, to_eye, 0.0, Vec3::new(0.0, 0.0, 0.0));
        // The light travels along -Z, the surface faces +Z: full hit.
        assert!((out.x - 1.0).abs() < 1e-6);
    }
}
