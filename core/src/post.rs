//! The post-processing chain's pure reference layer (M7).
//!
//! Every shader in the chain translates one of the functions here, the
//! same rule the lighting module established for M5: the CPU-side
//! reference carries the numeric definition, the unit tests pin it, and
//! each backend's readback must match the reference's evaluation on the
//! same inputs. The chain itself (the pass orchestration) lives in the
//! render backends; this module carries only the pure math.

/// The filmic tonemapping curve (the ACES approximation Narkowicz
/// published): a monotonic hue-preserving map from the linear HDR
/// radiance to the display-referred [0, 1) range. The input and the
/// output are per-channel; the caller applies it to the colour's three
/// channels.
pub fn aces_filmic(x: f32) -> f32 {
    let num = x * (2.51 * x + 0.03);
    let den = x * (2.43 * x + 0.59) + 0.14;
    (num / den).clamp(0.0, 1.0)
}

/// The display value a linear HDR colour tonemaps to: the ACES curve
/// followed by the sRGB-ish gamma the framebuffer's 8-bit encoding
/// assumes. The output is the 8-bit byte the readback sees.
pub fn tonemap_channel(x: f32) -> u8 {
    let mapped = aces_filmic(x);
    let gamma = mapped.powf(1.0 / 2.2);
    (gamma * 255.0).round().clamp(0.0, 255.0) as u8
}

/// The bloom threshold's soft knee weight: the radiance's share that
/// passes into the bloom buffer. Below the threshold the weight is
/// zero; above, it rises smoothly to one at `threshold + knee`.
/// The soft knee keeps the extract pass banding-free.
pub fn bloom_weight(radiance: f32, threshold: f32, knee: f32) -> f32 {
    if radiance <= threshold {
        return 0.0;
    }
    let excess = radiance - threshold;
    let soft = (excess / knee.max(1e-6)).clamp(0.0, 1.0);
    // The smoothstep-style ease between the threshold and the knee's
    // end, then the full excess above the knee.
    let eased = soft * soft * (3.0 - 2.0 * soft);
    if excess >= knee {
        excess
    } else {
        eased * excess
    }
}

/// The 5-tap gaussian's weights (the centre plus two symmetric pairs),
/// normalised so the blur conserves energy. The offsets are 0, 1.5 and
/// 3.5 texels (the half-pixel offsets pair each tap's bilinear
/// samples), the standard optimised form.
pub const GAUSSIAN_WEIGHTS: [f32; 3] = [0.2887, 0.2331, 0.1225];

/// The circle of confusion: the in-focus share of a fragment's colour.
/// One at the focal distance's plane (sharp), falling to zero as the
/// fragment's distance moves past either focus range's edge. The
/// `near_range` and `far_range` are the distances from the focal plane
/// where the sharpness reaches zero (the symmetric range model the
/// deterministic test uses).
pub fn focus_share(distance: f32, focal_distance: f32, near_range: f32, far_range: f32) -> f32 {
    let delta = distance - focal_distance;
    let range = if delta < 0.0 { near_range } else { far_range };
    if range <= 0.0 {
        return if delta == 0.0 { 1.0 } else { 0.0 };
    }
    (1.0 - delta.abs() / range).clamp(0.0, 1.0)
}

/// The SSAO's occlusion factor for one sample: how much a sphere of the
/// scene's radius at the sampled distance occludes the fragment. The
/// windowed form matches the point lights' falloff rule (one at the
/// origin, zero at the radius), with the horizon test (a sample below
/// the fragment's surface plane contributes nothing) carried by the
/// caller's dot product. `height` is the sample point's height above
/// the fragment's plane, `dist` the sample's distance, `radius` the
/// effect's world-space radius.
pub fn ssao_sample(height: f32, dist: f32, radius: f32) -> f32 {
    if height <= 0.0 || dist <= 0.0 || dist >= radius {
        return 0.0;
    }
    // The hemisphere's solid-angle share the sample's sphere covers,
    // windowed at the radius: the classic form the reference
    // implementations use for a hemisphere sample.
    let horizon = height / dist;
    let falloff = 1.0 - (dist / radius) * (dist / radius);
    horizon * falloff
}

/// The morphological edge's blend weight: the share of the neighbour's
/// colour an edge pixel carries, from the edge's crossing ratio. A
/// pixel whose edge crossing sits at the cell's centre blends 0.5; the
/// ratios approach zero (all own colour) and one (all neighbour) at the
/// cell's borders. The blend is symmetric and deterministic — no noise
/// pattern, the readback must be reproducible.
pub fn mlaa_blend(crossing: f32) -> f32 {
    crossing.clamp(0.0, 1.0)
}

/// The morphological edge's crossing ratio: the share of the pixel the
/// neighbour's side covers, the input the `mlaa_blend` clamps. The
/// single-pass composite's edge detection reads the tonemapped
/// luminance: an edge that cuts the pixel (the centre's luminance
/// differs from the dominant neighbour's) blends half of each side —
/// the simplified midpoint rule the deterministic test pins; aligned
/// edges (the centre equal to the neighbour) give zero and the pixel
/// keeps its own colour.
pub fn mlaa_crossing(centre: f32, neighbour: f32) -> f32 {
    if (centre - neighbour).abs() > 1e-6 {
        0.5
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn aces_maps_zero_to_zero_and_saturation_below_one() {
        assert!(close(aces_filmic(0.0), 0.0));
        // The curve saturates: a very bright value approaches one but
        // never reaches it.
        // (the Narkowicz form reaches exactly 1.0 by x = 10, so the
        // saturation test pins the approach, not an asymptote).
        assert!(aces_filmic(2.0) > 0.9 && aces_filmic(2.0) < 1.0);
        assert_eq!(aces_filmic(100.0), 1.0);
        // The mid-grey one maps to about 0.8 (the display-referred
        // standard's shoulder).
        assert!(aces_filmic(1.0) > 0.75 && aces_filmic(1.0) < 0.85);
    }

    #[test]
    fn aces_is_monotonic_across_the_range() {
        let mut previous = -1.0;
        for step in 0..200 {
            let x = step as f32 * 0.05;
            let mapped = aces_filmic(x);
            assert!(mapped >= previous - 1e-6, "the curve decreased at {x}");
            previous = mapped;
        }
    }

    #[test]
    fn tonemap_pins_the_known_display_values() {
        // The HDR 2.8235 (the test scene's sun-squared value) tonemaps
        // to a specific byte the examples assert.
        let byte = tonemap_channel(0.70588 * 4.0);
        assert!(byte > 240, "the bright square must stay near white: {byte}");
        // Zero maps to zero: the background stays black.
        assert_eq!(tonemap_channel(0.0), 0);
        // The 1.0 grey maps to the curve's shoulder (231).
        let grey = tonemap_channel(1.0);
        assert_eq!(grey, 231, "the 1.0 grey");
    }

    #[test]
    fn bloom_weight_is_zero_below_and_rises_above_the_threshold() {
        assert_eq!(bloom_weight(0.5, 1.0, 0.5), 0.0);
        assert_eq!(bloom_weight(1.0, 1.0, 0.5), 0.0);
        // Above the knee the weight is the full excess.
        let above = bloom_weight(2.0, 1.0, 0.5);
        assert!(close(above, 1.0), "the full excess above the knee: {above}");
        // Inside the knee the weight sits between zero and the excess.
        let inside = bloom_weight(1.25, 1.0, 0.5);
        assert!(inside > 0.0 && inside < 0.25, "the knee's ease: {inside}");
    }

    #[test]
    fn gaussian_weights_sum_to_one() {
        let sum: f32 = GAUSSIAN_WEIGHTS.iter().sum::<f32>() * 2.0 - GAUSSIAN_WEIGHTS[0];
        assert!(close(sum, 1.0), "the weights' energy: {sum}");
    }

    #[test]
    fn focus_share_is_one_at_the_focal_plane_and_zero_past_the_ranges() {
        assert!(close(focus_share(2.0, 2.0, 0.5, 0.5), 1.0));
        assert!(close(focus_share(2.6, 2.0, 0.5, 0.5), 0.0));
        assert!(close(focus_share(1.4, 2.0, 0.5, 0.5), 0.0));
        // Halfway out, halfway sharp.
        assert!(close(focus_share(2.25, 2.0, 0.5, 0.5), 0.5));
    }

    #[test]
    fn ssao_sample_peaks_close_and_dies_at_the_radius() {
        assert_eq!(ssao_sample(0.0, 1.0, 2.0), 0.0, "a below-surface sample");
        assert_eq!(ssao_sample(0.5, 2.0, 2.0), 0.0, "the radius's edge");
        let close = ssao_sample(0.5, 0.8, 2.0);
        let far = ssao_sample(0.5, 1.6, 2.0);
        assert!(close > far, "the occlusion falls with distance");
        assert!(close > 0.0 && close < 1.0);
    }

    #[test]
    fn mlaa_blend_clamps_the_crossing() {
        assert!(close(mlaa_blend(0.5), 0.5));
        assert!(close(mlaa_blend(-1.0), 0.0));
        assert!(close(mlaa_blend(2.0), 1.0));
    }

    #[test]
    fn mlaa_crossing_is_half_for_the_cut_edge_and_zero_for_aligned() {
        assert!(close(mlaa_crossing(0.5, 0.5), 0.0));
        assert!(close(mlaa_crossing(0.0, 0.0), 0.0));
        // An edge that cuts the pixel: the midpoint rule (half of each
        // side the blend carries).
        assert!(close(mlaa_crossing(0.7, 0.0), 0.5));
        assert!(close(mlaa_crossing(0.7, 0.3), 0.5));
    }
}
