//! The Metal backend's shader library. RULE (the same rule the Vulkan and
//! the OpenGL libraries record): every engine and test shader source lives
//! ONLY in this file, under the Metal module. The API-neutral render
//! contract never contains shading-language source.
//!
//! The shaders are MSL 2.0, compiled at runtime by the device's own
//! MTLLibrary compiler (the Metal path's rule: no offline SPIR-V step —
//! that is the Vulkan path's rule). The uniform block is the same layout
//! the Vulkan block carries: the MVP matrix (column-major, the engine's
//! Mat4 convention) followed by the RGBA colour, one [[buffer(0)]] bound
//! constant struct.

/// The diagnostic triangle's vertex shader (MSL 2.0).
pub const TRIANGLE_VERTEX_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct Uniforms
{
    float4x4 mvp;
    float4 color;
};
struct VertexOut
{
    float4 position [[position]];
};
struct VertexIn
{
    float3 position [[attribute(0)]];
};
vertex VertexOut vertexMain(VertexIn in [[stage_in]],
                            constant Uniforms& u [[buffer(0)]])
{
    VertexOut out;
    out.position = u.mvp * float4(in.position, 1.0);
    return out;
}
";

/// The diagnostic triangle's fragment shader (MSL 2.0).
pub const TRIANGLE_FRAGMENT_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct Uniforms
{
    float4x4 mvp;
    float4 color;
};
fragment float4 fragmentMain(constant Uniforms& u [[buffer(0)]])
{
    return u.color;
}
";

/// The generic mesh's vertex shader: the PNU vertex (the attribute table
/// the pipeline's MTLVertexDescriptor carries), the camera block (the
/// view-projection matrix the backend's set_camera builds) and the model
/// matrix the submission carries. RULE (M4c): the same Camera struct
/// declaration both stages carry — the uniform block is one byte layout
/// the two libraries share, so the fragment stage's struct stays
/// bit-identical to the vertex stage's.
pub const MESH_VERTEX_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct Camera
{
    float4x4 view_projection;
    float4x4 model;
};
struct MeshVertexIn
{
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(2)]];
};
struct MeshVertexOut
{
    float4 position [[position]];
    float3 normal;
    float2 uv;
};
vertex MeshVertexOut meshVertexMain(MeshVertexIn in [[stage_in]],
                                    constant Camera& camera [[buffer(0)]])
{
    MeshVertexOut out;
    float4 world = camera.model * float4(in.position, 1.0);
    out.position = camera.view_projection * world;
    out.normal = in.normal;
    out.uv = in.uv;
    return out;
}
";

/// The generic mesh's fragment shader: the bound texture's sample times
/// the surface tint (the material path's minimum). The Camera struct is
/// the same declaration the vertex stage carries (the M4c rule) even
/// though this stage only reads the texture — the sampler is
/// [[sampler(0)]] and the texture [[texture(0)]].
pub const MESH_FRAGMENT_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct Camera
{
    float4x4 view_projection;
    float4x4 model;
};
struct MeshVertexOut
{
    float4 position [[position]];
    float3 normal;
    float2 uv;
};
fragment float4 meshFragmentMain(MeshVertexOut in [[stage_in]],
                                 constant Camera& camera [[buffer(0)]],
                                 texture2d<float> surface_texture [[texture(0)]],
                                 sampler surface_sampler [[sampler(0)]])
{
    return surface_texture.sample(surface_sampler, in.uv);
}
";

/// The lit mesh's vertex shader (M5): the world position and the
/// world-space normal join the outputs (the lighting's inputs).
pub const LIT_VERTEX_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct Camera
{
    float4x4 view_projection;
    float4x4 model;
};
struct MeshVertexIn
{
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(2)]];
};
struct LitVertexOut
{
    float4 position [[position]];
    float3 normal;
    float2 uv;
    float3 world;
};
vertex LitVertexOut litVertexMain(MeshVertexIn in [[stage_in]],
                                  constant Camera& camera [[buffer(0)]])
{
    LitVertexOut out;
    float4 world = camera.model * float4(in.position, 1.0);
    out.position = camera.view_projection * world;
    // The matrix cast float4x4 -> float3x3 does not exist in the Metal
    // shading language: the 3x3 normal matrix is built from the model's
    // columns (the uniform scale keeps the normal direction).
    float3x3 normal_matrix = float3x3(
        camera.model[0].xyz,
        camera.model[1].xyz,
        camera.model[2].xyz);
    out.normal = normal_matrix * in.normal;
    out.uv = in.uv;
    out.world = world.xyz;
    return out;
}
";

/// The lit mesh's fragment shader (M5): the same Blinn-Phong shading
/// the other libraries carry (the sun plus up to four point lights with
/// the windowed inverse-square attenuation). The lights' buffer is
/// [[buffer(1)]] — the same byte layout the other backends map.
pub const LIT_FRAGMENT_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct LitVertexOut
{
    float4 position [[position]];
    float3 normal;
    float2 uv;
    float3 world;
};
struct Lights
{
    float4 sun_direction;
    float4 sun_color_intensity;
    float4 point_position_radius[4];
    float4 point_color_intensity[4];
    float4 params;
};
fragment float4 litFragmentMain(LitVertexOut in [[stage_in]],
                                texture2d<float> surface_texture [[texture(0)]],
                                sampler surface_sampler [[sampler(0)]],
                                constant Lights& lights [[buffer(1)]])
{
    float3 normal = normalize(in.normal);
    float3 base = surface_texture.sample(surface_sampler, in.uv).rgb;
    float3 sun_dir = normalize(-lights.sun_direction.xyz);
    float sun_diffuse = max(dot(normal, sun_dir), 0.0);
    float3 colour = base * lights.sun_color_intensity.rgb
        * lights.sun_color_intensity.a * sun_diffuse;
    for (int i = 0; i < 4; i++) {
        if (float(i) >= lights.params.x) {
            break;
        }
        float3 to_light = lights.point_position_radius[i].xyz - in.world;
        // The variable cannot be named `distance` — the shadowing
        // of the builtin function confuses the Metal front end.
        float dist = length(to_light);
        to_light /= dist;
        float radius = lights.point_position_radius[i].w;
        float window = 1.0 - (dist / radius) * (dist / radius);
        float attenuation = window * window / (1.0 + dist * dist);
        if (dist >= radius) {
            attenuation = 0.0;
        }
        float diffuse = max(dot(normal, to_light), 0.0);
        colour += base * lights.point_color_intensity[i].rgb
            * lights.point_color_intensity[i].a * attenuation * diffuse;
    }
    return float4(colour, 1.0);
}
";
/// The deferred geometry pass's vertex shader (M6): the mesh's stage_in
/// draw with the camera block at [[buffer(0)]] — the same inputs the lit
/// vertex stage carries (the world position joins the outputs).
pub const GBUFFER_VERTEX_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct Camera
{
    float4x4 view_projection;
    float4x4 model;
};
struct MeshVertexIn
{
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(2)]];
};
struct GbufferVertexOut
{
    float4 position [[position]];
    float3 normal;
    float2 uv;
    float3 world;
};
vertex GbufferVertexOut gbufferVertexMain(MeshVertexIn in [[stage_in]],
                                          constant Camera& camera [[buffer(0)]])
{
    GbufferVertexOut out;
    float4 world = camera.model * float4(in.position, 1.0);
    out.position = camera.view_projection * world;
    // The matrix cast float4x4 -> float3x3 does not exist in the Metal
    // shading language: the 3x3 normal matrix is built from the model's
    // columns (the same rule the lit vertex stage records).
    float3x3 normal_matrix = float3x3(
        camera.model[0].xyz,
        camera.model[1].xyz,
        camera.model[2].xyz);
    out.normal = normal_matrix * in.normal;
    out.uv = in.uv;
    out.world = world.xyz;
    return out;
}
";
/// The deferred geometry pass's fragment shader (M6): the G-buffer's
/// three attachments — the albedo (the alpha 0 marks geometry), the
/// half-offset normal and the world position. The attachment indices
/// follow the pass descriptor's colorAttachments order (0, 1, 2).
pub const GBUFFER_FRAGMENT_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct GbufferVertexOut
{
    float4 position [[position]];
    float3 normal;
    float2 uv;
    float3 world;
};
struct GbufferOut
{
    float4 albedo [[color(0)]];
    float4 normal [[color(1)]];
    float4 world [[color(2)]];
};
fragment GbufferOut gbufferFragmentMain(GbufferVertexOut in [[stage_in]],
                                        texture2d<float> surface_texture [[texture(0)]],
                                        sampler surface_sampler [[sampler(0)]])
{
    GbufferOut out;
    float4 base = surface_texture.sample(surface_sampler, in.uv);
    out.albedo = float4(base.rgb, 0.0);
    out.normal = float4(normalize(in.normal) * 0.5 + 0.5, 1.0);
    out.world = float4(in.world, 1.0);
    return out;
}
";
/// The deferred lighting pass's vertex shader (M6): the full-frame
/// triangle the vertex id generates (no vertex buffer, the same shape
/// the Vulkan and the D3D12 lighting vertex stages carry).
pub const LIGHTING_VERTEX_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct LightingVertexOut
{
    float4 position [[position]];
    float2 uv;
};
vertex LightingVertexOut lightingVertexMain(uint vertex_id [[vertex_id]])
{
    LightingVertexOut out;
    float2 corner = float2(float((vertex_id & 1u) << 2u) - 1.0,
                           float((vertex_id & 2u) << 1u) - 1.0);
    out.uv = corner * 0.5 + 0.5;
    out.position = float4(corner, 0.0, 1.0);
    return out;
}
";
/// The deferred lighting pass's fragment shader (M6): the G-buffer's
/// three attachments sampled, the same Blinn-Phong the forward path
/// carries, the background pixels copied through the albedo's alpha
/// rule. The lights' buffer is [[buffer(1)]] at the fragment stage (the
/// same byte layout the lit path maps).
pub const LIGHTING_FRAGMENT_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct LightingVertexOut
{
    float4 position [[position]];
    float2 uv;
};
struct Lights
{
    float4 sun_direction;
    float4 sun_color_intensity;
    float4 point_position_radius[4];
    float4 point_color_intensity[4];
    float4 params;
};
fragment float4 lightingFragmentMain(LightingVertexOut in [[stage_in]],
                                     constant Lights& lights [[buffer(1)]],
                                     texture2d<float> gbuffer_albedo [[texture(0)]],
                                     texture2d<float> gbuffer_normal [[texture(1)]],
                                     texture2d<float> gbuffer_world [[texture(2)]],
                                     sampler surface_sampler [[sampler(0)]])
{
    float4 albedo = gbuffer_albedo.sample(surface_sampler, in.uv);
    if (albedo.a != 0.0) {
        // The background: the G-buffer's clear colour passes through.
        return albedo;
    }
    float3 normal = normalize(gbuffer_normal.sample(surface_sampler, in.uv).rgb * 2.0 - 1.0);
    float3 world = gbuffer_world.sample(surface_sampler, in.uv).rgb;
    float3 base = albedo.rgb;
    float3 sun_dir = normalize(-lights.sun_direction.xyz);
    float sun_diffuse = max(dot(normal, sun_dir), 0.0);
    float3 colour = base * lights.sun_color_intensity.rgb
        * lights.sun_color_intensity.a * sun_diffuse;
    for (int i = 0; i < 4; i++) {
        if (float(i) >= lights.params.x) {
            break;
        }
        float3 to_light = lights.point_position_radius[i].xyz - world;
        // The variable cannot be named `distance` — the shadowing of
        // the builtin function confuses the Metal front end (the M5
        // rule).
        float dist = length(to_light);
        to_light /= dist;
        float radius = lights.point_position_radius[i].w;
        float window = 1.0 - (dist / radius) * (dist / radius);
        float attenuation = window * window / (1.0 + dist * dist);
        if (dist >= radius) {
            attenuation = 0.0;
        }
        float diffuse = max(dot(normal, to_light), 0.0);
        colour += base * lights.point_color_intensity[i].rgb
            * lights.point_color_intensity[i].a * attenuation * diffuse;
    }
    return float4(colour, 1.0);
}
";
/// The post chain's geometry pass's fragment shader (M7): the two HDR
/// writes — the full Blinn-Phong shading's linear radiance (color(0))
/// and the fragment's world position (color(1), the distances the SSAO
/// and the depth of field's focus share read). The background fragments
/// write the clear colour's radiance and a zero world position (the
/// alpha 0 marks background for the composite). The same shading the
/// forward path's shader carries. The lights' buffer is [[buffer(1)]]
/// at the fragment stage (the same byte layout the lit path maps).
pub const POST_GBUFFER_FRAGMENT_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct GbufferVertexOut
{
    float4 position [[position]];
    float3 normal;
    float2 uv;
    float3 world;
};
struct Lights
{
    float4 sun_direction;
    float4 sun_color_intensity;
    float4 point_position_radius[4];
    float4 point_color_intensity[4];
    float4 params;
};
struct PostGbufferOut
{
    float4 radiance [[color(0)]];
    float4 world [[color(1)]];
};
struct Volume
{
    float4 origin;
    float4 dims;
    float4 params;
};
// The light volume's face selection: the face the normal faces (the
// same rule the CPU reference's face_for_normal carries — the six
// axis directions' largest dot product).
int volume_face(float3 normal)
{
    float3 directions[6];
    directions[0] = float3(1.0, 0.0, 0.0);
    directions[1] = float3(-1.0, 0.0, 0.0);
    directions[2] = float3(0.0, 1.0, 0.0);
    directions[3] = float3(0.0, -1.0, 0.0);
    directions[4] = float3(0.0, 0.0, 1.0);
    directions[5] = float3(0.0, 0.0, -1.0);
    int best = 0;
    float best_dot = -1e10;
    for (int face = 0; face < 6; face++) {
        float current = dot(normal, directions[face]);
        if (current > best_dot) {
            best_dot = current;
            best = face;
        }
    }
    return best;
}
fragment PostGbufferOut postGbufferFragmentMain(GbufferVertexOut in [[stage_in]],
                                                constant Lights& lights [[buffer(1)]],
                                                constant Volume& volume [[buffer(2)]],
                                                texture2d<float> surface_texture [[texture(0)]],
                                                texture3d<float> volume_texture [[texture(1)]],
                                                sampler surface_sampler [[sampler(0)]])
{
    PostGbufferOut out;
    float3 normal = normalize(in.normal);
    float3 base = surface_texture.sample(surface_sampler, in.uv).rgb;
    float3 sun_dir = normalize(-lights.sun_direction.xyz);
    float sun_diffuse = max(dot(normal, sun_dir), 0.0);
    float3 colour = base * lights.sun_color_intensity.rgb
        * lights.sun_color_intensity.a * sun_diffuse;
    for (int i = 0; i < 4; i++) {
        if (float(i) >= lights.params.x) {
            break;
        }
        float3 to_light = lights.point_position_radius[i].xyz - in.world;
        // The variable cannot be named `distance` — the shadowing of
        // the builtin function confuses the Metal front end (the M5
        // rule).
        float dist = length(to_light);
        to_light /= dist;
        float radius = lights.point_position_radius[i].w;
        float window = 1.0 - (dist / radius) * (dist / radius);
        float attenuation = window * window / (1.0 + dist * dist);
        if (dist >= radius) {
            attenuation = 0.0;
        }
        float diffuse = max(dot(normal, to_light), 0.0);
        colour += base * lights.point_color_intensity[i].rgb
            * lights.point_color_intensity[i].a * attenuation * diffuse;
    }
    // The indirect light's term (M8): the volume's radiance shades the
    // surface's albedo (the `albedo * irradiance` rule the direct
    // lighting's diffuse carries) — the face the normal faces'
    // trilinear sample, scaled by the config's indirect intensity.
    if (volume.params.z > 0.5) {
        int face = volume_face(normal);
        int cells_z = int(volume.dims.z / 6.0);
        float3 continuous = (in.world - volume.origin.xyz)
            / volume.params.x - 0.5;
        int base[3];
        float fracs[3];
        for (int axis = 0; axis < 3; axis++) {
            float limit = axis == 0 ? volume.dims.x
                : (axis == 1 ? volume.dims.y : float(cells_z));
            if (continuous[axis] <= 0.0) {
                base[axis] = 0;
                fracs[axis] = 0.0;
            } else if (continuous[axis] >= limit - 1.0) {
                base[axis] = int(limit) - 1;
                fracs[axis] = 0.0;
            } else {
                base[axis] = int(floor(continuous[axis]));
                fracs[axis] = continuous[axis] - floor(continuous[axis]);
            }
        }
        float3 result = float3(0.0);
        for (int corner = 0; corner < 8; corner++) {
            int cx = base.x + ((corner & 1) != 0 ? 1 : 0);
            int cy = base.y + ((corner & 2) != 0 ? 1 : 0);
            int cz = base.z + ((corner & 4) != 0 ? 1 : 0);
            float wx = cx >= int(volume.dims.x) ? 0.0
                : ((corner & 1) != 0 ? fracs.x : 1.0 - fracs.x);
            float wy = cy >= int(volume.dims.y) ? 0.0
                : ((corner & 2) != 0 ? fracs.y : 1.0 - fracs.y);
            float wz = cz >= cells_z ? 0.0
                : ((corner & 4) != 0 ? fracs.z : 1.0 - fracs.z);
            float weight = wx * wy * wz;
            if (weight == 0.0) {
                continue;
            }
            int clamped_x = min(cx, int(volume.dims.x) - 1);
            int clamped_y = min(cy, int(volume.dims.y) - 1);
            int clamped_z = min(cz, cells_z - 1);
            // The face the normal faces' layer: the depth layer
            // `z * 6 + face` (the same pack the upload carries).
            float3 texel = float3(
                float(clamped_x) + 0.5,
                float(clamped_y) + 0.5,
                float(clamped_z * 6 + face) + 0.5
            ) / float3(volume.dims.x, volume.dims.y, volume.dims.z);
            result += weight * volume_texture.sample(
                surface_sampler, texel).rgb;
        }
        colour += base * result * volume.params.y;
    }
    out.radiance = float4(colour, 1.0);
    out.world = float4(in.world, 1.0);
    return out;
}
";
/// The post chain's composite pass's fragment shader (M7): the two HDR
/// attachments sampled and the enabled effects applied in the fixed
/// order — the SSAO's occlusion darkens the radiance, the bloom's
/// threshold-and-blur lifts the bright regions, the depth of field's
/// focus share blends the blurred radiance, and the filmic tonemapping
/// maps the result onto the display's range; the morphological
/// anti-aliasing's edge blend follows the tonemap (the LDR image the
/// edge detection reads). The config block carries the toggles and the
/// parameters (the same std140 shape the CPU reference's
/// PostChainConfig values map onto).
pub const POST_COMPOSITE_FRAGMENT_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct LightingVertexOut
{
    float4 position [[position]];
    float2 uv;
};
struct Config
{
    float4 params0; // tonemapping, bloom, dof, ssao (the 1.0/0.0 flags)
    float4 params1; // bloom_threshold, bloom_knee, bloom_intensity, ssao_radius
    float4 params2; // focal_distance, focus_near_range, focus_far_range, anti_aliasing
};
float3 sample_blurred(texture2d<float> tex, float2 uv, float2 texel,
    sampler surface_sampler)
{
    // The 5-tap gaussian: the centre plus the two symmetric pairs at
    // the 1.5-texel and 3.5-texel offsets (the weights the CPU
    // reference's GAUSSIAN_WEIGHTS carry).
    float3 sum = tex.sample(surface_sampler, uv).rgb * 0.2887;
    sum += tex.sample(surface_sampler, uv + float2(1.5, 0.0) * texel).rgb * 0.2331;
    sum += tex.sample(surface_sampler, uv - float2(1.5, 0.0) * texel).rgb * 0.2331;
    sum += tex.sample(surface_sampler, uv + float2(0.0, 1.5) * texel).rgb * 0.2331;
    sum += tex.sample(surface_sampler, uv - float2(0.0, 1.5) * texel).rgb * 0.2331;
    sum += tex.sample(surface_sampler, uv + float2(3.5, 0.0) * texel).rgb * 0.1225;
    sum += tex.sample(surface_sampler, uv - float2(3.5, 0.0) * texel).rgb * 0.1225;
    sum += tex.sample(surface_sampler, uv + float2(0.0, 3.5) * texel).rgb * 0.1225;
    sum += tex.sample(surface_sampler, uv - float2(0.0, 3.5) * texel).rgb * 0.1225;
    return sum;
}
float bloom_weight(float radiance, float threshold, float knee)
{
    if (radiance <= threshold) {
        return 0.0;
    }
    float excess = radiance - threshold;
    float soft = min(excess / max(knee, 1e-6), 1.0);
    float eased = soft * soft * (3.0 - 2.0 * soft);
    if (excess >= knee) {
        return excess;
    }
    return eased * excess;
}
float focus_share(float distance, float focal, float near_range, float far_range)
{
    float delta = distance - focal;
    float range = delta < 0.0 ? near_range : far_range;
    return clamp(1.0 - abs(delta) / max(range, 1e-6), 0.0, 1.0);
}
float ssao_sample(float height, float dist, float radius)
{
    if (height <= 0.0 || dist <= 0.0 || dist >= radius) {
        return 0.0;
    }
    float horizon = height / dist;
    float falloff = 1.0 - (dist / radius) * (dist / radius);
    return horizon * falloff;
}
float mlaa_crossing(float centre, float neighbour)
{
    // The midpoint rule: an edge that cuts the pixel (the centre's
    // luminance differs from the neighbour's) blends half of each side;
    // aligned edges keep the pixel's own colour.
    if (abs(centre - neighbour) > 1e-6) {
        return 0.5;
    }
    return 0.0;
}
float mlaa_blend(float crossing)
{
    return clamp(crossing, 0.0, 1.0);
}
fragment float4 postCompositeFragmentMain(LightingVertexOut in [[stage_in]],
                                          constant Config& config [[buffer(0)]],
                                          texture2d<float> radiance_texture [[texture(0)]],
                                          texture2d<float> world_texture [[texture(1)]],
                                          sampler surface_sampler [[sampler(0)]])
{
    float4 radiance_sample = radiance_texture.sample(surface_sampler, in.uv);
    float4 world_sample = world_texture.sample(surface_sampler, in.uv);
    float3 colour = radiance_sample.rgb;
    bool background = world_sample.a == 0.0;
    float texel_x = 1.0 / 320.0;
    float texel_y = 1.0 / 240.0;
    // The SSAO: the four hemisphere samples around the fragment's world
    // position (the +x, -x, +y, -y offsets at the radius's eighth), the
    // occlusion the world buffer's own heights against the fragment's
    // plane carry.
    if (config.params0.w > 0.5 && !background) {
        float3 world = world_sample.rgb;
        float radius = config.params1.w;
        float occlusion = 0.0;
        float2 offsets[4] = { float2(1.0, 0.0), float2(-1.0, 0.0),
                              float2(0.0, 1.0), float2(0.0, -1.0) };
        for (int i = 0; i < 4; i++) {
            float2 offset = offsets[i] * radius * 0.125;
            float3 sample_world = world_texture.sample(surface_sampler,
                in.uv + float2(offset.x * texel_x, offset.y * texel_y)).rgb;
            float3 to_sample = sample_world - world;
            float dist = length(to_sample);
            float height = to_sample.y;
            occlusion += ssao_sample(height, dist, radius);
        }
        colour *= 1.0 - clamp(occlusion * 0.25, 0.0, 1.0) * 0.5;
    }
    // The bloom: the blurred radiance's weight above the threshold
    // lifts the frame's bright regions (the composite's intensity the
    // params carry scales it).
    if (config.params0.y > 0.5) {
        float3 blurred = sample_blurred(radiance_texture, in.uv,
            float2(texel_x, texel_y), surface_sampler);
        float weight = bloom_weight(max(blurred.r, max(blurred.g, blurred.b)),
            config.params1.x, config.params1.y);
        colour += blurred * weight * config.params1.z;
    }
    // The depth of field: the focus share blends the blurred radiance
    // over the sharp one (the background fragments stay untouched — the
    // chain blurs the scene's own geometry).
    if (config.params0.z > 0.5 && !background) {
        float3 blurred = sample_blurred(radiance_texture, in.uv,
            float2(texel_x, texel_y), surface_sampler);
        float distance = length(world_sample.rgb);
        float share = 1.0 - focus_share(distance, config.params2.x,
            config.params2.y, config.params2.z);
        colour = mix(colour, blurred, clamp(share, 0.0, 1.0));
    }
    // The tonemapping: the filmic curve maps the linear radiance onto
    // the display's [0, 1] (the gamma the framebuffer's 8-bit encoding
    // assumes follows).
    if (config.params0.x > 0.5) {
        float num = colour.r * (2.51 * colour.r + 0.03);
        float den = colour.r * (2.43 * colour.r + 0.59) + 0.14;
        float mapped_r = clamp(num / den, 0.0, 1.0);
        num = colour.g * (2.51 * colour.g + 0.03);
        den = colour.g * (2.43 * colour.g + 0.59) + 0.14;
        float mapped_g = clamp(num / den, 0.0, 1.0);
        num = colour.b * (2.51 * colour.b + 0.03);
        den = colour.b * (2.43 * colour.b + 0.59) + 0.14;
        float mapped_b = clamp(num / den, 0.0, 1.0);
        colour = float3(mapped_r, mapped_g, mapped_b);
        colour = pow(colour, float3(1.0 / 2.2));
    }
    // The morphological anti-aliasing: the tonemapped luminance's edge
    // detection reads the four direct neighbours, the crossing ratio
    // against the dominant edge neighbour drives the blend (the single-
    // pass approximation of the classic three-pass morphological
    // operator — the deterministic edge test the example carries).
    if (config.params2.w > 0.5) {
        float centre = dot(colour, float3(0.299, 0.587, 0.114));
        float3 tonemapped[4];
        tonemapped[0] = radiance_texture.sample(surface_sampler,
            in.uv + float2(-texel_x, 0.0)).rgb;
        tonemapped[1] = radiance_texture.sample(surface_sampler,
            in.uv + float2(texel_x, 0.0)).rgb;
        tonemapped[2] = radiance_texture.sample(surface_sampler,
            in.uv + float2(0.0, texel_y)).rgb;
        tonemapped[3] = radiance_texture.sample(surface_sampler,
            in.uv + float2(0.0, -texel_y)).rgb;
        if (config.params0.x > 0.5) {
            for (int i = 0; i < 4; i++) {
                float num = tonemapped[i].r * (2.51 * tonemapped[i].r + 0.03);
                float den = tonemapped[i].r * (2.43 * tonemapped[i].r + 0.59) + 0.14;
                tonemapped[i].r = clamp(num / den, 0.0, 1.0);
                num = tonemapped[i].g * (2.51 * tonemapped[i].g + 0.03);
                den = tonemapped[i].g * (2.43 * tonemapped[i].g + 0.59) + 0.14;
                tonemapped[i].g = clamp(num / den, 0.0, 1.0);
                num = tonemapped[i].b * (2.51 * tonemapped[i].b + 0.03);
                den = tonemapped[i].b * (2.43 * tonemapped[i].b + 0.59) + 0.14;
                tonemapped[i].b = clamp(num / den, 0.0, 1.0);
                tonemapped[i] = pow(tonemapped[i], float3(1.0 / 2.2));
            }
        }
        float luma[4];
        luma[0] = dot(tonemapped[0], float3(0.299, 0.587, 0.114));
        luma[1] = dot(tonemapped[1], float3(0.299, 0.587, 0.114));
        luma[2] = dot(tonemapped[2], float3(0.299, 0.587, 0.114));
        luma[3] = dot(tonemapped[3], float3(0.299, 0.587, 0.114));
        int dominant = 0;
        float dominant_crossing = mlaa_crossing(centre, luma[0]);
        for (int i = 1; i < 4; i++) {
            float crossing = mlaa_crossing(centre, luma[i]);
            if (crossing > dominant_crossing) {
                dominant_crossing = crossing;
                dominant = i;
            }
        }
        colour = mix(colour, tonemapped[dominant], mlaa_blend(dominant_crossing));
    }
    return float4(colour, 1.0);
}
";
