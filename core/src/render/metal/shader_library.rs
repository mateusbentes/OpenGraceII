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
