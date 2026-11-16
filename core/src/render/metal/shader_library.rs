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
struct VertexUniforms
{
    float4x4 mvp;
};
struct VertexOut
{
    float4 position [[position]];
};
vertex VertexOut vertexMain(uint vid [[vertex_id]],
                            float3 position [[attribute(0)]],
                            constant VertexUniforms& u [[buffer(0)]])
{
    VertexOut out;
    out.position = u.mvp * float4(position, 1.0);
    return out;
}
";

/// The diagnostic triangle's fragment shader (MSL 2.0).
pub const TRIANGLE_FRAGMENT_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct FragmentUniforms
{
    float4 color;
};
fragment float4 fragmentMain(constant FragmentUniforms& u [[buffer(0)]])
{
    return u.color;
}
";
