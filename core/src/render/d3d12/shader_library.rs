//! The Direct3D12 backend's HLSL shader sources.
//!
//! The shaders are the same diagnostic triangle the Vulkan, the OpenGL
//! and the Metal backends record: one vertex stage (the position through
//! the MVP) and one pixel stage (the uniform colour). HLSL is compiled at
//! runtime by D3DCompile (the platform's own compiler; the same rule the
//! OpenGL backend's GLSL runtime compilation records — no SPIR-V
//! cross-compilation, no third-party compiler).
//!
//! The engine's Mat4 is COLUMN-MAJOR (m[c][r]). HLSL's `column_major`
//! qualifier on the constant buffer makes the mul(mvp, position) read the
//! matrix the engine uploaded, so the upload is as-is.

/// The diagnostic triangle's vertex shader (HLSL 5.0).
pub const TRIANGLE_VERTEX_SHADER: &str = "\
struct VsIn {\n\
\x20   float3 position : POSITION;\n\
};\n\
\n\
struct VsOut {\n\
\x20   float4 position : SV_POSITION;\n\
};\n\
\n\
cbuffer Frame : register(b0) {\n\
\x20   column_major float4x4 mvp;\n\
\x20   float4 color;\n\
};\n\
\n\
VsOut vsMain(VsIn input) {\n\
\x20   VsOut output;\n\
\x20   output.position = mul(mvp, float4(input.position, 1.0));\n\
\x20   return output;\n\
}\n\
";

/// The diagnostic triangle's pixel shader (HLSL 5.0).
pub const TRIANGLE_PIXEL_SHADER: &str = "\
struct VsOut {\n\
\x20   float4 position : SV_POSITION;\n\
};\n\
\n\
cbuffer Frame : register(b0) {\n\
\x20   column_major float4x4 mvp;\n\
\x20   float4 color;\n\
};\n\
\n\
float4 psMain(VsOut input) : SV_TARGET {\n\
\x20   return color;\n\
}\n\
";

/// The generic mesh's vertex shader (HLSL 5.0): the PNU layout's three
/// attributes (position 0, normal 1, texcoord0 2) through the camera's
/// view-projection and the model matrix. The same two-stage uniform
/// declaration rule the Vulkan block records: both stages declare the
/// same register(b0) block so the offsets line up.
pub const MESH_VERTEX_SHADER: &str = "\
struct MeshVsIn {\n\
\x20   float3 position : POSITION;\n\
\x20   float3 normal : NORMAL;\n\
\x20   float2 texcoord0 : TEXCOORD0;\n\
};\n\
\n\
struct MeshVsOut {\n\
\x20   float4 position : SV_POSITION;\n\
\x20   float2 uv : TEXCOORD0;\n\
};\n\
\n\
cbuffer Frame : register(b0) {\n\
\x20   column_major float4x4 view_projection;\n\
\x20   column_major float4x4 model;\n\
};\n\
\n\
MeshVsOut vsMain(MeshVsIn input) {\n\
\x20   MeshVsOut output;\n\
\x20   float4 world = mul(model, float4(input.position, 1.0));\n\
\x20   output.position = mul(view_projection, world);\n\
\x20   output.uv = input.texcoord0;\n\
\x20   return output;\n\
}\n\
";

/// The generic mesh's pixel shader (HLSL 5.0): the bound texture's
/// sample (the sampler's register(s0) the root signature's static
/// sampler covers).
pub const MESH_PIXEL_SHADER: &str = "\
struct MeshVsOut {\n\
\x20   float4 position : SV_POSITION;\n\
\x20   float2 uv : TEXCOORD0;\n\
};\n\
\n\
Texture2D surface_texture : register(t0);\n\
SamplerState surface_sampler : register(s0);\n\
\n\
float4 psMain(MeshVsOut input) : SV_TARGET {\n\
\x20   return surface_texture.Sample(surface_sampler, input.uv);\n\
}\n\
";
