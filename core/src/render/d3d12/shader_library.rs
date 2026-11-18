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

/// The lit mesh's vertex shader (M5, HLSL 5.0): the world position and
/// the world-space normal join the outputs (the lighting's inputs).
pub const LIT_VERTEX_SHADER: &str = "\
struct MeshVsIn {\n\
\x20   float3 position : POSITION;\n\
\x20   float3 normal : NORMAL;\n\
\x20   float2 texcoord0 : TEXCOORD0;\n\
};\n\
\n\
struct MeshVsOut {\n\
\x20   float4 position : SV_POSITION;\n\
\x20   float3 normal : NORMAL0;\n\
\x20   float2 uv : TEXCOORD0;\n\
\x20   float3 world : WORLD0;\n\
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
\x20   output.normal = mul((float3x3)model, input.normal);\n\
\x20   output.uv = input.texcoord0;\n\
\x20   output.world = world.xyz;\n\
\x20   return output;\n\
}\n\
";

/// The lit mesh's pixel shader (M5, HLSL 5.0): the same Blinn-Phong
/// shading the other libraries carry (the sun plus up to four point
/// lights with the windowed inverse-square attenuation). The lights'
/// cbuffer (register(b1)) is the same byte layout the other backends
/// map; the 256-byte alignment the vkd3d requires holds (the block is
/// 256 bytes).
pub const LIT_PIXEL_SHADER: &str = "\
struct MeshVsOut {\n\
\x20   float4 position : SV_POSITION;\n\
\x20   float3 normal : NORMAL0;\n\
\x20   float2 uv : TEXCOORD0;\n\
\x20   float3 world : WORLD0;\n\
};\n\
\n\
Texture2D surface_texture : register(t0);\n\
SamplerState surface_sampler : register(s0);\n\
\n\
cbuffer Lights : register(b1) {\n\
\x20   float4 sun_direction;\n\
\x20   float4 sun_color_intensity;\n\
\x20   float4 point_position_radius[4];\n\
\x20   float4 point_color_intensity[4];\n\
\x20   float4 params;\n\
};\n\
\n\
float4 psMain(MeshVsOut input) : SV_TARGET {\n\
\x20   float3 normal = normalize(input.normal);\n\
\x20   float3 base = surface_texture.Sample(surface_sampler, input.uv).rgb;\n\
\x20   float3 sun_dir = normalize(-sun_direction.xyz);\n\
\x20   float sun_diffuse = max(dot(normal, sun_dir), 0.0);\n\
\x20   float3 colour = base * sun_color_intensity.rgb\n\
\x20       * sun_color_intensity.a * sun_diffuse;\n\
\x20   for (int i = 0; i < 4; i++) {\n\
\x20       if (float(i) >= params.x) {\n\
\x20           break;\n\
\x20       }\n\
\x20       float3 to_light = point_position_radius[i].xyz - input.world;\n\
\x20       float distance = length(to_light);\n\
\x20       to_light /= distance;\n\
\x20       float radius = point_position_radius[i].w;\n\
\x20       float window = 1.0 - (distance / radius) * (distance / radius);\n\
\x20       float attenuation = window * window / (1.0 + distance * distance);\n\
\x20       if (distance >= radius) {\n\
\x20           attenuation = 0.0;\n\
\x20       }\n\
\x20       float diffuse = max(dot(normal, to_light), 0.0);\n\
\x20       colour += base * point_color_intensity[i].rgb\n\
\x20           * point_color_intensity[i].a * attenuation * diffuse;\n\
\x20   }\n\
\x20   return float4(colour, 1.0);\n\
}\n\
";
