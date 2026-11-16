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
