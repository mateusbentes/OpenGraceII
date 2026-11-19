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
/// The deferred geometry pass's vertex shader (M6, HLSL 5.0): the same
/// transform the lit path's vertex shader carries, writing the world
/// position and the world-space normal for the G-buffer's attachments.
pub const GBUFFER_VERTEX_SHADER: &str = "\
struct MeshVsIn {\n\
\x20   float3 position : POSITION;\n\
\x20   float3 normal : NORMAL;\n\
\x20   float2 texcoord0 : TEXCOORD0;\n\
};\n\
\n\
struct GbufferVsOut {\n\
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
GbufferVsOut vsMain(MeshVsIn input) {\n\
\x20   GbufferVsOut output;\n\
\x20   float4 world = mul(model, float4(input.position, 1.0));\n\
\x20   output.position = mul(view_projection, world);\n\
\x20   output.normal = mul((float3x3)model, input.normal);\n\
\x20   output.uv = input.texcoord0;\n\
\x20   output.world = world.xyz;\n\
\x20   return output;\n\
}\n\
";
/// The deferred geometry pass's pixel shader (M6, HLSL 5.0): the three
/// G-buffer writes (the albedo with its alpha at 0.0 marking geometry,
/// the 0.5-offset normal, the world position).
pub const GBUFFER_PIXEL_SHADER: &str = "\
struct GbufferVsOut {\n\
\x20   float4 position : SV_POSITION;\n\
\x20   float3 normal : NORMAL0;\n\
\x20   float2 uv : TEXCOORD0;\n\
\x20   float3 world : WORLD0;\n\
};\n\
\n\
Texture2D surface_texture : register(t0);\n\
SamplerState surface_sampler : register(s0);\n\
\n\
struct GbufferOut {\n\
\x20   float4 albedo : SV_TARGET0;\n\
\x20   float4 normal : SV_TARGET1;\n\
\x20   float4 world : SV_TARGET2;\n\
};\n\
\n\
GbufferOut psMain(GbufferVsOut input) {\n\
\x20   GbufferOut output;\n\
\x20   float4 base = surface_texture.Sample(surface_sampler, input.uv);\n\
\x20   output.albedo = float4(base.rgb, 0.0);\n\
\x20   output.normal = float4(normalize(input.normal) * 0.5 + 0.5, 1.0);\n\
\x20   output.world = float4(input.world, 1.0);\n\
\x20   return output;\n\
}\n\
";
/// The deferred lighting pass's vertex shader (M6, HLSL 5.0): the
/// fullscreen triangle from SV_VertexID (no vertex buffer, the same
/// shape the Vulkan and the OpenGL lighting vertex shaders carry).
pub const LIGHTING_VERTEX_SHADER: &str = "\
struct LightingVsOut {\n\
\x20   float4 position : SV_POSITION;\n\
\x20   float2 uv : TEXCOORD0;\n\
};\n\
\n\
LightingVsOut vsMain(uint vertex_id : SV_VertexID) {\n\
\x20   LightingVsOut output;\n\
\x20   float2 corner = float2(float((vertex_id & 1) << 2) - 1.0,\n\
\x20                          float((vertex_id & 2) << 1) - 1.0);\n\
\x20   output.uv = corner * 0.5 + 0.5;\n\
\x20   output.position = float4(corner, 0.0, 1.0);\n\
\x20   return output;\n\
}\n\
";
/// The deferred lighting pass's pixel shader (M6, HLSL 5.0): the same
/// Blinn-Phong shading the forward path carries, sampling the G-buffer's
/// three attachments (the background pixels copy through the albedo's
/// alpha rule).
pub const LIGHTING_PIXEL_SHADER: &str = "\
struct LightingVsOut {\n\
\x20   float4 position : SV_POSITION;\n\
\x20   float2 uv : TEXCOORD0;\n\
};\n\
\n\
Texture2D gbuffer_albedo : register(t0);\n\
Texture2D gbuffer_normal : register(t1);\n\
Texture2D gbuffer_world : register(t2);\n\
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
float4 psMain(LightingVsOut input) : SV_TARGET {\n\
\x20   float4 albedo = gbuffer_albedo.Sample(surface_sampler, input.uv);\n\
\x20   if (albedo.a != 0.0) {\n\
\x20       return albedo;\n\
\x20   }\n\
\x20   float3 normal = normalize(gbuffer_normal.Sample(surface_sampler, input.uv).rgb * 2.0 - 1.0);\n\
\x20   float3 world = gbuffer_world.Sample(surface_sampler, input.uv).rgb;\n\
\x20   float3 base = albedo.rgb;\n\
\x20   float3 sun_dir = normalize(-sun_direction.xyz);\n\
\x20   float sun_diffuse = max(dot(normal, sun_dir), 0.0);\n\
\x20   float3 colour = base * sun_color_intensity.rgb\n\
\x20       * sun_color_intensity.a * sun_diffuse;\n\
\x20   for (int i = 0; i < 4; i++) {\n\
\x20       if (float(i) >= params.x) {\n\
\x20           break;\n\
\x20       }\n\
\x20       float3 to_light = point_position_radius[i].xyz - world;\n\
\x20       float dist = length(to_light);\n\
\x20       to_light /= dist;\n\
\x20       float radius = point_position_radius[i].w;\n\
\x20       float window = 1.0 - (dist / radius) * (dist / radius);\n\
\x20       float attenuation = window * window / (1.0 + dist * dist);\n\
\x20       if (dist >= radius) {\n\
\x20           attenuation = 0.0;\n\
\x20       }\n\
\x20       float diffuse = max(dot(normal, to_light), 0.0);\n\
\x20       colour += base * point_color_intensity[i].rgb\n\
\x20           * point_color_intensity[i].a * attenuation * diffuse;\n\
\x20   }\n\
\x20   return float4(colour, 1.0);\n\
}\n\
";
/// The post chain's geometry pass's pixel shader (M7, HLSL 5.0): the two
/// HDR writes — the full Blinn-Phong shading's linear radiance
/// (attachment 0) and the fragment's world position (attachment 1). The
/// same shading the forward path's shader carries.
pub const POST_GBUFFER_PIXEL_SHADER: &str = "\
struct GbufferVsOut {\n\
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
struct PostGbufferOut {\n\
\x20   float4 radiance : SV_TARGET0;\n\
\x20   float4 world : SV_TARGET1;\n\
};\n\
\n\
PostGbufferOut psMain(GbufferVsOut input) {\n\
\x20   PostGbufferOut output;\n\
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
\x20       float dist = length(to_light);\n\
\x20       to_light /= dist;\n\
\x20       float radius = point_position_radius[i].w;\n\
\x20       float window = 1.0 - (dist / radius) * (dist / radius);\n\
\x20       float attenuation = window * window / (1.0 + dist * dist);\n\
\x20       if (dist >= radius) {\n\
\x20           attenuation = 0.0;\n\
\x20       }\n\
\x20       float diffuse = max(dot(normal, to_light), 0.0);\n\
\x20       colour += base * point_color_intensity[i].rgb\n\
\x20           * point_color_intensity[i].a * attenuation * diffuse;\n\
\x20   }\n\
\x20   output.radiance = float4(colour, 1.0);\n\
\x20   output.world = float4(input.world, 1.0);\n\
\x20   return output;\n\
}\n\
";
/// The post chain's composite pass's pixel shader (M7, HLSL 5.0): the
/// two HDR attachments sampled and the enabled effects applied in the
/// fixed order — the SSAO's occlusion darkens the radiance, the bloom's
/// threshold-and-blur lifts the bright regions, the depth of field's
/// focus share blends the blurred radiance, and the filmic tonemapping
/// maps the result onto the display's range; the morphological
/// anti-aliasing's edge blend follows the tonemap (the LDR image the
/// edge detection reads). The config block carries the toggles and the
/// parameters (the same std140 shape the CPU reference's
/// PostChainConfig values map onto).
pub const POST_COMPOSITE_PIXEL_SHADER: &str = "\
struct LightingVsOut {\n\
\x20   float4 position : SV_POSITION;\n\
\x20   float2 uv : TEXCOORD0;\n\
};\n\
\n\
Texture2D radiance_texture : register(t0);\n\
Texture2D world_texture : register(t1);\n\
SamplerState surface_sampler : register(s0);\n\
\n\
cbuffer Config : register(b0) {\n\
\x20   float4 params0; // tonemapping, bloom, dof, ssao (the 1.0/0.0 flags)\n\
\x20   float4 params1; // bloom_threshold, bloom_knee, bloom_intensity, ssao_radius\n\
\x20   float4 params2; // focal_distance, focus_near_range, focus_far_range, anti_aliasing\n\
};\n\
\n\
float3 sample_blurred(Texture2D tex, float2 uv, float2 texel)\n\
{\n\
\x20   // The 5-tap gaussian: the centre plus the two symmetric pairs at\n\
\x20   // the 1.5-texel and 3.5-texel offsets (the weights the CPU\n\
\x20   // reference's GAUSSIAN_WEIGHTS carry).\n\
\x20   float3 sum = tex.Sample(surface_sampler, uv).rgb * 0.2887;\n\
\x20   sum += tex.Sample(surface_sampler, uv + float2(1.5, 0.0) * texel).rgb * 0.2331;\n\
\x20   sum += tex.Sample(surface_sampler, uv - float2(1.5, 0.0) * texel).rgb * 0.2331;\n\
\x20   sum += tex.Sample(surface_sampler, uv + float2(3.5, 0.0) * texel).rgb * 0.1225;\n\
\x20   sum += tex.Sample(surface_sampler, uv - float2(3.5, 0.0) * texel).rgb * 0.1225;\n\
\x20   sum += tex.Sample(surface_sampler, uv + float2(0.0, 1.5) * texel).rgb * 0.2331;\n\
\x20   sum += tex.Sample(surface_sampler, uv - float2(0.0, 1.5) * texel).rgb * 0.2331;\n\
\x20   sum += tex.Sample(surface_sampler, uv + float2(0.0, 3.5) * texel).rgb * 0.1225;\n\
\x20   sum += tex.Sample(surface_sampler, uv - float2(0.0, 3.5) * texel).rgb * 0.1225;\n\
\x20   return sum;\n\
}\n\
float bloom_weight(float radiance, float threshold, float knee)\n\
{\n\
\x20   if (radiance <= threshold) {\n\
\x20       return 0.0;\n\
\x20   }\n\
\x20   float excess = radiance - threshold;\n\
\x20   float soft = min(excess / max(knee, 1e-6), 1.0);\n\
\x20   float eased = soft * soft * (3.0 - 2.0 * soft);\n\
\x20   if (excess >= knee) {\n\
\x20       return excess;\n\
\x20   }\n\
\x20   return eased * excess;\n\
}\n\
float focus_share(float distance, float focal, float near_range, float far_range)\n\
{\n\
\x20   float delta = distance - focal;\n\
\x20   float range = delta < 0.0 ? near_range : far_range;\n\
\x20   return clamp(1.0 - abs(delta) / max(range, 1e-6), 0.0, 1.0);\n\
}\n\
float ssao_sample(float height, float dist, float radius)\n\
{\n\
\x20   if (height <= 0.0 || dist <= 0.0 || dist >= radius) {\n\
\x20       return 0.0;\n\
\x20   }\n\
\x20   float horizon = height / dist;\n\
\x20   float falloff = 1.0 - (dist / radius) * (dist / radius);\n\
\x20   return horizon * falloff;\n\
}\n\
float mlaa_crossing(float centre, float neighbour)\n\
{\n\
\x20   // The midpoint rule: an edge that cuts the pixel (the centre's\n\
\x20   // luminance differs from the neighbour's) blends half of each\n\
\x20   // side; aligned edges keep the pixel's own colour.\n\
\x20   if (abs(centre - neighbour) > 1e-6) {\n\
\x20       return 0.5;\n\
\x20   }\n\
\x20   return 0.0;\n\
}\n\
float mlaa_blend(float crossing)\n\
{\n\
\x20   return clamp(crossing, 0.0, 1.0);\n\
}\n\
float tonemap_channel(float c)\n\
{\n\
\x20   float num = c * (2.51 * c + 0.03);\n\
\x20   float den = c * (2.43 * c + 0.59) + 0.14;\n\
\x20   return clamp(num / den, 0.0, 1.0);\n\
}\n\
\n\
float4 psMain(LightingVsOut input) : SV_TARGET {\n\
\x20   float4 radiance_sample = radiance_texture.Sample(surface_sampler, input.uv);\n\
\x20   float4 world_sample = world_texture.Sample(surface_sampler, input.uv);\n\
\x20   float3 colour = radiance_sample.rgb;\n\
\x20   bool background = world_sample.a == 0.0;\n\
\x20   float texel_x = 1.0 / 320.0;\n\
\x20   float texel_y = 1.0 / 240.0;\n\
\x20   // The SSAO: the four hemisphere samples around the fragment's world\n\
\x20   // position (the +x, -x, +y, -y offsets at the radius's eighth), the\n\
\x20   // occlusion the world buffer's own heights against the fragment's\n\
\x20   // plane carry.\n\
\x20   if (params0.w > 0.5 && !background) {\n\
\x20       float3 world = world_sample.rgb;\n\
\x20       float occlusion = 0.0;\n\
\x20       float radius = params1.w;\n\
\x20       for (int i = 0; i < 4; i++) {\n\
\x20           float2 offset = float2(0.0, 0.0);\n\
\x20           offset.x = (i == 0) ? radius * 0.125 : (i == 1) ? -radius * 0.125 : 0.0;\n\
\x20           offset.y = (i == 2) ? radius * 0.125 : (i == 3) ? -radius * 0.125 : 0.0;\n\
\x20           float3 sample_world = world_texture.Sample(surface_sampler,\n\
\x20               input.uv + float2(offset.x * texel_x, offset.y * texel_y)).rgb;\n\
\x20           float3 to_sample = sample_world - world;\n\
\x20           float dist = length(to_sample);\n\
\x20           float height = to_sample.y;\n\
\x20           occlusion += ssao_sample(height, dist, radius);\n\
\x20       }\n\
\x20       colour *= 1.0 - clamp(occlusion * 0.25, 0.0, 1.0) * 0.5;\n\
\x20   }\n\
\x20   // The bloom: the blurred radiance's weight above the threshold\n\
\x20   // lifts the frame's bright regions (the composite's intensity the\n\
\x20   // params carry scales it).\n\
\x20   if (params0.y > 0.5) {\n\
\x20       float3 blurred = sample_blurred(radiance_texture, input.uv,\n\
\x20           float2(texel_x, texel_y));\n\
\x20       float weight = bloom_weight(max(blurred.r, max(blurred.g, blurred.b)),\n\
\x20           params1.x, params1.y);\n\
\x20       colour += blurred * weight * params1.z;\n\
\x20   }\n\
\x20   // The depth of field: the focus share blends the blurred radiance\n\
\x20   // over the sharp one (the background fragments stay untouched - the\n\
\x20   // chain blurs the scene's own geometry).\n\
\x20   if (params0.z > 0.5 && !background) {\n\
\x20       float3 blurred = sample_blurred(radiance_texture, input.uv,\n\
\x20           float2(texel_x, texel_y));\n\
\x20       float distance = length(world_sample.rgb - float3(0.0, 0.0, 0.0));\n\
\x20       float share = 1.0 - focus_share(distance, params2.x, params2.y, params2.z);\n\
\x20       colour = lerp(colour, blurred, clamp(share, 0.0, 1.0));\n\
\x20   }\n\
\x20   // The tonemapping: the filmic curve maps the linear radiance onto\n\
\x20   // the display's [0, 1] (the gamma the framebuffer's 8-bit encoding\n\
\x20   // assumes follows).\n\
\x20   if (params0.x > 0.5) {\n\
\x20       float num = colour.r * (2.51 * colour.r + 0.03);\n\
\x20       float den = colour.r * (2.43 * colour.r + 0.59) + 0.14;\n\
\x20       float mapped_r = clamp(num / den, 0.0, 1.0);\n\
\x20       num = colour.g * (2.51 * colour.g + 0.03);\n\
\x20       den = colour.g * (2.43 * colour.g + 0.59) + 0.14;\n\
\x20       float mapped_g = clamp(num / den, 0.0, 1.0);\n\
\x20       num = colour.b * (2.51 * colour.b + 0.03);\n\
\x20       den = colour.b * (2.43 * colour.b + 0.59) + 0.14;\n\
\x20       float mapped_b = clamp(num / den, 0.0, 1.0);\n\
\x20       colour = float3(mapped_r, mapped_g, mapped_b);\n\
\x20       colour = pow(colour, 1.0 / 2.2);\n\
\x20   }\n\
\x20   // The morphological anti-aliasing: the tonemapped luminance's edge\n\
\x20   // detection reads the four direct neighbours, the crossing ratio\n\
\x20   // against the dominant edge neighbour drives the blend (the single-\n\
\x20   // pass approximation of the classic three-pass morphological\n\
\x20   // operator - the deterministic edge test the example carries).\n\
\x20   if (params2.w > 0.5) {\n\
\x20       float centre = dot(colour, float3(0.299, 0.587, 0.114));\n\
\x20       float3 t0 = radiance_texture.Sample(surface_sampler,\n\
\x20           input.uv + float2(-texel_x, 0.0)).rgb;\n\
\x20       float3 t1 = radiance_texture.Sample(surface_sampler,\n\
\x20           input.uv + float2(texel_x, 0.0)).rgb;\n\
\x20       float3 t2 = radiance_texture.Sample(surface_sampler,\n\
\x20           input.uv + float2(0.0, texel_y)).rgb;\n\
\x20       float3 t3 = radiance_texture.Sample(surface_sampler,\n\
\x20           input.uv + float2(0.0, -texel_y)).rgb;\n\
\x20       if (params0.x > 0.5) {\n\
\x20           t0 = pow(float3(tonemap_channel(t0.r), tonemap_channel(t0.g),\n\
\x20               tonemap_channel(t0.b)), 1.0 / 2.2);\n\
\x20           t1 = pow(float3(tonemap_channel(t1.r), tonemap_channel(t1.g),\n\
\x20               tonemap_channel(t1.b)), 1.0 / 2.2);\n\
\x20           t2 = pow(float3(tonemap_channel(t2.r), tonemap_channel(t2.g),\n\
\x20               tonemap_channel(t2.b)), 1.0 / 2.2);\n\
\x20           t3 = pow(float3(tonemap_channel(t3.r), tonemap_channel(t3.g),\n\
\x20               tonemap_channel(t3.b)), 1.0 / 2.2);\n\
\x20       }\n\
\x20       float l0 = dot(t0, float3(0.299, 0.587, 0.114));\n\
\x20       float l1 = dot(t1, float3(0.299, 0.587, 0.114));\n\
\x20       float l2 = dot(t2, float3(0.299, 0.587, 0.114));\n\
\x20       float l3 = dot(t3, float3(0.299, 0.587, 0.114));\n\
\x20       float dominant_crossing = mlaa_crossing(centre, l0);\n\
\x20       float3 dominant_colour = t0;\n\
\x20       float crossing = mlaa_crossing(centre, l1);\n\
\x20       if (crossing > dominant_crossing) {\n\
\x20           dominant_crossing = crossing;\n\
\x20           dominant_colour = t1;\n\
\x20       }\n\
\x20       crossing = mlaa_crossing(centre, l2);\n\
\x20       if (crossing > dominant_crossing) {\n\
\x20           dominant_crossing = crossing;\n\
\x20           dominant_colour = t2;\n\
\x20       }\n\
\x20       crossing = mlaa_crossing(centre, l3);\n\
\x20       if (crossing > dominant_crossing) {\n\
\x20           dominant_crossing = crossing;\n\
\x20           dominant_colour = t3;\n\
\x20       }\n\
\x20       colour = lerp(colour, dominant_colour,\n\
\x20           mlaa_blend(dominant_crossing));\n\
\x20   }\n\
\x20   return float4(colour, 1.0);\n\
}\n\
";
