//! The Vulkan backend's shader library. RULE (decision 44, the same rule
//! OpenGrace's shader library records): every engine and test shader source
//! lives ONLY in this file, under the Vulkan module. The API-neutral render
//! contract never contains shading-language source; programs are addressed
//! through the API-neutral program identity and resolved here.
//!
//! The shaders are GLSL 450 core with one std140 uniform block (set=0,
//! binding=0). The build script compiles them to SPIR-V with glslangValidator
//! (the build-time prerequisite, the same rule the OpenGL backend's build
//! recorded) and the generated `spirv.rs` carries the byte arrays.

/// The diagnostic triangle's vertex shader.
pub const TRIANGLE_VERTEX_SHADER: &str = r"
#version 450 core
layout(set = 0, binding = 0, std140) uniform Block
{
    mat4 mvp;
    vec4 color;
};
layout(location = 0) in vec3 position;
void main()
{
    gl_Position = mvp * vec4(position, 1.0);
}
";

/// The diagnostic triangle's fragment shader.
pub const TRIANGLE_FRAGMENT_SHADER: &str = r"
#version 450 core
layout(set = 0, binding = 0, std140) uniform Block
{
    mat4 mvp;
    vec4 color;
};
layout(location = 0) out vec4 outColor;
void main()
{
    outColor = color;
}
";

/// The generic mesh's vertex shader: the PNU vertex and the std140
/// camera block (the view-projection matrix the backend's set_camera
/// builds and the model matrix the submission writes — the same two
/// matrices the other backends' blocks carry; the M4c rule).
pub const CUBE_VERTEX_SHADER: &str = r"
#version 450 core
layout(set = 0, binding = 0, std140) uniform Camera
{
    mat4 view_projection;
    mat4 model;
};
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_uv;
void main()
{
    gl_Position = view_projection * model * vec4(position, 1.0);
    v_normal = normal;
    v_uv = uv;
}
";

/// The generic mesh's fragment shader: the bound texture times the
/// surface tint (the material path's minimum).
pub const CUBE_FRAGMENT_SHADER: &str = r"
#version 450 core
layout(set = 0, binding = 1) uniform sampler2D surface_texture;
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_uv;
layout(location = 0) out vec4 outColor;
void main()
{
    outColor = texture(surface_texture, v_uv);
}
";

/// The lit mesh's vertex shader (M5): the PNU vertex plus the world
/// position the lighting needs (the interpolated world-space position
/// the fragment stage's lighting equations consume).
pub const LIT_VERTEX_SHADER: &str = r"
#version 450 core
layout(set = 0, binding = 0, std140) uniform Camera
{
    mat4 view_projection;
    mat4 model;
};
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_uv;
layout(location = 2) out vec3 v_world;
void main()
{
    vec4 world = model * vec4(position, 1.0);
    gl_Position = view_projection * world;
    v_normal = mat3(model) * normal;
    v_uv = uv;
    v_world = world.xyz;
}
";

/// The lit mesh's fragment shader (M5): the Blinn-Phong shading the
/// lighting module's reference implementation carries — the sun's
/// directional contribution plus up to four point lights with the
/// windowed inverse-square attenuation. The lights' std140 block (set 0,
/// binding 2) is the same byte layout the other backends map.
pub const LIT_FRAGMENT_SHADER: &str = r"
#version 450 core
layout(set = 0, binding = 1) uniform sampler2D surface_texture;
layout(set = 0, binding = 2, std140) uniform Lights
{
    vec4 sun_direction;
    vec4 sun_color_intensity;
    vec4 point_position_radius[4];
    vec4 point_color_intensity[4];
    vec4 params;
};
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in vec3 v_world;
layout(location = 0) out vec4 outColor;
void main()
{
    vec3 normal = normalize(v_normal);
    vec3 base = texture(surface_texture, v_uv).rgb;
    // The sun: the direction the light travels in is the negated
    // to-light vector.
    vec3 sun_dir = normalize(-sun_direction.xyz);
    float sun_diffuse = max(dot(normal, sun_dir), 0.0);
    vec3 colour = base * sun_color_intensity.rgb
        * sun_color_intensity.a * sun_diffuse;
    // The point lights: the windowed inverse-square attenuation. The
    // four slots unroll (the loop-with-break form crashed the software
    // rasterizer's compiler in the debug run).
    for (int i = 0; i < 4; i++) {
        if (float(i) >= params.x) {
            continue;
        }
        vec3 to_light = point_position_radius[i].xyz - v_world;
        float distance = length(to_light);
        to_light /= distance;
        float radius = point_position_radius[i].w;
        float window = 1.0 - (distance / radius) * (distance / radius);
        float attenuation = window * window / (1.0 + distance * distance);
        if (distance >= radius) {
            attenuation = 0.0;
        }
        float diffuse = max(dot(normal, to_light), 0.0);
        colour += base * point_color_intensity[i].rgb
            * point_color_intensity[i].a * attenuation * diffuse;
    }
    outColor = vec4(colour, 1.0);
}
";

/// The deferred geometry pass's vertex shader (M6): the same stage the
/// lit mesh's vertex shader carries (the world position, the normal and
/// the texture coordinates feed the G-buffer's three attachments).
pub const GBUFFER_VERTEX_SHADER: &str = r"
#version 450 core
layout(set = 0, binding = 0, std140) uniform Camera
{
    mat4 view_projection;
    mat4 model;
};
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_uv;
layout(location = 2) out vec3 v_world;
void main()
{
    vec4 world = model * vec4(position, 1.0);
    gl_Position = view_projection * world;
    v_normal = normalize(mat3(model) * normal);
    v_uv = uv;
    v_world = world.xyz;
}
";

/// The deferred geometry pass's fragment shader (M6): the G-buffer's
/// three attachments — the albedo (the alpha 0 marks geometry), the
/// half-offset normal and the world position. The 16-bit float
/// attachment stores the world position's xyz (the alpha word stays
/// unused).
pub const GBUFFER_FRAGMENT_SHADER: &str = r"
#version 450 core
layout(set = 0, binding = 1) uniform sampler2D surface_texture;
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in vec3 v_world;
layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outWorld;
void main()
{
    vec3 base = texture(surface_texture, v_uv).rgb;
    outAlbedo = vec4(base, 0.0);
    outNormal = vec4(normalize(v_normal) * 0.5 + 0.5, 1.0);
    outWorld = vec4(v_world, 1.0);
}
";

/// The deferred lighting pass's vertex shader (M6): the full-frame
/// triangle the vertex index generates (the three corners cover the
/// viewport; the no-vertex-buffer draw draws it).
pub const LIGHTING_VERTEX_SHADER: &str = r"
#version 450 core
layout(location = 0) out vec2 v_uv;
void main()
{
    vec2 corner = vec2(
        float((gl_VertexIndex & 1) << 2) - 1.0,
        float((gl_VertexIndex & 2) << 1) - 1.0);
    v_uv = corner * 0.5 + 0.5;
    gl_Position = vec4(corner, 0.0, 1.0);
}
";

/// The deferred lighting pass's fragment shader (M6): the G-buffer's
/// three attachments sampled, the same Blinn-Phong reference the
/// forward path carries (the sun plus the point lights), the
/// background pixels copied from the albedo's alpha rule (0 = geometry,
/// anything else = the clear colour the G-buffer carries).
pub const LIGHTING_FRAGMENT_SHADER: &str = r"
#version 450 core
layout(set = 0, binding = 0, std140) uniform Lights
{
    vec4 sun_direction;
    vec4 sun_color_intensity;
    vec4 point_position_radius[4];
    vec4 point_color_intensity[4];
    vec4 params;
};
layout(set = 0, binding = 1) uniform sampler2D albedo_texture;
layout(set = 0, binding = 2) uniform sampler2D normal_texture;
layout(set = 0, binding = 3) uniform sampler2D world_texture;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 outColor;
void main()
{
    vec4 albedo = texture(albedo_texture, v_uv);
    if (albedo.a != 0.0) {
        // The background: the G-buffer's clear colour passes through.
        outColor = albedo;
        return;
    }
    vec3 normal = texture(normal_texture, v_uv).rgb * 2.0 - 1.0;
    vec3 world = texture(world_texture, v_uv).rgb;
    vec3 base = albedo.rgb;
    vec3 sun_dir = normalize(-sun_direction.xyz);
    float sun_diffuse = max(dot(normal, sun_dir), 0.0);
    vec3 colour = base * sun_color_intensity.rgb
        * sun_color_intensity.a * sun_diffuse;
    // The four slots unroll (the loop-with-break form crashed the
    // software rasterizer's compiler in the debug run).
    for (int i = 0; i < 4; i++) {
        if (float(i) >= params.x) {
            continue;
        }
        vec3 to_light = point_position_radius[i].xyz - world;
        float dist = length(to_light);
        to_light /= dist;
        float radius = point_position_radius[i].w;
        float window = 1.0 - (dist / radius) * (dist / radius);
        float attenuation = window * window / (1.0 + dist * dist);
        if (dist >= radius) {
            attenuation = 0.0;
        }
        float diffuse = max(dot(normal, to_light), 0.0);
        colour += base * point_color_intensity[i].rgb
            * point_color_intensity[i].a * attenuation * diffuse;
    }
    outColor = vec4(colour, 1.0);
}
";

/// The SPIR-V for the diagnostic triangle's shaders, generated by the
/// build script. The generated file is the build's output; the engine
/// never links a third-party Vulkan binding (lesson 3).
pub mod spirv {
    // The generated path is the build's normal output; the committed
    // path is the fallback for the machines without glslangValidator
    // (the macOS and the iOS builds where the Vulkan backend only
    // compiles and never runs). The build script picks the path with
    // the spirv_generated and the spirv_fallback cfgs.
    #[cfg(feature = "vulkan")]
    #[cfg(not(spirv_fallback))]
    include!(concat!(env!("OUT_DIR"), "/vulkan_spirv.rs"));

    #[cfg(feature = "vulkan")]
    #[cfg(spirv_fallback)]
    include!("vulkan_spirv_committed.rs");
}
