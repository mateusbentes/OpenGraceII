//! The OpenGL backend's shader library. RULE (decision 44, the same rule
//! the Vulkan backend's shader library records): every engine and test
//! shader source lives ONLY in this file, under the OpenGL module. The
//! API-neutral render contract never contains shading-language source;
//! programs are addressed through the API-neutral program identity and
//! resolved here.
//!
//! The shaders are GLSL 330 core, compiled at runtime by the driver (the
//! OpenGL path needs no build-time SPIR-V step; the Vulkan path's
//! glslangValidator step stays exclusive to it).

/// The diagnostic triangle's vertex shader.
pub const TRIANGLE_VERTEX_SHADER: &str = r"
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 mvp;
void main()
{
    gl_Position = mvp * vec4(position, 1.0);
}
";

/// The diagnostic triangle's fragment shader.
pub const TRIANGLE_FRAGMENT_SHADER: &str = r"
#version 330 core
uniform vec4 color;
out vec4 fragColor;
void main()
{
    fragColor = color;
}
";

/// The generic mesh's vertex shader: the PNU vertex and the two camera
/// uniforms (the same names the backend's uniform locations fetch).
pub const MESH_VERTEX_SHADER: &str = r"
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
uniform mat4 view_projection;
uniform mat4 model;
out vec3 v_normal;
out vec2 v_uv;
void main()
{
    gl_Position = view_projection * model * vec4(position, 1.0);
    v_normal = normal;
    v_uv = uv;
}
";

/// The generic mesh's fragment shader: the bound texture's colour (the
/// sampler at unit 0 the backend's bind_texture drives).
pub const MESH_FRAGMENT_SHADER: &str = r"
#version 330 core
in vec3 v_normal;
in vec2 v_uv;
uniform sampler2D surface_texture;
out vec4 fragColor;
void main()
{
    fragColor = texture(surface_texture, v_uv);
}
";

/// The lit mesh's vertex shader (M5): the world position and the
/// world-space normal join the outputs (the lighting's inputs).
pub const LIT_VERTEX_SHADER: &str = r"
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
uniform mat4 view_projection;
uniform mat4 model;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_world;
void main()
{
    vec4 world = model * vec4(position, 1.0);
    gl_Position = view_projection * world;
    v_normal = mat3(model) * normal;
    v_uv = uv;
    v_world = world.xyz;
}
";

/// The lit mesh's fragment shader (M5): the same Blinn-Phong shading
/// the Vulkan library carries, in the GLSL 330 form (the sun plus up
/// to four point lights with the windowed inverse-square attenuation;
/// the uniforms the backend's upload path drives by name).
pub const LIT_FRAGMENT_SHADER: &str = r"
#version 330 core
in vec3 v_normal;
in vec2 v_uv;
in vec3 v_world;
uniform sampler2D surface_texture;
uniform vec4 sun_direction;
uniform vec4 sun_color_intensity;
uniform vec4 point_position_radius[4];
uniform vec4 point_color_intensity[4];
uniform vec4 params;
out vec4 fragColor;
void main()
{
    vec3 normal = normalize(v_normal);
    vec3 base = texture(surface_texture, v_uv).rgb;
    vec3 sun_dir = normalize(-sun_direction.xyz);
    float sun_diffuse = max(dot(normal, sun_dir), 0.0);
    vec3 colour = base * sun_color_intensity.rgb
        * sun_color_intensity.a * sun_diffuse;
    for (int i = 0; i < 4; i++) {
        if (float(i) >= params.x) {
            break;
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
    fragColor = vec4(colour, 1.0);
}
";
