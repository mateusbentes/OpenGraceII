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
