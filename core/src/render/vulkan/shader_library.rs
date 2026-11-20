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

/// The post chain's geometry pass's fragment shader (M7): the two HDR
/// attachments — the full Blinn-Phong shading's linear radiance
/// (attachment 0) and the fragment's world position (attachment 1, the
/// distances the SSAO and the depth of field's focus share read). The
/// background fragments write the clear colour's radiance and a zero
/// world position (the alpha 0 marks background for the composite).
pub const POST_GBUFFER_FRAGMENT_SHADER: &str = r"
#version 450 core
layout(set = 0, binding = 1, std140) uniform Lights
{
    vec4 sun_direction;
    vec4 sun_color_intensity;
    vec4 point_position_radius[4];
    vec4 point_color_intensity[4];
    vec4 params;
};
layout(set = 0, binding = 2) uniform sampler2D surface_texture;
layout(set = 0, binding = 3, std140) uniform Volume
{
    vec4 volume_origin; // xyz + pad
    vec4 volume_dims; // x, y, z * 6 (the depth layers), pad
    vec4 volume_params; // x = the cell size, y = the indirect intensity, z = the toggle
};
layout(set = 0, binding = 4) uniform sampler3D volume_texture;
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in vec3 v_world;
layout(location = 0) out vec4 outRadiance;
layout(location = 1) out vec4 outWorld;
// The light volume's face selection: the face the normal faces (the
// same rule the CPU reference's face_for_normal carries — the six
// axis directions' largest dot product).
int volume_face(vec3 normal)
{
    vec3 directions[6];
    directions[0] = vec3(1.0, 0.0, 0.0);
    directions[1] = vec3(-1.0, 0.0, 0.0);
    directions[2] = vec3(0.0, 1.0, 0.0);
    directions[3] = vec3(0.0, -1.0, 0.0);
    directions[4] = vec3(0.0, 0.0, 1.0);
    directions[5] = vec3(0.0, 0.0, -1.0);
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
// The light volume's trilinear sample: the face the normal faces'
// radiance, interpolated between the 8 surrounding cells. The fetches
// land on the texels' centres (the sampler's NEAREST keeps each fetch
// exact) and the weights the CPU reference's sample carries; the
// position clamps into the grid (a position outside returns the
// border cell's radiance).
vec3 volume_sample(vec3 position, vec3 normal)
{
    int face = volume_face(normal);
    int cells_z = int(volume_dims.z / 6.0);
    vec3 continuous = (position - volume_origin.xyz) / volume_params.x - 0.5;
    ivec3 base = ivec3(0);
    vec3 frac = vec3(0.0);
    for (int axis = 0; axis < 3; axis++) {
        if (continuous[axis] <= 0.0) {
            base[axis] = 0;
            frac[axis] = 0.0;
        } else if (continuous[axis] >= (axis == 0 ? volume_dims.x :
            (axis == 1 ? volume_dims.y : cells_z)) - 1.0) {
            base[axis] = int(axis == 0 ? volume_dims.x :
                (axis == 1 ? volume_dims.y : cells_z)) - 1;
            frac[axis] = 0.0;
        } else {
            base[axis] = int(floor(continuous[axis]));
            frac[axis] = continuous[axis] - floor(continuous[axis]);
        }
    }
    vec3 result = vec3(0.0);
    for (int corner = 0; corner < 8; corner++) {
        int cx = base.x + ((corner & 1) != 0 ? 1 : 0);
        int cy = base.y + ((corner & 2) != 0 ? 1 : 0);
        int cz = base.z + ((corner & 4) != 0 ? 1 : 0);
        float wx = cx >= int(volume_dims.x) ? 0.0
            : ((corner & 1) != 0 ? frac.x : 1.0 - frac.x);
        float wy = cy >= int(volume_dims.y) ? 0.0
            : ((corner & 2) != 0 ? frac.y : 1.0 - frac.y);
        float wz = cz >= cells_z ? 0.0
            : ((corner & 4) != 0 ? frac.z : 1.0 - frac.z);
        float weight = wx * wy * wz;
        if (weight == 0.0) {
            continue;
        }
        int clamped_x = min(cx, int(volume_dims.x) - 1);
        int clamped_y = min(cy, int(volume_dims.y) - 1);
        int clamped_z = min(cz, cells_z - 1);
        vec3 texel = vec3(
            float(clamped_x) + 0.5,
            float(clamped_y) + 0.5,
            float(clamped_z * 6 + face) + 0.5
        ) / vec3(volume_dims.x, volume_dims.y, volume_dims.z);
        // The explicit LOD (the same rule the GLSL's textureLod carries):
        // the loop's control flow depends on the fragment's position (the
        // non-uniform branches), so the implicit-derivative sampling would
        // be undefined behaviour — the explicit level keeps the fetch
        // defined.
        result += weight * textureLod(volume_texture, texel, 0.0).rgb;
    }
    return result;
}
void main()
{
    vec3 base = texture(surface_texture, v_uv).rgb;
    vec3 normal = normalize(v_normal);
    vec3 sun_dir = normalize(-sun_direction.xyz);
    float sun_diffuse = max(dot(normal, sun_dir), 0.0);
    vec3 colour = base * sun_color_intensity.rgb
        * sun_color_intensity.a * sun_diffuse;
    for (int i = 0; i < 4; i++) {
        if (float(i) >= params.x) {
            continue;
        }
        vec3 to_light = point_position_radius[i].xyz - v_world;
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
    // The indirect light's term (M8): the volume's radiance shades the
    // surface's albedo (the same `albedo * irradiance` rule the direct
    // lighting's diffuse carries) — the face the normal faces'
    // radiance, scaled by the config's indirect intensity.
    if (volume_params.z > 0.5) {
        colour += base * volume_sample(v_world, normal) * volume_params.y;
    }
    outRadiance = vec4(colour, 1.0);
    outWorld = vec4(v_world, 1.0);
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
#version 450 core
layout(set = 0, binding = 0, std140) uniform Config
{
    vec4 params0; // tonemapping, bloom, dof, ssao (the 1.0/0.0 flags)
    vec4 params1; // bloom_threshold, bloom_knee, bloom_intensity, ssao_radius
    vec4 params2; // focal_distance, focus_near_range, focus_far_range, anti_aliasing
};
layout(set = 0, binding = 1) uniform sampler2D radiance_texture;
layout(set = 0, binding = 2) uniform sampler2D world_texture;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 outColor;
vec3 sample_blurred(sampler2D tex, vec2 uv, vec2 texel)
{
    // The 5-tap gaussian: the centre plus the two symmetric pairs at
    // the 1.5-texel and 3.5-texel offsets (the weights the CPU
    // reference's GAUSSIAN_WEIGHTS carry).
    vec3 sum = texture(tex, uv).rgb * 0.2887;
    sum += texture(tex, uv + vec2(1.5, 0.0) * texel).rgb * 0.2331;
    sum += texture(tex, uv - vec2(1.5, 0.0) * texel).rgb * 0.2331;
    sum += texture(tex, uv + vec2(3.5, 0.0) * texel).rgb * 0.1225;
    sum += texture(tex, uv - vec2(3.5, 0.0) * texel).rgb * 0.1225;
    sum += texture(tex, uv + vec2(0.0, 1.5) * texel).rgb * 0.2331;
    sum += texture(tex, uv - vec2(0.0, 1.5) * texel).rgb * 0.2331;
    sum += texture(tex, uv + vec2(0.0, 3.5) * texel).rgb * 0.1225;
    sum += texture(tex, uv - vec2(0.0, 3.5) * texel).rgb * 0.1225;
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
    // luminance differs from the neighbour's) blends half of each
    // side; aligned edges keep the pixel's own colour.
    if (abs(centre - neighbour) > 1e-6) {
        return 0.5;
    }
    return 0.0;
}
float mlaa_blend(float crossing)
{
    return clamp(crossing, 0.0, 1.0);
}
void main()
{
    vec4 radiance_sample = texture(radiance_texture, v_uv);
    vec4 world_sample = texture(world_texture, v_uv);
    vec3 colour = radiance_sample.rgb;
    bool background = world_sample.a == 0.0;
    float texel_x = 1.0 / 320.0;
    float texel_y = 1.0 / 240.0;
    // The SSAO: the four hemisphere samples around the fragment's world
    // position (the +x, -x, +y, -y offsets at the radius's eighth), the
    // occlusion the world buffer's own heights against the fragment's
    // plane carry.
    if (params0.w > 0.5 && !background) {
        vec3 world = world_sample.rgb;
        float occlusion = 0.0;
        float radius = params1.w;
        for (int i = 0; i < 4; i++) {
            vec2 offset = vec2(0.0);
            offset.x = (i == 0) ? radius * 0.125 : (i == 1) ? -radius * 0.125 : 0.0;
            offset.y = (i == 2) ? radius * 0.125 : (i == 3) ? -radius * 0.125 : 0.0;
            vec3 sample_world = texture(world_texture,
                v_uv + vec2(offset.x * texel_x, offset.y * texel_y)).rgb;
            vec3 to_sample = sample_world - world;
            float dist = length(to_sample);
            float height = to_sample.y;
            occlusion += ssao_sample(height, dist, radius);
        }
        colour *= 1.0 - clamp(occlusion * 0.25, 0.0, 1.0) * 0.5;
    }
    // The bloom: the blurred radiance's weight above the threshold
    // lifts the frame's bright regions (the composite's intensity the
    // config carries scales it).
    if (params0.y > 0.5) {
        vec3 blurred = sample_blurred(radiance_texture, v_uv,
            vec2(texel_x, texel_y));
        float weight = bloom_weight(max(blurred.r, max(blurred.g, blurred.b)),
            params1.x, params1.y);
        colour += blurred * weight * params1.z;
    }
    // The depth of field: the focus share blends the blurred radiance
    // over the sharp one (the background fragments stay untouched — the
    // chain blurs the scene's own geometry).
    if (params0.z > 0.5 && !background) {
        vec3 blurred = sample_blurred(radiance_texture, v_uv,
            vec2(texel_x, texel_y));
        float distance = length(world_sample.rgb - vec3(0.0));
        float share = 1.0 - focus_share(distance, params2.x, params2.y, params2.z);
        colour = mix(colour, blurred, clamp(share, 0.0, 1.0));
    }
    // The tonemapping: the filmic curve maps the linear radiance onto
    // the display's [0, 1] (the gamma the framebuffer's 8-bit encoding
    // assumes follows).
    if (params0.x > 0.5) {
        float num = colour.r * (2.51 * colour.r + 0.03);
        float den = colour.r * (2.43 * colour.r + 0.59) + 0.14;
        float mapped_r = clamp(num / den, 0.0, 1.0);
        num = colour.g * (2.51 * colour.g + 0.03);
        den = colour.g * (2.43 * colour.g + 0.59) + 0.14;
        float mapped_g = clamp(num / den, 0.0, 1.0);
        num = colour.b * (2.51 * colour.b + 0.03);
        den = colour.b * (2.43 * colour.b + 0.59) + 0.14;
        float mapped_b = clamp(num / den, 0.0, 1.0);
        colour = vec3(mapped_r, mapped_g, mapped_b);
        colour = pow(colour, vec3(1.0 / 2.2));
    }
    // The morphological anti-aliasing: the tonemapped luminance's edge
    // detection reads the four direct neighbours, the crossing ratio
    // against the dominant edge neighbour drives the blend (the single-
    // pass approximation of the classic three-pass morphological
    // operator — the deterministic edge test the example carries).
    if (params2.w > 0.5) {
        float centre = dot(colour, vec3(0.299, 0.587, 0.114));
        vec3 left = texture(radiance_texture, v_uv + vec2(-texel_x, 0.0)).rgb;
        vec3 right = texture(radiance_texture, v_uv + vec2(texel_x, 0.0)).rgb;
        vec3 up = texture(radiance_texture, v_uv + vec2(0.0, texel_y)).rgb;
        vec3 down = texture(radiance_texture, v_uv + vec2(0.0, -texel_y)).rgb;
        // The neighbours tonemap the same way (the LDR image the edge
        // detection reads).
        vec3 tonemapped[4];
        tonemapped[0] = left;
        tonemapped[1] = right;
        tonemapped[2] = up;
        tonemapped[3] = down;
        if (params0.x > 0.5) {
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
                tonemapped[i] = pow(tonemapped[i], vec3(1.0 / 2.2));
            }
        }
        float luma[4];
        luma[0] = dot(tonemapped[0], vec3(0.299, 0.587, 0.114));
        luma[1] = dot(tonemapped[1], vec3(0.299, 0.587, 0.114));
        luma[2] = dot(tonemapped[2], vec3(0.299, 0.587, 0.114));
        luma[3] = dot(tonemapped[3], vec3(0.299, 0.587, 0.114));
        // The dominant edge neighbour: the crossing's largest, the
        // direction the morphological operator blends towards.
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
