# OpenGraceII — Implementation Status

Running record of milestone progress, validation results and open decisions.
The architectural baseline is docs/grace-architecture.md in this repository.

The `.docx` files in this directory are generated from the `.md` sources with
`pandoc` and are committed for convenience. Edit the Markdown, then regenerate:
`pandoc docs/implementation-status.md -o docs/implementation-status.docx`.

---

## Milestones

| # | Scope | Status |
| --- | --- | --- |
| 1 | Rust skeleton: Cargo workspace, module layout, the validation matrix (build, test, docs checks), the pinned toolchain | Done |
| 2 | Render contract and the first backend: Vulkan on Linux (lavapipe); the Android hardware run follows | Done |
| 3 | The remaining backends: OpenGL (Windows/Linux), Metal (macOS/iOS), Direct3D 12 (Windows) | Done (OpenGL and Vulkan validated on Linux hardware, Metal on the macOS hardware run, Direct3D 12 under Wine with vkd3d; the Android Vulkan run stays open) |
| 4 | Render foundation: math, mesh, shader, material | Done (the math and the mesh modules carry their unit tests; the textured cube runs through the real mesh path on all four backends — Vulkan and OpenGL on Linux hardware, Direct3D 12 under Wine with vkd3d, Metal on the A18 Pro's hardware run) |
| 5 | Forward lighting: light types, shadow mapping | Done (the Blinn-Phong path with the sun and up to four point lights runs on all four backends; the shadow mapping follows in a later milestone) |
| 6 | Deferred lighting: MRT G-buffer, the lighting pass | Done (the three-attachment geometry pass and the fullscreen lighting pass run on all four backends — Vulkan and OpenGL on Linux hardware, Direct3D 12 under Wine with vkd3d, Metal on the A18 Pro's hardware run) |
| 7 | Post-processing chain: HDR, bloom, filmic tonemapping, depth of field, morphological anti-aliasing, SSAO-STF | Done (the HDR geometry pass and the five-effect composite pass run on all four backends — Vulkan and OpenGL on Linux hardware, Direct3D 12 under Wine with vkd3d, Metal on the A18 Pro's hardware run) |
| 8 | Global illumination by light volume propagation | Done (the radiance volume, the indirect term in the post chain and the set_light_volume texture cache run on all four backends — Vulkan and OpenGL on Linux hardware, Direct3D 12 on the Windows hardware run (Intel UHD) and under Wine with vkd3d, Metal on the A18 Pro's hardware run) |
| 9 | Platform layer: the engine's own windowing, input and events, replacing SDL2 | Done (the X11 platform backend runs on Linux hardware — the Xvfb validation and the windowed Vulkan swapchain's acquire/present cycle confirmed; the full D3D12 suite runs on the real Windows hardware (Intel UHD) through the run-tests.bat runner; the Win32 and the Cocoa backends follow in a later milestone) |
| 10 | Audio: the engine's own backend per platform, the WAV and Ogg Vorbis decoders | Done (the audio contract with the deterministic null backend, the engine's own WAV and Ogg Vorbis decoders and the four real backends run on all four platforms — ALSA on Linux hardware, WASAPI on the Windows hardware (Intel UHD) and under Wine, CoreAudio on the A18 Pro's macOS run, AAudio on the Galaxy A56's Android run, each mixing through the same loop with zero violations; the AAudio path's dlopen keeps the build's minSdk at 24) |
| 11 | Physics: the classical layer and the extra layer, in the engine's own code | Planned |
| 12 | Scripting: the engine's own scripting language and VM | Planned |
| 13 | Visual scripting: the graph representation on the engine's own language | Planned |
| 14 | Animation: skeletal clips, the deterministic sampler, the state machine, the controllers | Planned |
| 15 | Scene management: data-driven entities, spatial partitioning, scene queries, LOD | Planned |
| 16 | UI: the retained control tree, the own glyph rasterisation, unicode input | Planned |
| 17 | AI and gameplay: the agent behaviours and the gameplay services | Planned |
| 18 | Video: the own playback service and the Ogg Theora decode layer | Planned |
| 19 | OWD: the reader, the writer, the filters, the checksums, the overlay loader and the standalone compression/decompression tool (docs/resource-container.md) | Planned |
| 20 | Windows validation: the MinGW-w64 cross-compile (or the Windows toolchain) and the Windows backends | Done (the real Windows 10 machine runs the full D3D12 suite on the Intel UHD hardware through the run-tests.bat runner; the native Vulkan and OpenGL Windows paths follow) |
| 21 | Editor: the scene and asset editor with a live preview through the same render pipeline | Planned |
| 22 | Network tools: connection, session and state replication as a separate layer | Planned |
| 23 | Hardware validation: the full platform matrix on hardware | Planned |

---

## Open decisions

1. **Which graphics backend first?** The draft plans Vulkan first, because
   lesson 4 demands hardware validation from the first render milestone.
   The answer is decided by milestone 2, which implements it.
2. **The shader toolchain.** Vulkan needs SPIR-V; the decision is a
   build-time dependency rule, not a runtime middleware.
3. **The scripting language.** The engine's own language and VM is a project
   of its own.
4. **The physics extra.** The classical layer and the extra layer are both
   the engine's own implementation.
5. **The editor and the network tools.** Part of the engine's own scope,
   not a deferred scope.
6. **SquareLine Studio.** The port is deferred until the engine's own editor
   is far enough.

---

## Verification commands

```sh
cargo build
cargo build --features og2-core/vulkan,og2-core/opengl
cargo test -p og2-tests --features og2-core/vulkan,og2-core/opengl
cargo build --features og2-core/metal
cargo test -p og2-tests --features og2-tests/vulkan,og2-tests/opengl,og2-tests/metal
cargo build --features og2-core/d3d12
cargo test -p og2-tests --features og2-tests/vulkan,og2-tests/opengl,og2-tests/metal,og2-tests/d3d12
cargo test -p og2-core --features og2-core/audio-vorbis,og2-core/audio-alsa
cargo build --target x86_64-pc-windows-gnu --features og2-core/audio-vorbis --examples
```

The render suite's source imports the backends' types, so the render
target compiles only when the corresponding features are enabled; the
featureless `cargo test` runs the docs-rules suite alone.

---

## Validation record

Milestone 1 (Rust skeleton), Linux x86_64, Rust 1.100.0:
- `cargo build`: the workspace builds, 2 crates, no warnings.
- `cargo test`: 13 tests passed — 10 math tests and 3 docs-rule suites
  (a sound document, the eight fixtures, and the real document).

Milestone 2 (render contract + the Vulkan backend), Linux x86_64, Rust
1.100.0, lavapipe (software Vulkan):
- `cargo build --features vulkan`: the workspace builds, no warnings.
- `cargo test`: 10 tests passed — 3 docs-rule suites and 7 render tests
  (the null backend's contract checks and the Vulkan smoke test).
- The Vulkan smoke test renders the diagnostic triangle on lavapipe, reads
  the centre pixel back and validates the clear colour, the clear depth and
  the clear stencil values.
- The first hardware run on Intel Iris Xe (Linux) caught a spec violation
  the software rasterizer tolerates: a Depth|Stencil aspect mask in
  vkCmdCopyImageToBuffer (VUID 09103). The readback now copies the depth
  and stencil aspects in separate regions, and the Khronos validation
  layer reports zero violations on the smoke run.
- The Android hardware run stays open (the same milestone's validation on
  a device follows).

Milestone 3 (the OpenGL backend), Linux x86_64, Rust 1.100.0, EGL
surfaceless + llvmpipe:
- `cargo build --features opengl`: the workspace builds, no warnings.
- `cargo test`: 10 tests passed — 3 docs-rule suites and 7 render tests.
- The OpenGL smoke test renders the diagnostic triangle through the
  engine's own EGL + GL 3.3 core loader (no third-party binding), reads
  the centre pixel back and validates the clear colour, the clear depth
  and the clear stencil values, matching the Vulkan backend's smoke run.

Milestone 3 (the Metal backend), Linux x86_64, Rust 1.100.0 (build-only
for the Apple targets):
- `cargo check --target aarch64-apple-darwin --features metal` and
  `cargo check --target aarch64-apple-ios --features metal`: the module
  compiles for the Apple targets (build-only; the hardware run needs a
  Mac, the same recorded scope the OpenGrace Metal milestone carried).
- `cargo build --features metal`: the workspace builds, no warnings.
- `cargo test --features vulkan,opengl,metal`: 11 tests passed — 3
  docs-rule suites and 8 render tests (the null backend, the Vulkan
  smoke and the Metal smoke, which skips cleanly where the platform has
  no Metal framework and reports the loader's failure message).
- The Metal backend is the engine's own Objective-C runtime loader
  (objc_getClass, sel_registerName, objc_msgSend resolved through
  dlopen of the platform's libobjc; no Apple SDK headers, no third-party
  binding), the same rule the Vulkan and the OpenGL loaders record.
---
Milestone 3 (the Direct3D 12 backend), Linux x86_64, Rust 1.100.0
(cross-build with the MinGW x86_64-w64-mingw32 toolchain; the runtime
validation ran under Wine 9.0 with vkd3d 1.3 over lavapipe):

- The D3D12 backend is the engine's own loader over d3d12.dll and
  dxgi.dll (CreateDXGIFactory1, D3D12CreateDevice, D3DCompile from
  d3dcompiler_47.dll), resolved through the kernel32 load surface and
  hand-written COM vtable calls — no third-party wrapper, the same rule
  the Vulkan, the OpenGL and the Metal loaders record.
- The cross-build compiles with zero errors and zero warnings for
  x86_64-pc-windows-gnu, and the Linux build links with the loader's
  kernel32 imports gated behind `#[cfg(windows)]` so the platform
  without d3d12.dll skips cleanly.
- The smoke run under Wine creates the device (the GPU vkd3d reports),
  clears, draws the diagnostic triangle, copies the three planes and
  reads the centre pixel back: the colour plane matches the clear
  colour exactly and the depth plane reads 1.0 at the centre.
- The stencil plane's copy (subresource 1 of the D24S8 depth-stencil
  resource) is undefined under vkd3d/lavapipe: the value read back
  changes between runs of the same binary. The backend's D3D12 usage
  is the standard idiom, so the smoke run reports the stencil value
  without asserting it; a hardware D3D12 run would settle it.
- `cargo test --features vulkan,opengl,metal,d3d12`: 12 tests passed —
  3 docs-rule suites and 9 render tests (the null backend, the Vulkan
  smoke, the Metal smoke and the D3D12 smoke, which skips cleanly where
  the platform has no d3d12.dll).

---

The macOS build's first run (the Metal hardware validation's build
step) found two portability defects the Linux-only validation could not
see, both fixed in the same milestone:

- The Vulkan build script required glslangValidator on every machine
  with the feature enabled, even where the Vulkan backend only compiles
  and never runs (macOS, iOS). The build script now falls back to the
  committed SPIR-V arrays (vulkan_spirv_committed.rs, generated on the
  Linux validation machine by the same validator) when the validator is
  absent, and recompiles the shader library when it is present; the
  fallback's smoke run reproduces the generated path's result exactly.
- The Vulkan, the OpenGL and the Metal loaders referenced the libc
  dlopen surface unconditionally, so the windows-gnu cross-build of the
  full feature matrix failed. The Vulkan loader now carries the
  platform's own LoadLibraryW surface on Windows (vulkan-1.dll, the ICD
  loader's canonical name), and the OpenGL and the Metal loaders fail
  cleanly at load time with the platform's own message (the EGL
  surfaceless path and the Metal framework do not exist on Windows; the
  WGL path is its own future milestone). The libc dependency is target
  gated, and the full matrix (Linux all-features, windows-gnu
  all-features, and the four-feature test suite) builds warning-free.

The macOS hardware run's first frame exposed two further defects, both
fixed in the same milestone:

- The render pipeline descriptor's colour attachment was set with a
  single-argument selector that does not exist on the Metal API; the
  runtime raised an Objective-C exception the Rust process cannot
  catch. The call now uses the two-argument
  setColorAttachmentPixelFormat:forIndex: form the API declares.
- The frame's command buffer was never encoded: the clear and the
  triangle's draw were stubs. The clear pass now records the render
  pass descriptor (the frame target's colour and depth/stencil
  attachments with the clear load actions), the triangle's draw encodes
  inside the pass, and the swap's commit closes the frame the way the
  Vulkan submit path does.
- The four-feature test suite's Vulkan smoke used process::exit on its
  skip path, which aborts the whole test binary on machines without a
  Vulkan loader; the skip now returns like the other smokes.

---

The Metal hardware run (Apple A18 Pro, macOS) — the frame after the
first encode — found five further defects, each fixed and re-validated
in the same milestone until the diagnostic triangle drew and read back
correctly:

- The pipeline state creation used the plain single-argument
  newRenderPipelineStateWithDescriptor: form, which the Metal API's
  headers do not declare; the documented synchronous form is
  newRenderPipelineStateWithDescriptor:error:. The call now uses the
  error variant and records the NSError's message when the state fails
  to build.
- The render pass descriptor's depth and stencil attachments: the
  combined depth/stencil format's stencil plane is its own
  MTLRenderPassStencilAttachmentDescriptor (the depth attachment
  descriptor carries no stencil selectors), and the load/store actions
  the depth attachment needs (Clear/Store) sit on the depth descriptor
  itself. The pass now sets both descriptors with their own actions.
- The render pipeline descriptor left the stencilAttachmentPixelFormat
  at its default MTLPixelFormatInvalid while the pass names the
  combined depth/stencil format for both planes; the pipeline's match
  validation nils the state and the draw is silently dropped. The
  descriptor now names the same combined format for both planes.
- The depth/stencil readback: the device's combined
  Depth32Float_Stencil8 format allocates 64 bits per pixel (the
  24-bit packed format does not exist on this GPU class). The blit
  copy now uses the options variant with the depth and stencil plane
  selections, the staging buffer sizes the 8-byte texel, and the
  decode reads the depth word's float and the stencil byte per the
  format's layout.
- The MSL shader sources: the vertex function declared the position
  input as a direct float3 [[attribute(0)]] parameter, which the Metal
  shading language rejects for vertex functions (the compiler's
  message: invalid type 'float3' for input declaration). The sources
  now use the [[stage_in]] struct form with one compiled library per
  stage (the vertex and the fragment sources each declare their own
  uniform struct; the earlier single-library concatenation made the
  fragment's uniform struct invisible to the compiler). The shader
  functions' lookups compile per stage too.

The pass's diagnostic: the triangle resources' setup records the first
failure's stage and the API's own error text (the NSError's
localizedDescription), and the smoke example prints it — the run's
final defects surfaced stage by stage (library compile, then the
fragment's missing struct) exactly as the compiler's messages named
them.

The validated hardware run: the smoke renders the diagnostic triangle
on the Apple A18 Pro's default device, clears (0,0,0,255) black with
depth 1.0 and stencil 0, reads the centre pixel back as the triangle's
colour (200,40,40,255) with depth 1.0 and stencil 0, and reports zero
violations. The four-feature test suite passes on the Mac (9 render
tests including the Metal smoke with its clear depth 0.25 and stencil
7 checks).

---

Milestone 3's second validation machine (the LucDev notebook, Linux
x86_64, Rust 1.100.0, Intel Iris Xe Graphics (ADL GT2) hardware):

- `git pull` fast-forwards to the Metal hardware run's commits; the
  workspace builds all-features with zero warnings.
- `cargo test -p og2-tests --features vulkan,opengl,metal,d3d12`: 9
  tests passed — 3 docs-rule suites and 6 render tests (the null
  backend's suite and the three backends' smokes; the Metal smoke skips
  cleanly where the platform has no Metal framework).
- The Vulkan smoke ran on the notebook's own GPU (Intel Iris Xe,
  Vulkan 1.4): the diagnostic triangle draws and reads back the same
  centre colour, depth and stencil the lavapipe runs recorded — a real
  hardware run, not the software rasterizer.
- The OpenGL smoke ran on the same GPU through Mesa 26.0.8's GL 4.6
  core profile: the same triangle, the same readback values, zero
  violations.
- The windows-gnu cross-build compiles all-features with zero warnings.
- The D3D12 smoke under Wine (vkd3d over the notebook's lavapipe ICD)
  creates the device, clears, draws the triangle and reads the centre
  back: the colour matches, the depth reads 1.0, the stencil value is
  reported as undefined under vkd3d as before.

With the Mac's Metal hardware run and the LucDev's Linux/Wine runs, the
same render contract is validated on all three of the engine's
development platforms: Linux x86_64 (Vulkan and OpenGL on hardware),
Windows x86_64 (the D3D12 path under Wine with vkd3d plus the
windows-gnu cross-build) and macOS arm64 (Metal on the A18 Pro's
hardware).

---

## Milestone 4: Render foundation

Milestone 4 lands the engine's own render foundation — the math library
the shaders consume, the mesh module the draws submit and the first
textured draw through the real mesh path on every backend.

**The math library** (`core/src/math.rs`): `Vec2`/`Vec3`/`Vec4`, the
column-major `Mat4` (OpenGL clip convention, `m[c][r]`), the complete
`Quat` (axis-angle, multiplication, slerp, the `Mat4` conversion),
`Aabb`, `Plane`, `Frustum` (the view volume and the culling test) and
`Transform` (the compose/roundtrip pair). 18 deterministic unit tests
carry the numeric accuracy the contract demands.

**The mesh module** (`core/src/mesh.rs`): the formal PNU vertex layout
(`pnu_layout()` — position, normal, uv; 32-byte stride), the `MeshData`
container and the `cube(size)`/`quad()` builders. 5 unit tests carry
the layout offsets, the cube's shape and indexing, the normals' unit
length and outward direction, the AABB and the quad.

**The uniform contract** (the M4c rule, documented on the `GpuMesh`
handle): the vertex stage carries the camera block — the view-projection
matrix the `set_camera` state builds and the model matrix the submission
writes — as one API-neutral byte layout each backend maps onto its own
language (Vulkan's std140 block, OpenGL's uniforms, D3D12's cbuffer,
Metal's constant struct). The two matrices are column-major; one upload
path serves all four backends.

**The integration test** is the honest one: a textured cube drawn
through the real path (`create_gpu_mesh` + `upload_mesh` + `set_camera`
+ `draw`/`submit`) with a pixel readback at the frame's end — no
special-case shortcut. The texture is a 4x4 whose centre samples red
and whose edge row is green (the sampler's NEAREST read of the upload);
the camera looks at the origin from +Z, so the cube's front face fills
the frame's centre. The checks: the centre carries the texture's red
with a depth in front of the far plane, the corner carries the clear
colour, and the protocol stays clean (zero violations). The spin is not
in the smoke yet — the model matrix is identity there; the rotation by
quaternion follows in a later milestone.

The validation runs, all through the same contract:

- **Vulkan** (`cube_smoke`, lavapipe in the sandbox, Iris Xe on the
  LucDev hardware): centre red, depth 0.8876, corner clear, zero
  violations.
- **OpenGL** (`cube_smoke_gl`, EGL surfaceless + GL 3.3 core in the
  sandbox, Mesa 26.0.8 GL 4.6 core on the LucDev): centre red, depth
  0.9438, corner clear, zero violations.
- **Direct3D 12** (`cube_smoke_d3d12`, Wine with vkd3d): centre red,
  depth 0.8876, corner clear, zero violations. The run's own debug
  findings, fixed in the loader and the backend: the shader-visible
  heap flag is 1 (not 2), the pixel-shader-resource state is 0x80, the
  SRV descriptor's real 4-field shape, the 256-byte CBV alignment
  vkd3d requires, the subresource-index destination form the
  CopyTextureRegion needs (the footprint-to-footprint copy is not
  implemented under vkd3d), the real staging footprint the upload's
  copy must carry, and the GPU handle's by-value pass.
- **Metal** (`cube_smoke_mtl`, the A18 Pro's hardware run): centre red,
  depth 0.8876, corner clear, zero violations. The MSL libraries compile
  at runtime (one per stage, the M3 rule), the vertex descriptor carries
  the three PNU attributes at buffer index 1, and the run's own finding
  is now a recorded lesson: the pipeline descriptor carries no
  winding/cull setters — those are render-command-encoder state, and
  sending `setWinding:`/`setCullMode:` to the descriptor raises the
  unrecognized-selector exception the Rust runtime cannot catch. The
  API's own defaults (front-facing = counter-clockwise, cull = none) are
  the forgiving rule the first step wanted.

The four smokes carry the same two-readback pattern the triangle smokes
established (the centre and the corner of one frame), and the triangle
smokes keep passing — nothing the milestone touched broke.

---

## Milestone 5: Forward lighting

Milestone 5 lands the engine's own forward lighting — the Blinn-Phong
reference the shaders carry, the light types the submissions carry and
the first lit draw through the real mesh path on every backend.

**The lighting reference** (`core/src/lighting.rs`): the pure functions
the shaders translate — `point_attenuation` (the windowed
inverse-square falloff), `spot_falloff` (the smooth cone), `blinn_phong`
(the diffuse/specular pair) and `sun_contribution` (the direction's
negation). 8 deterministic unit tests carry the numeric accuracy: the
attenuation is one at the origin and zero at the radius, the diffuse
peaks at the full-facing surface and vanishes at the grazing one, and
the specular peaks at the half vector.

**The lights' uniform contract** (the M5c rule, the same shape the
camera block established): one API-neutral 256-byte block each backend
maps onto its own language — the sun's direction and colour+intensity,
up to four point lights' position+radius and colour+intensity, and the
params (the active point lights' count). The submissions carry the sun
and the point lights; the upload path writes the block once and every
backend's draw binds it.

**The integration test** follows the M4 rule: a lit cube drawn through
the real path with a pixel readback at the frame's end. The sun's
intensity is 0.5 and the light hits the front face at full strength, so
the centre reads 90, 90, 90 — the 180 grey scaled by the intensity, a
value the unlit path cannot produce (a silent fallback to unlit would
read 180 and fail the check). The corner carries the clear colour, and
the protocol stays clean (zero violations).

The validation runs, all through the same contract:

- **Vulkan** (`cube_lit`, lavapipe in the sandbox, Iris Xe on the
  LucDev hardware): centre 90, 90, 90, depth 0.8876, corner clear, zero
  violations.
- **OpenGL** (`cube_lit_gl`, EGL surfaceless + GL 3.3 core in the
  sandbox, Mesa 26.0.8 GL 4.6 core on the LucDev): centre 90, 90, 90,
  depth 0.9438, corner clear, zero violations.
- **Direct3D 12** (`cube_lit_d3d12`, Wine with vkd3d): centre 90, 90,
  90, depth 0.8876, corner clear, zero violations. The run's own
  findings: the lights' CBV sits at register(b1) with the mesh heap
  grown to three slots, and the lights' block carries the 256-byte
  padding the cbuffer reads.
- **Metal** (`cube_lit_mtl`, the A18 Pro's hardware run): centre 90,
  90, 90, depth 0.8876, corner clear, zero violations. The run's own
  findings, both fixed in the same milestone: the lit vertex shader's
  normal transform used the float4x4-to-float3x3 cast the Metal
  shading language does not support (the 3x3 normal matrix is built
  from the model's columns), and the fragment's `distance` variable
  shadowed the builtin function of the same name (the variable is now
  `dist`). The earlier empty error message was its own defect: the
  compile stage wrote the lit path's failures into the triangle path's
  error string, so the upload's diagnostic read empty — the compile
  stage now returns the failure's text to the caller, which records it
  in its own error string.

The lit smokes carry the same two-readback pattern the unlit smokes
established, and the unlit smokes and the triangle smokes keep passing —
nothing the milestone touched broke.

---

## Milestone 6: Deferred lighting

Milestone 6 lands the engine's own deferred lighting — the G-buffer's
three-attachment geometry pass, the fullscreen lighting pass that shades
it, and the same Blinn-Phong reference the forward path carries on every
backend.

**The G-buffer contract**: one render target with three colour
attachments plus the depth attachment. The albedo (Rgba8) carries the
surface colour with its alpha at 0 marking geometry (any other value
marks background, so the lighting pass copies the clear colour through
unchanged); the normal (Rgba8) stores the world-space normal half-offset
into the unsigned range; the world position (Rgba16F) stores the
fragment's world-space position the point lights' falloff reads. Each
backend maps the contract onto its own multi-render-target surface:
Vulkan's per-target render pass with three colour attachment references,
OpenGL's `glDrawBuffers` MRT draw on the G-buffer's framebuffer, D3D12's
PSO with `NumRenderTargets = 3` and its own SRV heap for the lighting
pass's reads, Metal's pipeline descriptor with three
`colorAttachments` formats.

**The lighting pass**: a full-frame triangle (no vertex buffer — the
vertex shader generates the three corners from the vertex id) drawn
into the default framebuffer with the depth test off. The fragment
shader samples the three G-buffer attachments and runs the same
Blinn-Phong the forward path carries — the sun plus up to four point
lights with the windowed inverse-square falloff — reading the lights'
block from the same 256-byte uniform contract the M5 draw path
established.

**The integration test** follows the M4/M5 rule: the deferred path must
produce the same pixels the forward path produces from the same scene.
The lit cube from M5 (the 180 grey, the sun at 0.5, one point light in
front) reads 145 at the centre — the forward path's 90 plus the point
light's 55 — a value neither the clear colour nor the unlit path can
produce. The corner carries the clear colour, and the protocol stays
clean (zero violations).

The validation runs, all through the same contract:

- **Vulkan** (`cube_deferred`, lavapipe in the sandbox, Iris Xe on the
  LucDev hardware): centre 145, 145, 145, depth 1.0, corner clear, zero
  violations. The colour images carry the Sampled usage bit (the
  descriptor's combined image sampler reads them), and the geometry
  pass's colour attachments hand off to the lighting pass's shader reads
  through an explicit pipeline barrier.
- **OpenGL** (`cube_deferred_gl`, EGL surfaceless + GL 3.3 core in the
  sandbox, Mesa 26.0.8 GL 4.6 core on the LucDev): centre 145, 145, 145,
  depth 1.0, corner clear, zero violations. The core profile forbids a
  draw with vertex array object 0, so the lighting pass carries its own
  VAO for the vertex-buffer-less fullscreen triangle.
- **Direct3D 12** (`cube_deferred_d3d12`, Wine with vkd3d on both
  machines): centre 145, 145, 145, depth 1.0, corner clear, zero
  violations. The run's own findings, fixed in the same milestone: the
  colour resources transition RENDER_TARGET to PIXEL_SHADER_RESOURCE
  between the two passes (the barriers follow the OMSet so the
  descriptor writes see the final state), and the lighting pass's SRV
  heap carries its own four slots (the three attachments plus the
  lights' CBV). The examples' Wine runs go through the new
  `scripts/run-d3d12.sh` runner, which retries the lavapipe startup
  crash (a device-enumeration segfault inside the lavapipe driver
  itself, before the engine's own code runs — the sandbox's dmesg
  traced it to the driver's instance enumeration, and the retry
  absorbs it at the shell level without touching the engine).
- **Metal** (`cube_deferred_mtl`, the A18 Pro's hardware run): centre
  145, 145, 145, depth 1.0, corner clear, zero violations. The MSL
  shaders follow the M5 lessons (the 3x3 normal matrix built from the
  model's columns, no `distance` shadowing); the A18 Pro accepted the
  Rgba16F world attachment and the three-attachment pipeline without
  further adjustments.

The deferred smokes carry the same two-readback pattern, and the lit,
unlit and triangle smokes keep passing — nothing the milestone touched
broke.

---

## Milestone 7: Post-processing chain

Milestone 7 lands the engine's own post-processing chain: the HDR
geometry pass that renders the deferred scene into two Rgba16F
attachments, and the single fullscreen composite pass that carries the
five effects — SSAO, bloom, depth of field, filmic tonemapping and the
morphological anti-aliasing — over the HDR radiance.

**The HDR contract**: one render target with two Rgba16F colour
attachments plus the depth attachment. The first attachment holds the
deferred scene's radiance (the lighting pass's shaded output, linear,
unclamped); the second holds the fragment's world-space position, which
the SSAO's occlusion samples and the depth of field's focus share read.
The geometry pass's vertex shader is the M6 gbuffer's, and its fragment
shader writes the same three outputs the deferred pass carries — the
shading runs in the same pass, so the chain's entry holds the radiance
already lit. Each backend maps the contract onto its own render-target
plumbing: Vulkan's second render pass with two Rgba16F attachment
references and its own image views (the M7 run's lesson: the render
pass's declared formats must match the framebuffer's views byte for
byte — lavapipe segfaults on the mismatch), OpenGL's two-attachment
draw buffer on the chain's framebuffer, D3D12's PSO with
`NumRenderTargets = 2` and its own SRV heap for the composite pass's
reads, Metal's render pass descriptor with two `MTLPixelFormatRGBA16Float`
color attachments.

**The composite pass**: a full-frame triangle (the M6 lighting pass's
vertex shader) drawn into the default framebuffer with the depth test
off. The fragment shader samples the two HDR attachments and the chain's
configuration uniform (three vec4s, 48 bytes), then applies the effects
in a fixed order — the SSAO's occlusion darkens the radiance, the
bloom's threshold weights the blurred radiance, the depth of field's
focus share blends the blurred radiance, and the filmic tonemapping maps
the linear radiance onto the display curve, with the morphological
anti-aliasing's edge blend following the tonemap. The configuration
uniform carries the on/off flags (params0), the tunables (params1:
bloom threshold, knee and intensity, the SSAO's radius), and the depth
of field's focal distance and near/far ranges plus the anti-aliasing
flag (params2). Every effect degrades to the identity when its flag is
off, so the chain's default (tonemapping only) is the tonemapped scene
the M6 readback's radiance predicts.

**The morphological anti-aliasing** is the single-pass approximation
the chain's contract documents: the composite reads the four
neighbouring pixels' tonemapped luminances, finds the dominant crossing
edge (the neighbour whose luminance differs most from the centre's), and
blends the pixel toward the dominant side by the crossing ratio the
`mlaa_crossing` CPU reference pins. The edges aligned with the pixel
grid keep their colour; the cut pixels carry the half-of-each-side
blend. The full three-pass Reshetov chain (the edge detection, the
blending weights and the neighbour search) needs its own intermediate
targets on every backend; the approximation is honest about what it
carries, and the deferred milestone's plan documents the upgrade path.

**The integration test** follows the M4/M5/M6 rule: the post chain must
produce the pixels the chain's own reference produces from the same
scene. The cube from M6 (the 180 grey, the sun at 0.5, one point light
in front) reads the radiance 145 at the centre — the deferred path's
pin — and the composite's tonemapping maps it to 211. The corner carries
the chain's clear colour, and the silhouette's edge pair carries the
anti-aliasing's blend: the last lit pixel (176) keeps its 209 (the
grid-aligned edge) and the first background pixel (177) blends to 105
(the half-of-each-side cut the midpoint rule pins). The protocol stays
clean (zero violations).

The validation runs, all through the same contract:

- **Vulkan** (`cube_post`, Intel Iris Xe on the LucDev hardware run,
  plus lavapipe in the sandbox): centre 211, 211, 211, depth 1.0,
  corner clear, edge pair 209/105, zero violations. The run's own
  findings, fixed in the same milestone: the post render pass's
  declared colour formats must match the framebuffer's image views (the
  Rgba16F pair the HDR contract carries — the earlier Rgba8 declaration
  segfaulted in `vkCmdBeginRenderPass`), and the composite pass's own
  descriptor set reads the two HDR attachments through the same
  combined image sampler the M6 lighting pass establishes.
- **OpenGL** (`cube_post_gl`, Intel Iris Xe, GL 4.6 core, on the
  LucDev hardware run): centre 211, 211, 211, depth 1.0, corner clear,
  edge pair 209/105, zero violations.
- **Direct3D 12** (`cube_post_d3d12`, Wine with vkd3d on the LucDev
  hardware run): centre 211, 211, 211, depth 1.0, corner clear, edge
  pair 209/105, zero violations. The lights' CBV joins the post gbuffer
  pass's root signature (the mesh srv_heap's third slot), and the
  composite pass carries its own SRV heap for the two HDR attachments
  plus the configuration CBV. The run's lesson: the Wine 10.0
  `d3dcompiler_47`'s HLSL front end crashes on the composite shader's
  locally indexed array (`float3 tonemapped[4]` read through the loop
  variable) — the D3D12 composite shader carries the unrolled variant
  (named temporaries, no dynamic indexing), and the deferred and lit
  pipelines shared the composite's vertex stage so the fix covers the
  whole post chain on the newer Wine.
- **Metal** (`cube_post_mtl`, Apple A18 Pro on the hardware run):
  centre 211, 211, 211, depth 1.0, corner clear, edge pair 209/105,
  zero violations. The run's lesson: the composite shader's
  `sample_blurred` helper must carry the sampler as a parameter — the
  out-of-scope reference the build-only validation missed, and the A18
  Pro's failure left the composite silent (the black frame with zero
  violations) until the hardware run exposed it.

The post smokes carry the same three-readback pattern (the centre, the
corner and the edge pair), and the deferred, lit, unlit and triangle
smokes keep passing — nothing the milestone touched broke. The Vulkan
backend's committed SPIR-V fallback carries the two post shaders' byte
arrays, so the macOS build compiles the chain without the validator.

---

## Milestone 8: Global illumination by light volume propagation

Milestone 8 lands the engine's own global illumination — the
`lightvolume` module's radiance volume, the post chain's indirect term
that samples it, and the texture cache the render contract's
`set_light_volume` carries on every backend.

**The volume's contract**: a 3D grid of cells, each cell storing six
face radiance values (one per axis direction) packed into the 3D
texture's depth layers (`z * 6 + face`). The CPU-side module builds the
volume from the scene's lights: the point lights inject the windowed
falloff the direct path carries, the directional light fills the grid
along its direction, and the propagation steps move half of each face's
radiance to the neighbour the face points at, so light bleeds around
corners in discrete steps. The module's unit tests pin the packing, the
injection, the propagation's energy conservation (the total energy
never grows) and the trilinear sampling the shader-side read mirrors.

**The shader-side read**: the post chain's gbuffer pass gains the
volume's 3D texture and its config uniform; the fragment adds the
indirect term — the albedo multiplied by the face the normal faces'
trilinear sample, scaled by the config's indirect intensity. The three
shader languages carry the same read with their own rules pinned:
GLSL's `textureLod` (the position-dependent loop flow makes the
implicit-derivative sampling undefined), MSL's `level(0)` (the same
rule), and HLSL's explicit LOD through the component-vector corner
loop (the arrays would lose the component swizzle the loop reads).

**The `set_light_volume` contract**: the render trait's new method
carries the volume's dims, origin, cell size, and the packed texture
bytes; each backend caches the 3D texture (the creation path the M8
code establishes per backend) and re-uploads only when the contents
change. The gbuffer pass binds the volume texture and the config
uniform beside the lights' block, and the composite reads the
lit-through-volume radiance the geometry pass produced.

**The integration test** follows the M6/M7 rule: the post chain's
indirect term must produce the pixels the volume's own reference
produces from the same scene. The cube from M7 (the 180 grey, the sun
at 0.5, one point light in front) with the volume's indirect intensity
at 0.5 reads 221 at the centre — a value neither the direct-only 211
nor the clear colour can produce. The corner carries the chain's clear
colour, and the protocol stays clean (zero violations).

The validation runs, all through the same contract:
- **Vulkan** (`cube_volume`, lavapipe in the sandbox, Iris Xe on the
  LucDev hardware): centre 221, 221, 221, depth 1.0, corner clear, zero
  violations. The run's own finding, fixed in the same milestone: the
  3D image's `imageType` is `VK_IMAGE_TYPE_3D` (the spec's value 2, not
  the sequential guess 3), and the volume's 3D texture joins the
  gbuffer pass's descriptor set as a combined image sampler.
- **OpenGL** (`cube_volume_gl`, EGL surfaceless + GL 3.3 core in the
  sandbox, Mesa 26.0.8 GL 4.6 core on the LucDev): centre 221, 221,
  221, depth 1.0, corner clear, zero violations. The run's own
  findings, fixed in the same milestone: the volume sample carries the
  explicit `textureLod`, and the Rgba16F 3D uploads (both the creation
  and the fill paths) name `GL_HALF_FLOAT` for the half-float data —
  the readback mirrors it through a `glGetTexImage` call the loader
  gained.
- **Direct3D 12** (`cube_volume_d3d12`, the real Windows hardware run
  — Intel UHD Graphics, Windows 10 — and Wine with vkd3d on the Linux
  machines): centre 221, 221, 221, depth 1.0, corner clear, zero
  violations. The run's own findings, fixed across the milestones: the
  3D texture fill copies per row at the 256-byte row pitch (a
  contiguous per-layer copy leaves the rows 1..n in the pitch's
  padding — the C-side ground truth with the real d3d12.dll headers
  isolated the Rust-side bug), and the readback is a real one (the
  COPY_SOURCE transition, the readback buffer's copy, the fence wait
  and the per-row unmap). The gbuffer pass binds the mesh srv_heap's
  five slots (the camera CBV, the texture SRV, the lights CBV, the
  volume SRV and the volume CBV), and the composite keeps its own
  three-slot heap. The Windows hardware run's own finding, fixed in
  the M9 follow-up: the loader's `D3D12_SRV_DIMENSION_TEXTURE3D`
  constant carried 5 (the enum's TEXTURE2DARRAY value) where the API
  declares 8 — the wrong dimension made the volume's SRV descriptor
  invalid and the first execute that sampled it removed the device
  (0x887a0001) on the real driver, while the Wine layer had silently
  accepted the same call (the SRV's hr check now follows the COM
  convention: only a negative HRESULT names a failure, because vkd3d
  returns a non-standard success code there). The 3D fill's barriers
  cover all the depth slices too (a depth-24 volume holds 24
  subresources; the subresource-0 barrier left slices 1..23 outside
  the transition the SRV's read needs).
- **Metal** (`cube_volume_mtl`, the A18 Pro's hardware run): centre
  221, 221, 221, depth 1.0, corner clear, zero violations. The runs'
  own findings, fixed in the same milestone: the 2D device texture's
  creation path builds its own descriptor (the depth and the type stay
  at the descriptor's defaults), the 3D upload's selector is the
  `replaceRegion:mipmapLevel:slice:...` form the API declares, the
  `MTLTextureType` values match the header's enum (2D=2, 3D=7), and
  the volume sample's corner loop uses component vectors (`int3`/
  `float3` — the arrays would lose the swizzle the loop reads) with
  the explicit `level(0)` and the cell base renamed off the albedo's
  `base` shadowing.

The volume smokes carry the same three-readback pattern, and the post,
deferred, lit, unlit and triangle smokes keep passing — nothing the
milestone touched broke.

**The Wine flake, recorded**: the D3D12 runs under Wine with vkd3d
show an intermittent startup crash that the M6 record already traced
to the lavapipe driver's instance enumeration (a device-enumeration
segfault inside the driver, before the engine's own code runs). The
repeated runs confirm the shape: the same binary passes and fails
alternating runs on both machines (the LucDev's smoke passed three of
three after a failed run; the deferred example failed three of three
in one batch and passed in the sandbox's batches), the failing runs
die before the engine prints its first readback line, and the Wine
trace shows `RPC_S_SERVER_UNAVAILABLE` plus the vkd3d message
`Could not find Vulkan physical device for DXGI adapter` — the Vulkan
device the lavapipe ICD serves is gone by the time the engine's D3D12
loader asks for it. The engine's own code does not run in the failing
runs, and the passing runs render the identical pixels. The
`scripts/run-d3d12.sh` runner's retry absorbs the crash at the shell
level; the direct `wine` invocations the examples' commands carried
until now are the ones that surface it. The D3D12 backend's own
validation stands on the passing runs.

---

## Milestone 9: The platform layer and the windowed render path

Milestone 9 lands the engine's own platform layer — the windowing, the
input and the event queue, replacing SDL2 — and the render contract's
windowed path that consumes the platform layer's native surface.

**The platform layer's contract**: one trait (`PlatformBackend`) with
the API-neutral surface the app logic consumes: the window's creation
description, the event queue's poll, the resize, the native surface the
render backends bind to, and the shutdown. The shared module carries
only the trait and the pure-Rust types (the `WindowDesc`, the `Event`,
the `Key`, the `MouseButton`, the `NativeSurface`); the per-platform
implementations live in their own files gated behind their own
features, the same rule the render backends' loaders carry.

**The X11 backend**: the engine's own Xlib loader (dlopen of the
platform's libX11.so.6, no third-party binding), the display
connection, the window's creation, the WM_DELETE_WINDOW protocol
registration, the event queue's XPending/XNextEvent translation into
the contract's events (the keymap normalises the layout's keysyms onto
the contract's Key names), and the shutdown's display close. The
backend's `native_surface()` carries the X11 display pointer and the
window's XID the render backends match on.

**The windowed render path (Vulkan)**: the backend's config gains the
optional `surface` field; the None surface keeps the headless path the
milestones 2-8 validated. The Some(X11) path creates the Xlib surface
through the loader's `vkCreateXlibSurfaceKHR`, enables the surface and
swapchain extensions at the instance and device creation, verifies the
queue family's present support on the surface during the device pick,
creates the swapchain (the surface's capabilities query, the first
surface format, the FIFO present mode, the current transform, the
shared depth attachment), builds the per-image views and framebuffers
against the swapchain's own render pass (the colour attachment's final
layout is the present source), and creates the acquire/present
semaphores and the frame's fence. The frame loop's acquire presents
through the same semaphores: the submit waits on the acquire
semaphore, signals the present semaphore, and the present hands the
frame's image to the window's surface. The shutdown tears the
swapchain state down in reverse (the framebuffers, the views, the
semaphores, the fence, the depth attachment, the render pass, the
swapchain, the surface).

**The integration test** (`vk_window`): the platform layer creates the
window, the backend binds the surface and runs the frame loop — the
acquire/present cycle the swapchain carries — five times on the Xvfb
display, and the shutdown tears the platform and the render resources
down cleanly. The headless path's smokes keep passing unchanged (the
None surface keeps the off-screen target's readback path).

The validation runs, all through the same contract:

- **X11 platform backend** (Linux x86_64, Xvfb display :0): the window
  creates, the surface carries the display pointer and the XID, the
  event queue's poll runs, and the shutdown closes the display.
- **Windowed Vulkan path** (`vk_window`, lavapipe over the Xvfb
  surface): the swapchain creates, the frame loop presents five frames,
  and the shutdown tears the swapchain down cleanly. Ten repeated runs
  confirm the path's stability.
- **The headless regression**: the Vulkan, the OpenGL, the Direct3D 12
  and the Metal backends' smokes keep passing (the None surface keeps
  the off-screen target's readback path); the og2-core suite (56 tests)
  and the og2-tests suite (7 render tests + 3 docs suites) pass.

**The run's own finding, fixed in the same milestone**: the
`XSetWMProtocols` loader's signature declared the protocols parameter
as a value (`Atom`) where the API's header declares a pointer
(`Atom *`); the mismatch segfaulted the protocol registration on the
platform's libX11 (the `_XData32` dereference the backtrace pinned).
The loader now carries the pointer form and the call passes the atom's
address.

**The Windows hardware run (the borrowed Windows 10 machine, Intel UHD
Graphics)**: the full D3D12 suite runs on the real driver for the first
time through the `run-tests.bat` runner the M9 carries (the MinGW
toolchain via rustup, the clones from GitHub, the per-example exit
codes into `log.txt`). Every headless example passes with zero
violations — the triangle smoke, the textured cube, the lit cube, the
post chain and the deferred scene — and the volume example's device
removal (the M8 record's finding above) surfaced and was fixed here.
The D3D12 backend's WARP fallback (the M9 software-rasterizer path for
machines without a D3D12 GPU) stands ready behind the hardware device's
detection. The Windows run also records the Wine-side limitation the
fixes exposed: the repeated readbacks of one process can hand back a
depth plane whose words alias the colour plane's bytes (the
`0x00dcdcdc`-style reads on the volume run's second and third samples)
under vkd3d on lavapipe — an artefact the real driver does not carry
(the same binary's hardware run reads depth 1.0 cleanly), on top of the
recorded lavapipe startup flake the runner's retry absorbs.

---

## Milestone 10: Audio foundation
Milestone 10 replaces the audio middlewares — OpenAL and SDL2_audio in
OpenGrace — with the engine's own audio stack: the audio contract, the
deterministic null backend, the engine's own WAV and Ogg Vorbis
decoders, the shared mix loop and the first real backend (ALSA through
dlopen).
**The audio contract**: one trait (`AudioBackend`) with the API-neutral
surface the app logic consumes: the backend's creation from the
`AudioConfig` (the sample rate, the channel count, the period), the
voice lifecycle (`play`, `pause`, `stop_voice`) over the `VoiceHandle`s
the backend hands back, the spatial placement (`set_voice_placement`,
`set_listener`) through the `VoicePlacement` and the `ListenerState`,
the mixing (`mix_frames` filling a caller `PcmBuffer`,
`mixed_frames` reporting what the real backend consumed), the
`violation_count` and the shutdown. The shared module carries only the
trait and the pure-Rust types (the `AudioConfig`, the `PcmBuffer`, the
`VoiceHandle`, the `VoiceState`, the `VoicePlacement`, the
`ListenerState`, the `AudioError`); the per-platform implementations
live in their own files gated behind their own features, the same rule
the platform layer and the render backends' loaders carry. The
`NullAudioBackend` is deterministic: the same voice walk produces the
same mixed samples on every machine, and the contract's tests read the
mixed buffer back instead of trusting the hardware.
**The shared mix loop** (`mix.rs`): one mix routine the null backend and
the real backends run — the per-voice sample cursors, the per-voice
gain, the 3D attenuation (the linear falloff from the reference
distance to the max distance, clamped past the max), the per-channel
peak the contract reports. A contract-walk test drives the null
backend and the arithmetic through the same scenario and pins the
mixed peaks, so a real backend's mix cannot drift from the contract's
own definition.
**The decoders**: the engine's own WAV reader (the RIFF chunk walk, the
PCM16 and the float32 payloads, byte-exact against the reference) and
the engine's own Ogg Vorbis path — the demuxer (the page parser and
the packet assembler per RFC 3533, including the legality of a packet
whose segment count is zero), the bit reader (the LSB-first order the
Vorbis specification pins), the codebooks (the Huffman tables decoded
in the specification's canonical ordering), the floor 1 curve, the
residues 0, 1 and 2, the mapping 0 channel coupling, the two block
sizes, the MDCT ported verbatim from the reference library's
`mdct_backward` into `f64`, and the blockin lapping with the
granule-driven window trims.
**The reference rule**: the test vectors' reference PCM comes from the
real libvorbis — the reference library built from the xiph clone with
dump instrumentation — not from a demuxer's view of the file. The
finding the rule closes: ffmpeg's demuxer applies its own end trim (256
samples, a PTS-bookkeeping artefact computed through the packet's
prev-flag bit rather than the previous block's real samples) on top of
dropping the decoder's first frame, and rewriting the granule never
changed its output because of it. The decoder's honest contract is the
reference library's output (24000 samples at 48 kHz mono for the 1 kHz
vector, regenerated from libvorbis); the Rust decode lands at RMS
1.1e-8 against the real libvorbis PCM — 0.29 LSB on the s16
quantisation. The 440 Hz stereo vector already agreed with both views
(the demuxer's 440-sample trim equals raw minus granule there).
**The ALSA backend**: the engine's own loader (dlopen of the platform's
libasound.so.2, no third-party binding), `snd_pcm_open` on the
"default" PCM, `snd_pcm_set_params` with the native-endian S16
interleaved layout, `snd_pcm_writei` feeding the shared mix loop's
frames with `snd_pcm_recover` for the EPIPE and the ESTRPIPE underruns,
and `snd_pcm_close` in the `Drop`. Gated behind
`#[cfg(all(unix, feature = "audio-alsa"))]`, the same rule the X11
backend carries.
**The WASAPI backend**: the engine's own loader (LoadLibraryA of the
platform's ole32.dll and mmdevapi.dll, the D3D12 loader's own surface),
`CoInitializeEx` and `CoCreateInstance` of the device enumerator, the
default render endpoint, the shared-mode `IAudioClient` with the
interleaved s16 format and the auto-convert flags, the polling write
model (`GetCurrentPadding` names the device's filled frames, the
backend writes what the buffer has room for and sleeps while it is
full), and the release/teardown in reverse in the `Drop`. The COM vtable
slots and the GUIDs are the published ABI the D3D12 loader verifies
against. Gated behind `#[cfg(all(windows, feature = "audio-wasapi"))]`.
**The CoreAudio backend**: the engine's own loader (dlopen of the
system's CoreAudio framework), the default-output AudioComponent, the
AudioUnit instance, the interleaved s16 stream format at the config's
rate and channels, and the render callback the OS's render thread pulls
the frames through (the pending queue's cap keeps the latency bounded;
the short fill renders the honest silence). Gated behind
`#[cfg(all(target_os = "macos", feature = "audio-coreaudio"))]`.
**The AAudio backend**: the engine's own loader (dlopen of the
platform's libaaudio.so, the ALSA loader's own surface),
`AAudio_createStreamBuilder` with the output direction, the shared
mode, the interleaved s16 format at the config's rate and channels and
the device buffer at four periods, `AAudioStreamBuilder_openStream` and
the builder's delete, `AAudioStream_requestStart`, and the blocking
`AAudioStream_write` pushing each mixed block (the same polling-write
model the WASAPI and the ALSA backends carry; the engine keeps no
callback thread of its own), and `AAudioStream_close` in the `Drop`.
The dlopen surface keeps the build's minSdk at 24: the library exists
from API 26 onward, and the machines before it report the open error
and stay on the null path (the same graceful-skip rule the CoreAudio
backend follows when the device is missing). Gated behind
`#[cfg(all(target_os = "android", feature = "audio-aaudio"))]`.
**The integration test** (`audio_smoke`): the null walk (the contract's
voices, the placements, the listener, the deterministic mix and its
pinned peaks), the WAV decode (44.1 kHz mono, 44100 frames), the Vorbis
decode (48 kHz mono, 24000 frames, the peak 0.1316 pinned) and — when the
platform's audio feature is enabled and the device opens — the real
mix through the ALSA, the WASAPI, the CoreAudio or the AAudio backend's
own loop
(the same mix arithmetic every backend runs). The test vectors embed
with `include_bytes!` so the example runs standalone on a device
without the source tree.
The validation runs, all through the same contract:
- **The headless suite**: the og2-core suite (73 tests: the contract
  walk, the decoders' unit tests, the two reference vectors against the
  real libvorbis's PCM) passes with the `audio-vorbis` and the
  `audio-alsa` features; the docs suite passes.
- **The ALSA hardware run** (Linux x86_64, the LucDev, the default PCM
  device): `describe: alsa sample_rate=44100 channels=2 period=512`,
  `alsa mixed frames: 12288`, `alsa violations: 0` — the sound left the
  speakers.
- **The Wine cross-build** (x86_64-pc-windows-gnu): the WASAPI path's
  full COM stack opens under Wine —
  `describe: wasapi sample_rate=44100 channels=2 period=512 buffer=2048`,
  `wasapi mixed frames: 12288`, `wasapi violations: 0` — and no ALSA
  line appears (the cfg gate holds under the Windows target).
- **The Windows hardware run** (DESKTOP-DFS7FIQ, Intel UHD Graphics,
  the rustup GNU toolchain): the WASAPI backend mixes through the real
  driver — the same `buffer=2048` describe, `12288` frames,
  `violations: 0` — re-confirmed on the committed 10c tree.
- **The macOS hardware run** (the A18 Pro): the CoreAudio backend
  mixes through the default output unit —
  `describe: coreaudio sample_rate=44100 channels=2 period=512`,
  `coreaudio mixed frames: 12288`, `coreaudio violations: 0`.
- **The Android decoder run** (the Galaxy A56, arm64, the NDK
  30.0.16248370 through the `aarch64-linux-android24` linker): the full
  cycle decodes both vectors on the real arm64 hardware with the
  embedded vectors and pins the same 0.1316 peak.
- **The Android hardware run** (the same Galaxy A56, Android 8.0's
  AAudio onward): the AAudio backend mixes through the platform's own
  stream —
  `describe: aaudio sample_rate=44100 channels=2 period=512`,
  `aaudio mixed frames: 12288`, `aaudio violations: 0` — on the
  committed 10e tree, with the ALSA regression re-run clean beside it
  (the same 12288 frames, `violations: 0`, the 73-test suite passing).
**The run's own findings, fixed in the same milestone**: the Android
run's first attempt aborted on the missing test-vector files (the
device has no source tree), fixed by embedding the vectors in the
example binary (93 KB, ~800 KB of binary); the LucDev's rustc exposed
the MDCT functions' `#[allow(non_snake_case)]` attached only to the
containing type instead of each function, fixed by moving the
attribute onto every function; and the cross-compiles for the
windows-gnu and the aarch64-apple-darwin targets exposed the ALSA
module compiling outside its unix gate, fixed by the
`#[cfg(all(unix, feature = "audio-alsa"))]` the module now carries.
The WASAPI and the CoreAudio backends' own hardware runs (the 10c
campaign) found the ABI's own traps: the CoreAudio's first run aborted
on the default-output component's find — the engine carried the
OfflineEffect's FourCC (`auol`) where the output unit's type
(`auou`) belongs, a finding the standalone diagnostic probe
(`dbg_ca`) pinned by walking the component registry's own answers —
and then persisted on the unknown-property error (`who?`) until the
engine stopped passing FourCC codes where the AudioUnit's properties
carry numbered enums (`kAudioUnitProperty_StreamFormat` is 8,
`SetRenderCallback` is 23), with the scopes' own values
(Global = 0, Input = 1, Output = 2) verified against the SDK's headers
alongside. The committed tree re-ran clean on all four platforms.
---
## Lessons carried from OpenGrace

The five lessons recorded at the close of OpenGrace's first full campaign
shape the milestone plan above:

1. Fewer middlewares, preferably none.
2. No proprietary middleware.
3. Memory safety at the boundary, enforced by the language.
4. Hardware runs are the only honest validation.
5. Pin the environment, not the machine.

The full statements are in the README and in the architecture document.

---

*OpenGraceII is Apache-2.0 licensed. See LICENSE in the repository.*
