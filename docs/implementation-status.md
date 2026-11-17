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
| 3 | The remaining backends: OpenGL (Windows/Linux), Metal (macOS/iOS), Direct3D 12 (Windows) | In progress (OpenGL done, Metal implemented, Direct3D 12 validated under Wine with vkd3d; the Metal hardware run follows) |
| 4 | Render foundation: math, mesh, shader, material | Planned |
| 5 | Forward lighting: light types, shadow mapping | Planned |
| 6 | Deferred lighting: MRT G-buffer, the lighting pass | Planned |
| 7 | Post-processing chain: HDR, bloom, filmic tonemapping, depth of field, morphological anti-aliasing, SSAO-STF | Planned |
| 8 | Global illumination by light volume propagation | Planned |
| 9 | Platform layer: the engine's own windowing, input and events, replacing SDL2 | Planned |
| 10 | Audio: the engine's own backend per platform, the WAV and Ogg Vorbis decoders | Planned |
| 11 | Physics: the classical layer and the extra layer, in the engine's own code | Planned |
| 12 | Scripting: the engine's own scripting language and VM | Planned |
| 13 | Visual scripting: the graph representation on the engine's own language | Planned |
| 14 | Animation: skeletal clips, the deterministic sampler, the state machine, the controllers | Planned |
| 15 | Scene management: data-driven entities, spatial partitioning, scene queries, LOD | Planned |
| 16 | UI: the retained control tree, the own glyph rasterisation, unicode input | Planned |
| 17 | AI and gameplay: the agent behaviours and the gameplay services | Planned |
| 18 | Video: the own playback service and the Ogg Theora decode layer | Planned |
| 19 | OWD: the reader, the writer, the filters, the checksums, the overlay loader and the standalone compression/decompression tool (docs/resource-container.md) | Planned |
| 20 | Windows validation: the MinGW-w64 cross-compile (or the Windows toolchain) and the Windows backends | Planned |
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
