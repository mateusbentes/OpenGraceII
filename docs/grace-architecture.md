# OpenGraceII — Architectural Reference

Working specification for an independent, memory-safe engine written in Rust,
shaped by the lessons of OpenGrace (github.com/mateusbentes/OpenGrace).
Apache-2.0 licensed.

This document is compiled **exclusively from publicly available sources**. It
contains no proprietary material, no extracted data, and no content derived
from non-public software. See "Sources" at the end for the full provenance
list.

---

## 1. What OpenGraceII is

OpenGraceII is a new engine, written in Rust, that starts from the lessons
OpenGrace recorded instead of from its code. It is **not a port**: the code is
new, and it implements the whole engine in new code.

The relationship between the projects:

- OpenGrace reimplemented the **architecture** of the GRACE 2 engine
  (github.com/mateusbentes/OpenGrace) — the renderer, physics, scripting,
  animation, scene, audio, UI, AI and video of the GRACE 2 engine — and
  closed its platform matrix with hardware validation on Linux, Windows
  (under Wine), macOS and Android.
- OpenGraceII is the engine the OpenGrace project set out to build: the same
  architecture scope, plus the parts OpenGrace deferred (Editor, network
  tools), and with the middlewares replaced by the engine's own
  implementations.

The GRACE 2 architectural baseline is the same provenance record of OpenGrace
(grace-architecture.md in the OpenGrace repository). OpenGraceII does not
reproduce that record: it implements the engine.

## 2. The lessons from OpenGrace

The five lessons recorded at the close of OpenGrace's first full campaign
(milestone 23, decision 52) shape every decision in this document:

1. **Fewer middlewares, preferably none.** Every pinned submodule is a build
   system, a version pin, a licence and a patch queue to maintain for the life
   of the project. OpenGraceII implements each subsystem in the engine's own
   code and treats each middleware as a cost first.
2. **No proprietary middleware.** A proprietary component can disappear from
   distribution, change its licence, or stop building on the toolchain the
   project needs. OpenGraceII depends on nothing proprietary. Platform
   graphics APIs (Vulkan, Metal, Direct3D 12, OpenGL) are not middlewares:
   they are the operating systems' own interfaces, and OpenGraceII reaches
   them through the engine's own loaders.
3. **Memory-safe languages at the boundary.** OpenGrace's most subtle defects
   came from its hand-written binding layers (a loader with a wrong entry
   point signature, a command buffer that was ended twice, a swapchain
   recreated with a stale extent). OpenGraceII is written in Rust, so the
   memory safety of the boundary is enforced by the language instead of by
   audit.
4. **Hardware runs are the only honest validation.** A clean compile and a
   software renderer's pass do not prove the engine works on a phone's
   driver. OpenGraceII plans the hardware validation from the first render
   milestone instead of bolting it on at the end.
5. **Pin the environment, not the machine.** OpenGraceII pins its toolchain
   (the Rust toolchain, the NDK, the SDKs, the build tools) from the first
   milestone, so every machine reproduces the same validation.

## 3. Platform matrix

| Platform | Graphics APIs |
| --- | --- |
| Windows x86_64 | OpenGL, Vulkan, Direct3D 12 |
| Linux x86_64 | OpenGL, Vulkan |
| macOS arm64 | Metal |
| iOS arm64 | Metal |
| Android arm64-v8a | Vulkan |

Each backend is behind a build option, exactly as OpenGrace (`GRACE_ENABLE_*`
becomes a Cargo feature). The backends are the engine's own code — not a
middleware abstraction. Direct3D 12 is the Windows platform API (not a
middleware): on Windows it is the most reliable native path, and the engine
reaches it through its own loader.

## 4. Modules

| Module | Responsibility |
| --- | --- |
| Core | Object model, resource manager, threading, math, curves, profiler |
| Std | Standard-library layer, containers, strings, localisation |
| Graphics | Renderer, scene graph, materials, shadows, post-processing |
| GraphicsShaders | Shader sources and variants |
| Physics | Own physics: the classical layer and the extra layer |
| Script | The engine's own scripting language and visual scripting system |
| Audio | Own audio: 2D/3D playback, streaming, reverb |
| Video | Own video playback service |
| Motion | Animation, state machines, blending |
| AI | Agent behaviour |
| GUI | Retained UI system, fonts, unicode input |
| Game | Gameplay-level services |
| App | Application lifecycle, platform entry points |
| Tools | The Editor and the network tools (OpenGrace's deferred scope) |

## 5. The middleware replacement plan

OpenGrace's middlewares become the engine's own implementations in Rust. The
table records what changes and what stays:

| OpenGrace dependency | OpenGraceII approach |
| --- | --- |
| SDL2 | Own platform layer: windowing, input, events, threads. The
  per-platform layer is the engine's own code (Win32, X11/Wayland, Cocoa,
  Android NDK, iOS UIKit). |
| OpenAL Soft | Own audio backend per platform, implementing the same
  audio contract (the OpenGrace backend layer). |
| FreeType | Own glyph rasterisation. |
| libtheora / libvorbisfile | Own Ogg Vorbis and Theora decoders. |
| zlib | Own Deflate. |
| AngelScript | The engine's own scripting language and VM. |
| Tetragramaton (VAS) | The engine's own visual scripting system, built on
  the engine's own language. |
| Bullet / PhysX 3.4 / Box2D | Own physics: the classical layer (rigid
  bodies, joints, buoyancy, force fields) and the extra layer (cloth, soft
  body, particles) implemented in the engine's own code. |
| glslangValidator | Own shader toolchain: the engine's own shader language
  and its per-backend compilers (see Open decisions). |
| wgpu | Not used. OpenGraceII implements its own loaders and backends
  (OpenGL, Vulkan, Metal, Direct3D 12), as OpenGrace did. |

The rule is the same as OpenGrace's lesson 1: the engine's own code is the
first choice, and the platform APIs are the only external surface the engine
cannot replace.

## 6. Licensing boundary

This section carries OpenGrace's provenance rules over unchanged, so the
boundary is explicit rather than assumed:

- **What is public and usable.** The feature documentation, the licence terms,
  the ModDB entry, the developer's forum posts, and the AngelScript project's
  list of applications are all public. Architectural facts drawn from them —
  module decomposition, subsystem responsibilities, backend choices, rendering
  techniques — may be described and implemented freely.
- **What is not.** The GRACE 2 SDK was distributed under a free
  **non-commercial** licence. Its headers, libraries, and any content derived
  from them may not be incorporated into this project, redistributed, or used
  as an implementation source. OpenGraceII is Apache-2.0, which grants
  unrestricted commercial rights to recipients; that grant cannot be extended
  to material the original rights holder never licensed for redistribution.
- **Consequences for this repository.** No SDK file, header, excerpt, symbol,
  or translation may be committed. Architectural knowledge from the SDK stays
  out of the public tree. Public documentation is cited; private material is
  not.

## 7. Milestone plan (draft)

Sequenced so that each milestone is independently testable and leaves the
tree buildable. The draft is for the project owner to confirm:

1. **Rust skeleton.** Cargo workspace, module layout, the validation matrix
   (build, test, docs_structure-equivalent), the pinned toolchain.
2. **Render contract and the first backend.** Vulkan on Linux (lavapipe) and
   Android hardware — the lessons demand hardware validation from the first
   render milestone, not the last.
3. **The remaining backends.** OpenGL (Windows/Linux), Metal (macOS/iOS),
   Direct3D 12 (Windows).
4. **Render foundation.** Math, mesh, shader, material.
5. **Forward lighting.** Point, spot and directional lights; shadow mapping.
6. **Deferred lighting.** G-buffer, unlimited lights, light volume rendering.
7. **Post-processing chain.** HDR, bloom, filmic tonemapping, depth of field
   with bokeh, morphological anti-aliasing, SSAO-STF.
8. **Global illumination.** Light volume propagation.
9. **Platform layer.** The engine's own windowing, input and events, replacing
   SDL2.
10. **Audio.** The own audio backend per platform, the WAV and Ogg Vorbis
    decoders.
11. **Physics.** The classical layer and the extra layer.
12. **Scripting.** The engine's own scripting language and VM.
13. **Visual scripting.** The graph representation on the engine's own
    language.
14. **Animation.** Skeletal clips, the deterministic sampler, the state
    machine, the controllers.
15. **Scene management.** Data-driven entities, spatial partitioning, scene
    queries, LOD.
16. **UI.** The retained control tree, the own glyph rasterisation, unicode
    input.
17. **AI and gameplay.** The agent behaviours and gameplay services.
18. **Video.** The own playback service and the Ogg Theora decode layer.
19. **OWD.** The reader, the writer, the filters, the checksums, the overlay
    loader and the standalone compression/decompression tool.
20. **Windows validation.** The MinGW-w64 cross-compile (or the Windows
    toolchain) and the Windows backends.
21. **Editor.** The scene and asset editor with a live preview through the
    same render pipeline.
22. **Network tools.** Tools for connection, session and state replication.
23. **Hardware validation.** The full platform matrix on hardware.

## 8. Open decisions

- **Which graphics backend first?** Vulkan first would deliver Android
  hardware validation from the first render milestone; OpenGL first would be
  the simplest path on Linux. OpenGraceII can implement both, but the order
  matters for the hardware-validation lesson.
- **The shader toolchain.** Vulkan needs SPIR-V. OpenGrace used
  glslangValidator at build time. OpenGraceII can use a Rust-native SPIR-V
  toolchain (for example, naga) or a build-time compiler. The decision is a
  build-time dependency rule, not a runtime middleware.
- **The scripting language.** The engine's own scripting language and VM is
  written in Rust, replacing AngelScript and the visual scripting system.
  The language design is a project of its own.
- **The physics extra.** The GRACE 2 physics features (cloth, soft body,
  particles, buoyancy) are the engine's own implementation — the classical
  layer is the same.
- **The editor and the network tools.** The OpenGrace deferred scope is now
  the project's own scope: the editor and the network tools are part of the
  engine's own code.
- **SquareLine Studio.** The visual editor tool is a planned scope for
  OpenGraceII; the port is deferred until the engine's own editor is far
  enough.

## 9. Sources

The GRACE 2 architectural baseline is the same provenance list of OpenGrace's
grace-architecture.md
(github.com/mateusbentes/OpenGrace/blob/main/docs/grace-architecture.md).

---

*OpenGraceII is Apache-2.0 licensed. See LICENSE in the repository.*
