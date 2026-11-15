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
| 1 | Rust skeleton: Cargo workspace, module layout, the validation matrix (build, test, docs checks), the pinned toolchain | Planned |
| 2 | Render contract and the first backend: Vulkan on Linux (lavapipe) and on Android hardware | Planned |
| 3 | The remaining backends: OpenGL (Windows/Linux), Metal (macOS/iOS), Direct3D 12 (Windows) | Planned |
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

Carried from the architecture document's open decisions; each becomes a
decision record when its milestone is specified:

- **Which graphics backend first?** The draft plans Vulkan first, because
  lesson 4 demands hardware validation from the first render milestone.
- **The shader toolchain.** Vulkan needs SPIR-V; the decision is a
  build-time dependency rule, not a runtime middleware.
- **The scripting language.** The engine's own language and VM is a project
  of its own.
- **The physics extra.** The classical layer and the extra layer are both
  the engine's own implementation.
- **The editor and the network tools.** Part of the engine's own scope,
  not a deferred scope.
- **SquareLine Studio.** The port is deferred until the engine's own editor
  is far enough.

---

## Validation record

Empty until the first milestone lands. The record follows the same rule as
OpenGrace: every milestone row records the suites that ran, the counts they
measured, and the hardware that produced them, so the reader can tell what
was proven from what was planned.

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
