# OpenGraceII

An independent, memory-safe engine written in Rust, shaped by the lessons
of OpenGrace — the source-only reimplementation of the GRACE 2 engine
architecture (github.com/mateusbentes/OpenGrace).

OpenGraceII is not a port of OpenGrace: the code is new, and it starts
from the lessons that the OpenGrace project recorded instead of from its
code.

## Status

The repository is at the start of its life. There is no engine code yet.
The first milestones will define the architecture, the build system and
the graphics layer. This README records the intent so the project does
not lose its bearings while it grows.

## The lessons from OpenGrace

OpenGrace is a complete C++ reimplementation of the GRACE 2 engine
architecture, validated on hardware across Linux, Windows (under Wine),
macOS and Android. It worked, and its hardware run found ten real defects
that the desktop validation never surfaced. The lessons that shape
OpenGraceII:

1. **Minimal middlewares — preferably none.** Every middleware adds a
   dependency, a build system and a licence surface. OpenGraceII keeps
   the middlewares to the minimum the design actually requires.
2. **No proprietary middlewares.** Proprietary middleware blocks the
    licence and the reproducibility of the whole engine. OpenGraceII is
    Apache-2.0 and depends on nothing proprietary. Platform graphics
    APIs (Vulkan, Metal, Direct3D) are not middlewares: they are the
    OS's own interfaces, and OpenGraceII reaches them through wgpu's
    public backend abstraction instead of linking a vendor SDK.
3. **Memory-safe languages at the boundary.** OpenGrace's most subtle
   defects came from the hand-written binding layers (a loader with a
   wrong entry point signature, a command buffer that was ended twice,
   and state machine errors that only one hardware driver surfaced).
   OpenGraceII is written in Rust, so the memory safety of the boundary
   is enforced by the language instead of by audit.
4. **Hardware runs are the only honest validation.** A clean compile and
   a software renderer's pass do not prove the engine works on a phone's
   driver. OpenGraceII plans the hardware validation from the first
   milestone instead of bolting it on at the end.
5. **Pinned environments.** OpenGrace's reproducibility came from a
   pinned toolchain (NDK r27c, SDL2 2.30.12, GCC 13.3.0). OpenGraceII
   pins its toolchain the same way from the start.

## Planned platforms

Linux x86_64, Windows x86_64, macOS arm64, iOS arm64 and Android
arm64-v8a, with the graphics layer planned through wgpu. The
first-class backends are Vulkan (Linux, Android), Metal (macOS, iOS)
and Direct3D 12 (Windows, where it is the most reliable native path
through wgpu). OpenGL is not a target: it is kept only as a fallback
validation backend on Linux for machines without Vulkan, matching the
minimal-middlewares philosophy — no GL-specific features are designed
around it.

## Planned scope

OpenGraceII is planned as a full engine in the spirit of GRACE 2, not
only a renderer. Besides the graphics layer, the milestones will cover:

- **Editor.** A scene and asset editor with a live preview through the
  same wgpu pipeline the game uses.
- **Network tools.** Tools for connection, session and state
  replication, kept as a separate layer so they can run without a
  renderer.
- **Physics.** A fixed-timestep physics subsystem with a memory-safe
  boundary.
- **Lighting.** The full lighting pipeline of the engine (lightmaps,
  dynamic lights and the PBR path the renderer serves).
- **OWD support.** A reader and writer for the OWD format used by the
  engine, plus a standalone compression and decompression tool. The
  format logic is reimplemented from documented behaviour; no
  proprietary data files are committed to the repository.

Every subsystem follows the same lessons: minimal middlewares, no
proprietary dependencies, a Rust-enforced memory-safe boundary and
hardware validation from the first milestone.

## License

Apache-2.0. See LICENSE.
