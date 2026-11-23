# OpenGraceII

An independent engine written in Rust, shaped by the lessons of OpenGrace —
the source-only reimplementation of the GRACE 2 engine architecture
(github.com/mateusbentes/OpenGrace).

OpenGraceII is not a port of OpenGrace: the code is new, and it starts
from the lessons that the OpenGrace project recorded instead of from its
code.

## Status

Milestone 1 (the Rust skeleton) is done: the Cargo workspace builds, the
module layout is in place, the toolchain is pinned (rust-toolchain.toml),
and the validation matrix runs. Milestone 2 (the render contract and the
first backend) is done: the API-neutral render contract, the deterministic
null backend and the Vulkan backend (the engine's own loader, no third-
party Vulkan binding), validated on Linux with lavapipe; the Android
hardware run follows. The running record is docs/implementation-status.md.

Milestones 3 through 8 are done: the four graphics backends (the engine's
own OpenGL, Vulkan, Metal and Direct3D 12 loaders) plus the render
foundation (the engine's own math, mesh and shader material paths), the
forward lighting (the Blinn-Phong path with the sun and up to four point
lights), the deferred lighting (the three-attachment G-buffer and the
fullscreen lighting pass), the post-processing chain (the HDR geometry
pass and the five-effect composite) and the global illumination by light
volume propagation (the radiance volume, the post chain's indirect term
and the set_light_volume texture cache) all run on all four backends —
Vulkan and OpenGL on Linux hardware, Direct3D 12 under Wine with vkd3d,
Metal on the A18 Pro's hardware run. The engine's own lighting module,
the post chain's reference math and the light volume's packing,
injection, propagation and sampling carry their unit tests, and every
milestone's integration test pins the pixels its own reference produces.

Milestone 9 (the platform layer) is done: the engine's own X11 windowing,
input and event path runs on Linux hardware, the windowed Vulkan
swapchain presents through the same contract, and the full D3D12 suite
now runs on real Windows hardware (Intel UHD Graphics, Windows 10)
through the repository's run-tests.bat runner — the D3D12 path's
validation no longer rests on Wine alone.
Milestone 10 (the audio foundation) is done: the engine's own audio
contract and deterministic null backend, the engine's own WAV and Ogg
Vorbis decoders (no OpenAL, no SDL2_audio) and the three real backends
— ALSA through dlopen, WASAPI through the platform's own COM surface
and CoreAudio through the system's framework — run on all four
platforms, each mixing through the same loop with zero violations: the
ALSA mix on Linux hardware, the WASAPI mix on the Windows hardware
(Intel UHD) and under Wine, and the CoreAudio mix on the A18 Pro's
macOS run (the decoders' path also runs on the Galaxy A56's arm64 NDK
run; the AAudio backend follows with the Android minSdk decision).

## The lessons from OpenGrace

OpenGrace is a complete C++ reimplementation of the GRACE 2 engine
architecture, validated on hardware across Linux, Windows (under Wine),
macOS and Android. It worked, and its hardware run found ten real defects
that the desktop validation never surfaced. The lessons that shape
OpenGraceII:

1. **Fewer middlewares, preferably none.** Every pinned submodule is a
   build system, a version pin, a licence and a patch queue to maintain
   for the life of the project. OpenGraceII implements each subsystem in
   the engine's own code and treats each middleware as a cost first.
2. **No proprietary middlewares.** Proprietary middleware can disappear
   from distribution, change its licence, or stop building, and the
   project cannot fix any of that. OpenGraceII is Apache-2.0 and depends
   on nothing proprietary. Platform graphics APIs (Vulkan, Metal, Direct3D
   12, OpenGL) are not middlewares: they are the operating systems' own
   interfaces, and OpenGraceII reaches them through the engine's own
   loaders.
3. **A memory-safe language.** OpenGrace's most subtle defects came from
   its hand-written binding layers (a loader with a wrong entry point
   signature, a command buffer that was ended twice, a swapchain
   recreated with a stale extent). OpenGraceII is written in Rust, so the
   memory safety of the boundary is enforced by the language instead of
   by audit.
4. **Hardware runs are the only honest validation.** A clean compile and
   a software renderer's pass do not prove the engine works on a phone's
   driver. OpenGraceII plans the hardware validation from the first
   render milestone instead of bolting it on at the end.
5. **Pin the environment, not the machine.** OpenGraceII pins its
   toolchain from the first milestone, so every machine reproduces the
   same validation.

## Platforms

| Platform | Graphics APIs |
| --- | --- |
| Windows x86_64 | OpenGL, Vulkan, Direct3D 12 |
| Linux x86_64 | OpenGL, Vulkan |
| macOS arm64 | Metal |
| iOS arm64 | Metal |
| Android arm64-v8a | Vulkan |

Direct3D 12 is the Windows platform API (not a middleware): on Windows it
is the most reliable native path. The backends are the engine's own code
in Rust, not a middleware abstraction — the same approach OpenGrace took
with its own loaders.

## Planned scope

OpenGraceII is planned as a full engine in the spirit of GRACE 2, not
only a renderer. Besides the graphics layer, the milestones will cover:

- **Editor.** A scene and asset editor with a live preview through the
  same render pipeline the game uses.
- **Network tools.** Tools for connection, session and state
  replication, kept as a separate layer so they can run without a
  renderer.
- **Physics.** The classical layer (rigid bodies, joints, buoyancy,
  force fields) and the extra layer (cloth, soft body, particles),
  implemented in the engine's own code.
- **Lighting.** The full lighting pipeline: forward and deferred paths,
  shadow maps, HDR, bloom, tonemapping, depth of field, morphological
  anti-aliasing, SSAO-STF and global illumination by light volume
  propagation.
- **Audio.** The engine's own audio backend per platform, with the WAV
  and Ogg Vorbis decoders.
- **Video.** The engine's own playback service and the Ogg Theora decode
  layer.
- **Scripting.** The engine's own scripting language and VM, plus the
  visual scripting system built on it.
- **UI.** The retained control tree, the engine's own glyph
  rasterisation and unicode input.
- **OWD support.** A reader and writer for the OWD resource container,
  plus a standalone compression and decompression tool. The format logic
  is reimplemented from documented behaviour; no proprietary data files
  are committed to the repository.
- **Scene management.** Data-driven entities, selectable spatial
  partitioning, scene queries and LOD.
- **Animation.** Skeletal clips, the deterministic sampler, the state
  machine, the node controllers and the blendshapes.

Every subsystem follows the same lessons: minimal middlewares, no
proprietary dependencies, a Rust-enforced memory-safe boundary and
hardware validation from the first render milestone.

## License

Apache-2.0. See LICENSE.
