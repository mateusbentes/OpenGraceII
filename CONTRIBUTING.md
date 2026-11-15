# OpenGraceII — Contributing

Thank you for considering a contribution. This document records how the
project works: what it accepts, how it validates and how it licenses. It
mirrors the structure of the OpenGrace project's contribution rules
(github.com/mateusbentes/OpenGrace), which produced the lessons this
repository is built on.

## 1. The lessons govern every patch

The five lessons from OpenGrace (README.md and docs/grace-architecture.md)
are not history: they are the review criteria for every patch.

1. **Fewer middlewares, preferably none.** A patch that pins a new submodule
   or a new third-party crate is a cost-first decision: it must state the
   build system, the version pin, the licence and the patch queue it
   carries. The engine's own code is the default; the platform APIs
   (Vulkan, Metal, Direct3D 12, OpenGL) are the only external surface the
   engine cannot replace.
2. **No proprietary middleware.** No patch may introduce a component whose
   licence, distribution or buildability the project does not control.
3. **Memory safety at the boundary.** The engine is written in Rust. A
   patch may use `unsafe` only where the boundary genuinely requires it,
   and every `unsafe` block must state the invariant it relies on.
4. **Hardware runs are the only honest validation.** A render milestone is
   not complete until it has run on real hardware: Linux, Windows (under
   Wine), macOS and Android. A clean compile and a software renderer's pass
   do not close a milestone.
5. **Pin the environment, not the machine.** The toolchain is pinned
   (`rust-toolchain.toml`), the NDK and SDK versions are recorded in the
   build documentation, and the validation matrix is the same on every
   machine.

## 2. Building and validating

The build system is Cargo. Each milestone leaves the tree buildable and the
validation matrix green:

- `cargo build` on the default target (Linux x86_64).
- `cargo test` on the same target, including the documentation structure
  test (`docs_structure`), which checks the rules of section 4.
- Cross-platform builds: the MinGW-w64 cross-compile for Windows, the NDK
  build for Android arm64-v8a, and the Metal build for macOS arm64 (the
  Metal device checks run on macOS only).
- Documentation: regenerate the .docx renderings with pandoc after every
  markdown change, and run the documentation structure test.

A patch is not accepted until it has been validated on every platform the
project claims to support. The validation matrix is recorded in
docs/implementation-status.md, one row per milestone.

## 3. The patch process

Patches are plain text diffs, one patch per round:

- `git format-patch origin/main` is the preferred format; a single patch
  per message keeps the review honest.
- The patch must apply cleanly on the published tree, and it must state
  what it changes and what it validates.
- The project owner commits and pushes. The repository's main branch is
  not rewritten by default, and the published tree is the only tree.
- The validation matrix is recorded in the patch message and in
  docs/implementation-status.md.

## 4. Documentation rules

The documentation is written in English and follows the rules of the
OpenGrace project, because the documentation is the only honest record of
the engine's state:

- **Line length.** Markdown files are limited to 79 characters per line
  (the same rule the OpenGrace repository enforces). Table rows are
  exempt, because the content is the column.
- **No double blank lines.** One blank line separates paragraphs.
- **No dangling references.** Every reference must be complete: a file
  path, a milestone number, a decision number or a repository URL. If a
  reference cannot be completed, it is removed.
- **No proprietary material.** The provenance rules of section 6 of
  docs/grace-architecture.md apply: no SDK file, header, excerpt, symbol
  or translation may be committed, and no proprietary data files are
  committed to the repository.
- **Honesty.** A build-only validation is recorded as build-only. A
  hardware run is recorded with its hardware, its numbers and its
  defects. Never claim a milestone that was not validated.

The documentation structure test (`docs_structure`) enforces the first
two rules mechanically.

## 5. Licensing

- **The engine is Apache-2.0.** The licence text is the canonical Apache
  License 2.0 in the LICENSE file.
- **The documentation is Apache-2.0.** The same licence covers the docs
  and the README.
- **No proprietary material.** A patch may not commit material the
  original rights holder never licensed for redistribution: the GRACE 2
  SDK headers, libraries or content derived from them, and any
  non-public documentation. Public documentation is cited; private
  material is not.
- **No heavy binaries or data files.** Keystores, build outputs, SDK
  directories and proprietary data files are committed to .gitignore, not
  to the repository.

## 6. Where to start

The milestone plan is in docs/grace-architecture.md, section 7. The current
status is in docs/implementation-status.md. The repository is at the start
of its life: the first milestone is the Rust skeleton, the toolchain pin
and the validation matrix, and the first render milestone carries the
Android hardware validation from the start.

---

*OpenGraceII is Apache-2.0 licensed. See LICENSE in the repository.*
