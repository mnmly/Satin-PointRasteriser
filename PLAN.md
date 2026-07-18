# Satin-PointRasteriser — Implementation Plan

> **Status (2026-07-16): all 7 slices implemented and green (42/42 tests).**
> See README.md / API.md for current docs and benchmark numbers. Notable outcomes:
> - 64-bit atomics work on M5 Max via void-return `atomic_max_explicit`
>   (`__HAVE_ATOMIC_ULONG_MIN_MAX__`); the sibling's old failure used the
>   unsupported `fetch_` form. Nearest mode single-pass 64-bit is ~10% faster
>   than the two-pass fallback.
> - SIMD-group pre-aggregation: −11% HQ frame time; default on.
> - Exact VAST 2011 cone rejection proved stable (no fallback needed).
> - Baseline vs sibling (10M pts, M5 Max): full-sweep default HQ is ~1.5–1.7×
>   slower than the sibling's direct-cull (extra per-frame compaction pass);
>   amortized LOD (steady state) and nearest-64 match or beat it.
>   RESOLVED: LODSelect now skips on static (camera, viewport, selection
>   config, cloud contentGeneration, world transform) — static HQ 1.85→1.01 ms,
>   nearest 1.07→0.32 ms at 10M pts; below sibling parity.
> - GPU packer ported (bit-identical to CPU pack incl. batch shuffle;
>   requires mathMode .safe): San Marco 43.9M-pt PLY load 125.4 s → 1.18 s.
> - COPC streaming (slices 9–10 + SwiftPDAL 1.23.0): coarse-pin coverage,
>   decode gate + pack-workspace fixes — time-to-detail 4.7 s → 91 ms release.
> - Slice 8 (2026-07-16): COPC/SwiftPDAL streaming landed — slot-pool residency in
>   PointRasteriserPointCloud (mutations join the next eligible sweep; no-partial-sweep
>   guarantee preserved) + SatinPointRasteriserStreaming target with the ported
>   StreamingAdapter. Verified against a real 2.7GB / 396.6M-point COPC.

## Context

A new high-performance point-cloud rasteriser for Satin, architected after the Magnopus
article "How we render extremely large point clouds" (saved locally as
`ref_How we render extremely large point clouds — Magnopus.html`), which synthesizes four
papers (Schütz 2019/2021/2022 + VAST 2011 screen-space operators).

The sibling project `../Satin-ComputeRasteriser` already implements a large subset
(quantized 10/20/30-bit batches, CLOD survivor-prefix, cull → indirect depth/color →
resolve → composite, reverse-Z uint-atomic depth). This project supersedes it with the
reference's missing pieces, while **preserving** its extensibility features (TintPass,
DisplacementPass, picking, point-size modes, etc.). Proven code is copy-adapted from the
sibling, not reinvented.

Target hardware includes **M5 Max** (Apple GPU family 9+): 64-bit buffer atomics
(`atomic_ulong` min/max, MSL 3.1) are used where they help, gated at runtime on
`device.supportsFamily(.apple9)` with a portable 32-bit fallback.

## What's new vs. Satin-ComputeRasteriser

| Reference technique | Status in sibling | This project |
|---|---|---|
| Amortized, double-buffered LOD compaction ("LOD point cloud") | absent (per-frame cull only) | core new pass |
| Point rejection (7×7 cone occlusion) merged with resolve | absent | new pass |
| Batch shuffle after Morton sort (inter-wave contention) | absent | added to packer |
| 64-bit packed depth+index atomics | rejected on old HW | apple9+ fast path |
| SIMD-group pre-aggregation before atomics | designed, never landed | implemented + benchmarked |

## Architecture

Swift package `SatinPointRasteriser`, mirroring the sibling's structure:
- Satin dep: `Fabric-Project/Satin` exact `20.0.0-Beta-1` (same as sibling).
- Platforms: macOS 15 / iOS 18, swift-tools 6.0. Strict concurrency.
- `Pipelines/` copied resource bundle, kernels compiled at runtime via `Bundle.module`
  (same pattern as sibling; keeps MetalFileCompiler live-reload working).
- Core types (`RasterBatch`, `PackedPointCloud`, 5-buffer quantized layout, decode
  helpers) copy-adapted from sibling — the byte layouts are proven and shared with the
  Tint/Displacement ABI.

### Per-frame pipeline (high-quality average mode)

1. **LODSelect (compute, amortized)** — per source batch: frustum cull + pixel-footprint
   precision level; per surviving point: CLOD threshold + dither weight → append
   `{position (object-space packed_float3), color uint, sourceIndex uint}` into a
   compacted SoA **LOD buffer** (per cloud). Amortized: process ≤ budget points/frame,
   resume cursor next frame; **double-buffered** — swap only when a full sweep completes.
   Non-amortized full-sweep mode first (slice 2), amortization in slice 5.
2. **Clear** — zero screen-sized `RasterPixel` accumulation buffer.
3. **DepthPass** — project LOD-buffer points (displacement via `sourceIndex` if enabled),
   splat footprint, `atomic_fetch_max` reverse-Z uint depth.
   *apple9+ nearest mode*: single pass, `atomic_max` on `ulong(depthUint<<32 | lodIndex)`
   replaces depth+index passes entirely.
4. **ColorPass** — re-project, epsilon depth test, `atomic_fetch_add` RGB+count
   (tint via `sourceIndex`; OIT/coverage + motion-blur variants preserved from sibling).
5. **Reject+Resolve (merged)** — per pixel: 7×7 neighborhood cone-occlusion test
   (VAST 2011); survivors resolve (Σcolor/count) directly into output + depth textures.
6. **HoleFill** — 3 iterations of neighbor-average expansion (color + depth), ping-pong.
7. **Composite** — Satin `SourceMaterial`+`PostProcessEncoder`, depth-aware
   (`writesSceneDepth`, reverse-Z `.greaterEqual`) or always-on-top, ported from sibling.

### M5 Max / capability gating

- `RasteriserCapabilities` probe at init: `supportsFamily(.apple9)` → `use64BitAtomics`;
  compile kernels with function constants so both variants come from one source.
- Empirical probe test: tiny kernel doing `atomic_fetch_max_explicit` on `atomic_ulong`,
  validated in unit test on this machine (sibling's old failure predates apple9 HW).
- SIMD-group aggregation (simd_shuffle pixel-match + lane reduction before the atomic) is
  benchmarked against plain atomics before being enabled by default (slice 4).

### Preserved features (ported from sibling, same public contracts)

- **TintPass / DisplacementPass** — identical preamble + buffer-slot ABI (slots 0–15),
  MetalFileCompiler live reload; indices remain pack-order `sourceIndex` so user kernels
  are unchanged. NaN-displacement cull sentinel, `tint.a` semantics, coverage/OIT mode.
- Point picking (`pickPointIndex`), point-size modes (screen/world), min/max/scale,
  colorizeChunks/colorizeOverdraw debug, background color, motion blur, hole-fill count.
- PLY loader; `PackedPointCloudFixtures` (+ **batch shuffle** step added: swap
  every-other batch with mirror-from-end, constant time, per Magnopus).
- COPC/SwiftPDAL streaming: **deferred follow-up** (slot-pool design carries over; not in
  initial slices).

## Slices (each ends green: build + tests)

1. **Scaffold + data model** — Package.swift, types, CPU packer (+shuffle) with tests,
   capability probe incl. 64-bit atomic runtime test. [sonnet]
2. **Minimal pipeline end-to-end** — full-sweep LODSelect → depth → color → resolve →
   composite; example app (SwiftUI + SatinMetalView) renders fixture cube grid. [opus]
3. **Reject+Resolve merged + HoleFill.** [opus kernels]
4. **Fast paths** — 64-bit nearest mode, SIMD-group aggregation, benchmark harness
   comparing modes on 10M+ fixture. [opus]
5. **Amortized LOD + double buffering.** [opus]
6. **Tint/Displacement/picking/motion-blur ports + PLY loader.** [sonnet]
7. **Example-app parity** (settings UI, DoF recipe) + docs. [sonnet]

## Verification

- Unit tests: packer round-trip decode (CPU reference), layout byte-compat tests,
  64-bit atomic probe, LODSelect determinism, reject/hole-fill golden images on fixture.
- `xcodebuild` build + test per slice (Metal involved).
- Example app manual run on M5 Max: fixture + a real PLY; Instruments GPU capture for
  before/after pass timings; overdraw debug view for contention sanity.
- Baseline comparison: same PLY in sibling app vs. this app (frame time, VRAM).
