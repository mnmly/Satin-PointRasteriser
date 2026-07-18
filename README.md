# Satin Point Rasteriser

Satin Point Rasteriser is a Swift package that renders large point clouds through Satin and
Metal, architected after Magnopus's article ["How we render extremely large point
clouds"](<./ref_How%20we%20render%20extremely%20large%20point%20clouds%20—%20Magnopus.html>)
(synthesizing Schütz 2019/2021/2022 + the VAST 2011 screen-space occlusion operator). It
supersedes [Satin-ComputeRasteriser](../Satin-ComputeRasteriser), preserving that package's
extensibility contracts (`TintPass`, `DisplacementPass`, picking, point-size modes) while
adding the reference's missing pieces:

- **Amortized, double-buffered LOD compaction** — a persistent "LOD point cloud" is rebuilt
  by a resumable per-frame budget instead of a full-sweep cull every frame, so very large
  clouds stay responsive under camera motion.
- **Point rejection merged with resolve** — the VAST 2011 7×7 screen-space cone-occlusion
  operator discards far points that leak through gaps in a nearer surface, in the same pass
  that resolves color/depth.
- **Batch shuffle** after Morton sort, mitigating inter-wave atomic contention.
- **64-bit packed depth+index atomics** on Apple9+ hardware (M5-class GPUs, MSL 3.1), with a
  portable 32-bit two-pass fallback everywhere else, gated by a runtime capability probe.
- **SIMD-group pre-aggregation** before the depth/color atomics, benchmarked against plain
  atomics rather than assumed.

## Pipeline (high-quality average mode)

1. **LODSelect** (compute, amortized) — per source batch: frustum cull + pixel-footprint
   precision level; per surviving point: CLOD threshold + dither weight → append into a
   compacted SoA LOD buffer per cloud.
2. **Clear** the screen-sized accumulation buffer.
3. **DepthPass** — project LOD points, splat footprint, `atomic_fetch_max` reverse-Z depth
   (or the single-pass 64-bit nearest fast path on Apple9+).
4. **ColorPass** — re-project, epsilon depth test, `atomic_fetch_add` RGB+count.
5. **Reject+Resolve** (merged) — 7×7 cone-occlusion test; survivors resolve into output +
   depth textures.
6. **HoleFill** — configurable neighbor-average expansion iterations, ping-pong.
7. **Composite** — Satin `SourceMaterial` + `PostProcessEncoder`, depth-aware
   (`writesSceneDepth`) or always-on-top.

`.nearestPoint` mode replaces steps 3–4 with a single 64-bit packed-atomic winner pass (or a
portable two-pass fallback), then a nearest-mode reject+resolve — no averaging, single cloud.

## Relationship to Satin-ComputeRasteriser

This package is the **successor** to `Satin-ComputeRasteriser`, sharing its core packed
30-bit position / RGBA8 color layout and Metal ABI, its `TintPass`/`DisplacementPass`
sketch-kernel contracts (buffer slots, `SCR_*` macros, `scr_decodePointAt`/`scr_decodeColorAt`
helpers), and its point-picking / point-size / motion-blur / OIT features — existing
sketches written against the sibling compile unchanged here. What's new is the amortized
LOD compaction, merged point rejection, batch shuffle, and the M5/apple9 64-bit + SIMD fast
paths (see the table in [PLAN.md](PLAN.md#whats-new-vs-satin-computerasteriser)).

**Streaming/COPC is supported** via the `SatinPointRasteriserStreaming` product
(SwiftPDAL/copc-lib, C++ interop): `StreamingAdapter` streams COPC nodes in and out of a
fixed slot pool as the camera moves (halo or distance-only residency, point budget), with
per-batch "join next sweep" semantics that preserve the amortized-LOD no-partial-sweep
guarantee. The example app takes repeatable `--copc <path>` arguments or "Open COPC…".
Non-streaming consumers depending only on `SatinPointRasteriser` pay no C++-interop cost.

## Requirements

- macOS 15+ or iOS 18+
- Xcode with Metal tooling
- Swift 6 toolchain (strict concurrency)

The package depends on Satin:

```swift
.package(url: "https://github.com/Fabric-Project/Satin", exact: "20.0.0-Beta-1")
```

## Quickstart

```swift
import Satin
import SatinPointRasteriser

let rasteriser = PointRasteriser(context: context)
scene.add(rasteriser)

let cloud = PointRasteriserPointCloud(
    context: context,
    packed: PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 64)
)
rasteriser.addPointCloud(cloud)

rasteriser.configuration.pointSizeScale = 5
rasteriser.configuration.maximumPointSize = 6

// Per resize:
rasteriser.resize(size: size, scaleFactor: scaleFactor)

// Per frame, inside your Renderer's draw(renderPassDescriptor:commandBuffer:):
renderer.draw(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer, scene: scene, camera: camera)
rasteriser.draw(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer)
```

`PointRasteriser` is a Satin `Object` — adding it to the scene drives its per-frame
`update`/`encode` through `RenderEncoder.draw`; `draw(renderPassDescriptor:commandBuffer:)`
composites the resolved cloud in a following pass, same two-call shape as the sibling.

Load a real point cloud:

```swift
let packed = try PLYPointCloudLoader.load(url: url)
let cloud = PointRasteriserPointCloud(context: context, packed: packed)
rasteriser.addPointCloud(cloud)
```

See [API.md](API.md) for the full type reference, the `TintPass`/`DisplacementPass` sketch
contracts, picking, and amortization semantics.

## Example app

`Sources/PointRasteriserExample` is a macOS SwiftUI app (`SatinMetalView`) with:

- A **Settings** sheet (toolbar button) exposing every `PointRasteriserConfiguration`
  field — render mode, point sizing, LOD/culling, amortized-LOD budget + a "Restart sweep"
  button + live sweep-progress/LOD-count/overflow readout, point rejection, hole fill,
  SIMD aggregation, scene-depth writing, chunk/overdraw debug colorizing, background color,
  and motion blur.
- **Open PLY…** (toolbar button, `fileImporter`), loading off the main thread and reframing
  the camera to the loaded cloud's bounds.
- A **Depth of field** section — the same translucent-defocus (weighted-blended OIT) +
  jitter-spread recipe as `Satin-ComputeRasteriser`'s example app, built from a
  `DisplacementPass` + `TintPass` pair (see API.md).
- Keyboard toggles: `A` cycles the amortized LOD budget, `R` restarts the sweep, `D` toggles
  a demo sine-wave `DisplacementPass` sketch.

Run it:

```sh
swift run PointRasteriserExample
```

There's also a headless GPU benchmark:

```sh
swift run -c release PointRasteriserBench
```

## Benchmark (Slice 4, M5 Max)

10.08M points (216³ fixture, CLOD disabled), 1024×1024, 12 timed frames, mean GPU ms
(`gpuEndTime - gpuStartTime`):

| Variant | Framed view | Overdraw view (View 5) |
|---|---|---|
| HQ average | 1.89 ms | 1.46 ms |
| HQ average + SIMD aggregation | 1.69 ms | 1.34 ms |
| HQ average + SIMD + rejection | 1.83 ms | 1.36 ms |
| Nearest, 64-bit (Apple9) | 1.09 ms | 1.21 ms |
| Nearest, portable fallback | 1.21 ms | 1.21 ms |

"Framed view" is a camera centered on the cloud at a normal distance; "Overdraw view"
(Magnopus's "View 5") pulls the camera far back so the whole cloud collapses into a tiny,
high-contention screen region — the worst case the SIMD-aggregation and 64-bit fast paths
target. Reproduce with `swift run -c release PointRasteriserBench`; see
`Sources/PointRasteriserBench/main.swift`.

For a same-hardware, same-data comparison against `Satin-ComputeRasteriser`, see the
throwaway `CompareBench` results in the Slice 7 report (not checked into this repo).

## Migrating from Satin-ComputeRasteriser

Public contracts (Tint/Displacement sketch ABI, `RasterBatch` layout, picking indices)
are unchanged. Known signature deltas for host apps (e.g. WABF):

- `ComputeRasteriserPointCloud` → `PointRasteriserPointCloud`. There is no
  `capacity: ComputeRasteriserCapacity` init — create a reusable GPU-pack cloud with
  `PointRasteriserPointCloud(context:gpuPackPointCount:label:)` (or the `gpuPacked(...)`
  factory), then call `replacePackedPointCloud(...)` on reload. Both replace overloads
  (`_ packed:` and `packer:queue:positions:colors:count:`) match the old shapes and are
  identity-stable in place (the GPU overload adds a defaulted `shuffle:`); wholesale
  clouds only — slot pools use `addBatches`/`removeBatches`.
- `GPUPacker(context:)` → `GPUPacker(device:)`.
- `draw(renderPassDescriptor:commandBuffer:viewport:)` matches the old signature.
- A resizing replace resets displacement/tint buffers to fresh zeroed storage.

## Attribution

See `PLAN.md` for the architecture derivation from the Magnopus article and the four
synthesized papers (Schütz 2019/2021/2022, VAST 2011).

## API

See [API.md](API.md) for integration details.
