# API

This document covers the package's public surface. There is no out-of-core/streaming API
yet (see [README.md](README.md#relationship-to-satin-computerasteriser)) — for COPC/SwiftPDAL
streaming, use `Satin-ComputeRasteriser` today.

## Main Types

### `PointRasteriser`

`PointRasteriser` is a Satin `Object` that owns the LOD-compaction + rasterization compute
pipeline and the composited output texture.

```swift
let rasteriser = PointRasteriser(context: context)
scene.add(rasteriser)
```

Use it inside a Satin renderer, same two-call shape as the sibling:

```swift
renderer.draw(
    renderPassDescriptor: renderPassDescriptor,
    commandBuffer: commandBuffer,
    scene: scene,
    camera: camera
)

rasteriser.draw(
    renderPassDescriptor: renderPassDescriptor,
    commandBuffer: commandBuffer
)
```

Call `resize` from your view renderer's resize path:

```swift
rasteriser.resize(size: size, scaleFactor: scaleFactor)
```

Key properties:

```swift
public var configuration: PointRasteriserConfiguration
public private(set) var outputTexture: MTLTexture?   // .rgba8Unorm, composited result
public private(set) var depthTexture: MTLTexture?     // .r32Float, reverse-Z NDC depth
public let capabilities: RasteriserCapabilities
```

Point cloud management:

```swift
@discardableResult
public func addPointCloud(_ cloud: PointRasteriserPointCloud) -> PointRasteriserPointCloud

public func removePointCloud(_ cloud: PointRasteriserPointCloud)

public var pointClouds: [PointRasteriserPointCloud]       // whole subtree
public var visiblePointClouds: [PointRasteriserPointCloud] // honors Object.visible up the chain
```

Each phase (LODSelect, depth, color) runs across all visible clouds before the next phase
starts, so occlusion between clouds is independent of add order.

Init takes an optional override for the 64-bit nearest-mode fast path (normally probed from
the device):

```swift
public init(context: Context, label: String = "PointRasteriser", use64BitAtomics: Bool? = nil)
```

### `PointRasteriserPointCloud`

`PointRasteriserPointCloud` is a Satin `Object` that owns the GPU buffers for one packed
point cloud: the quantized source layout (uploaded once) plus the compacted "LOD point
cloud" the LODSelect pass fills every frame (or every amortized sweep).

```swift
let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 64)
let cloud = PointRasteriserPointCloud(context: context, packed: packed, lodCapacity: 32_000_000)
rasteriser.addPointCloud(cloud)
```

Because it's an `Object`, the cloud can be transformed through Satin object transforms:

```swift
cloud.position = [0, 0, -2]
cloud.scale = [2, 2, 2]
```

There is no in-place `replacePackedPointCloud` — swap clouds by constructing a new
`PointRasteriserPointCloud` and calling `removePointCloud`/`addPointCloud` (see the example
app's `loadPLY`).

CPU-readable LOD sweep telemetry (no GPU stall — backed by `.storageModeShared` buffers):

```swift
public var lodCount: Int            // front sweep's survivor count
public var lodOverflow: Int         // points dropped for want of LOD capacity
public var lodOverflowed: Bool
public var lodSweepProgress: Float  // [0, 1] progress of the in-flight back sweep
public var isDoubleBuffered: Bool   // true once amortization has allocated the 2nd LOD set
```

### Per-point displacement (animated noise, deformers, jitter)

`PointRasteriserPointCloud.displacementBuffer` is an optional `MTLBuffer` of `float3` deltas
(stride 16, one entry per pack-order point index). When
`PointRasteriserConfiguration.applyDisplacement == true`, the depth and color passes add
`displacements[pointIndex]` to each decoded position before projection — both passes see the
same delta, so depth and color land on the same pixel. `.highQualityAverage` mode only;
`.nearestPoint` ignores displacement (mirrors the sibling — picking also uses undisplaced
positions).

The easiest way to drive it is a `DisplacementPass`:

```swift
let pass = DisplacementPass(rasteriser: rasteriser, kernelURL: url, live: true)
pass.bindUserBuffers = { encoder in
    var t = time
    encoder.setBytes(&t, length: MemoryLayout<Float>.stride, index: DisplacementPass.bufferUser0)
}
// Every frame, before rasteriser.encode(...):
pass.encode(commandBuffer: commandBuffer)   // targets the first cloud, or pass `cloud:`
```

`DisplacementPass` owns the pipeline build (with `MetalFileCompiler` live reload when
`live: true`), auto-allocates `displacementBuffer` on first encode, and flips
`configuration.applyDisplacement = true` for you. Call `pass.disable()` to flip it back off
without tearing the pass down.

**Sketch contract** (byte-identical to Satin-ComputeRasteriser's `DisplacementPass` — sketches
written against the sibling compile unchanged here). The package's `ScrSketchPreamble` is
concatenated before your kernel source and exposes:

```metal
typedef struct {
    int  state;
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    uint numPoints;
    uint firstPoint;
    uint fileIndex;
    // ...padding
} RasterBatch;

float3 scr_decodePointAt(uint pointIndex, RasterBatch batch,
    device const uint *xyzLow, device const uint *xyzMed, device const uint *xyzHigh,
    device const uchar *levels);

float4 scr_decodeColorAt(uint pointIndex, device const uint *colors);

bool scr_resolveDisplacementThread(uint id, constant ScrDisplacementInfo &info,
    device const RasterBatch *batches,
    thread RasterBatch &batch, thread uint &pointIndex, thread uint &localOffset);
```

Buffer slot constants (mirrors `SCR_DISP_BUF_*`), auto-bound by the `SCR_DISPLACEMENT_KERNEL_BUFFERS` macro:

```swift
public static let bufferBatches: Int = 0
public static let bufferXYZLow: Int = 1
public static let bufferXYZMed: Int = 2
public static let bufferXYZHigh: Int = 3
public static let bufferLevels: Int = 4
public static let bufferDisplacements: Int = 5
public static let bufferInfo: Int = 6
public static let bufferColors: Int = 7
public static let bufferUser0: Int = 8   // ... bufferUser7 = 15
```

Minimal kernel:

```metal
kernel void computeDisplacement(
    uint id [[thread_position_in_grid]],
    SCR_DISPLACEMENT_KERNEL_BUFFERS,
    constant float &time [[buffer(SCR_DISP_BUF_USER0)]]
) {
    RasterBatch batch; uint pointIndex; uint localOffset;
    if (!scr_resolveDisplacementThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
    const float3 p = scr_decodePointAt(pointIndex, batch, xyzLow, xyzMed, xyzHigh, levels);
    displacements[pointIndex] = float3(0.0, sin(p.x * 8.0 + time * 3.0) * 0.08, 0.0);
}
```

Notes:

- **Deltas, not absolute positions.** `displacements[i]` is added after decode.
- Write a **NaN** displacement to cull a point — both depth and color passes skip a point
  whose displaced position is NaN (a true removal, not just invisible).
- **Pack-order indexing.** `pointIndex` is the same index `lodSourceIndices` and
  `PackedPointCloud.orderedPositions`/`sourceIndices` use.
- Internal difference from the sibling (contract preserved): this package packs a cloud's
  points **contiguously**, so `scr_resolveDisplacementThread` binary-searches the batch table
  by `firstPoint` rather than dividing by a fixed slot stride. User kernels are unaffected.
- **Frustum culling sees undisplaced bounds.** Large displacements can push points outside
  their batch's AABB; disable `enableFrustumCulling` or keep displacement within roughly one
  batch radius.

### Per-point tint (color replace / translucent defocus)

`PointRasteriserPointCloud.tintBuffer` is an optional `MTLBuffer` of `float4` (pack-order
index). When `PointRasteriserConfiguration.applyTint == true`, the color pass composes
`final = mix(stored.rgb, tint.rgb, tint.a)`:

- `tint.a == 0` — pass-through (no visual change).
- `tint.a == 1` — full color replacement.
- `tint.a < 0` — discard sentinel; the point contributes nothing (a pixel covered only by
  such points ends up transparent).

Driven the same way as displacement, via `TintPass`:

```swift
let tint = TintPass(rasteriser: rasteriser, kernelURL: url, live: true)
tint.encode(commandBuffer: commandBuffer)   // flips configuration.applyTint = true
```

Set `tint.alphaIsCoverage = true` to opt into **translucent defocus** (weighted-blended
OIT): the kernel writes a circle-of-confusion into `tints[i].a`, and the rasteriser then
treats those points as translucent (skip depth write, accumulate coverage-weighted) instead
of a color mix. This flips `PointRasteriserConfiguration.tintAlphaIsCoverage` on encode —
see the [Depth of field recipe](#depth-of-field-recipe) below, which combines this with
`DisplacementPass`.

Buffer slots mirror `DisplacementPass` (`SCR_TINT_BUF_*`, `bufferBatches`...`bufferColors`
at 0–7, `bufferUser0`...`bufferUser7` at 8–15); the output buffer is `device float4 *tints`.
`.highQualityAverage` mode only; nearest-point mode ignores tint.

### Depth of field recipe

Both packages ship the same DoF recipe on top of `DisplacementPass` + `TintPass`: a
`DisplacementPass` scatters out-of-focus points along a per-point random direction
(`dofHash3`), scaled by a circle-of-confusion computed from view-space depth vs. a focal
distance; a `TintPass` (with `alphaIsCoverage = true`) writes that same circle-of-confusion
into `tint.a` so out-of-focus points become translucent instead of hard-occluding. Both
kernels share a small `CameraUniforms`/`DofParams` byte-compatible pair bound at
`bufferUser0`/`bufferUser1`:

```metal
typedef struct {
    float4x4 modelView;     // camera.view · cloud.world
    float    near;
    float    far;
    float    focalDistance; // sharp distance from the camera, view-space units
} CameraUniforms;
typedef struct {
    float band;       // sharp half-band, fraction of focalDistance
    float falloff;    // ramp to full effect, fraction of focalDistance
    float scatter;    // jitter spread, fraction of focalDistance
    float maxDefocus;  // transparency cap (1 = a point can fully vanish)
} DofParams;
```

`band`/`falloff`/`scatter` are fractions of `focalDistance`, so the effect auto-scales to
whatever cloud is loaded. See `Sources/PointRasteriserExample/PointRasteriserExampleRenderer.swift`
(`makeDofPasses`/`encodeDof`/`bindDof` + the embedded kernel strings) for the full,
directly-portable implementation — it is a verbatim adaptation of
`Satin-ComputeRasteriser`'s `ComputeRasteriserAppRenderer` DoF extension, since the sketch
contract is unchanged between the two packages.

### Picking

```swift
public func pickPointIndex(
    atNDC ndc: SIMD2<Float>,
    in cloud: PointRasteriserPointCloud,
    camera: Camera,
    searchRadius: Int = 10
) -> UInt32?
```

Runs the nearest-winner pass for `cloud` off-screen on its own command buffer, **waits**,
reads back the winning LOD index in a small band around the cursor (point clouds are sparse,
so the exact pixel is often empty), and maps it through the cloud's `lodSourceIndices` to a
**pack-order source index** — the same index `DisplacementPass`/`TintPass` buffers use, and
into `PackedPointCloud.orderedPositions`/`sourceIndices`. `ndc` is `[-1, 1]` with **y up**.
Call it between frames (e.g. a click handler), not inside the render loop.

### `PackedPointCloud`

CPU-side packed point data, produced by `PackedPointCloudFixtures.pack(...)` or
`PLYPointCloudLoader`:

```swift
public struct PackedPointCloud: Sendable {
    public var batches: [RasterBatch]
    public var files: [RasterFile]
    public var xyzLow: [UInt32]
    public var xyzMed: [UInt32]
    public var xyzHigh: [UInt32]
    public var colors: [UInt32]
    public var levels: [UInt8]
    public var boundsMin: SIMD3<Float>
    public var boundsMax: SIMD3<Float>
    public var orderedPositions: [SIMD3<Float>]  // pack-order; empty if a loader didn't preserve it
    public var sourceIndices: [UInt32]            // pack-order → original input index
    public var pointCount: Int { colors.count }
    public var batchCount: Int { batches.count }
}
```

Coordinates are quantized per batch into three 10-bit chunks (`xyzLow`/`xyzMed`/`xyzHigh` —
30-bit combined at LOD level 0, 20-bit at level 1, 10-bit at level 2+, matching each point's
`levels` entry). Colors pack as `R | (G << 8) | (B << 16) | (A << 24)`.

### `PointRasteriserConfiguration`

```swift
public struct PointRasteriserConfiguration: Sendable {
    public var renderMode: RenderMode                    // .highQualityAverage | .nearestPoint
    public var enableSimdAggregation: Bool                // simdgroup pre-reduce before atomics
    public var pointSizeMode: PointSizeMode               // .screenSpace | .worldSpace
    public var minimumPointSize: Float
    public var maximumPointSize: Float
    public var pointSizeScale: Float
    public var enableFrustumCulling: Bool
    public var enableCLOD: Bool
    public var lodBias: Int
    public var enableLODDither: Bool
    public var lodPointsPerFrame: Int                     // 0 = full sweep; see Amortization below
    public var backgroundColor: SIMD4<Float>
    public var depthTolerance: Float
    public var enablePointRejection: Bool                 // VAST 2011 cone occlusion, merged into resolve
    public var rejectionConeThreshold: Float              // radians
    public var holeFillIterations: Int
    public var colorizeChunks: Bool
    public var colorizeOverdraw: Bool
    public var writesSceneDepth: Bool
    public var lodCapacity: Int                           // currently unused; capacity is fixed per-cloud, see below
    public var applyDisplacement: Bool                     // owned by DisplacementPass.encode()/disable()
    public var applyTint: Bool                             // owned by TintPass.encode()/disable()
    public var tintAlphaIsCoverage: Bool                   // owned by TintPass.alphaIsCoverage
    public var motionBlur: Float
    public var motionBlurSamples: Int
    public var motionBlurMaxSpread: Float
}
```

Defaults: `renderMode = .highQualityAverage`, `enableSimdAggregation = true`,
`pointSizeMode = .screenSpace`, `minimumPointSize = 1`, `maximumPointSize = 5`,
`pointSizeScale = 5`, `enableFrustumCulling = true`, `enableCLOD = true`, `lodBias = 0`,
`enableLODDither = true`, `lodPointsPerFrame = 0`, `backgroundColor = [0,0,0,0]`,
`depthTolerance = 0.01`, `enablePointRejection = true`, `rejectionConeThreshold = 0.5`,
`holeFillIterations = 0`, `writesSceneDepth = true`, `lodCapacity = 32_000_000`,
`motionBlurSamples = 8`, `motionBlurMaxSpread = 64`; the rest `false`/`0`.

> **`lodCapacity` on `PointRasteriserConfiguration` is currently a no-op** — the effective
> per-cloud capacity is fixed at `PointRasteriserPointCloud` construction
> (`min(totalPoints, lodCapacity)`, that initializer's own `lodCapacity` parameter, default
> `32_000_000`), not read from this struct. Set it on the cloud, not the configuration.

> **`applyDisplacement` / `applyTint` / `tintAlphaIsCoverage` are pass-managed.** Setting
> them directly works, but the intended flow is: construct a `DisplacementPass`/`TintPass`,
> call `.encode(commandBuffer:)` each frame you want the effect active (it flips the flag on
> for you) and `.disable()` when you don't (flips it off). If you rebuild `configuration`
> from scratch each frame (as the example app does), simply omit these three from your
> rebuild — the pass's `encode()`/`disable()` call, made later in the same frame before the
> rasteriser consumes them, is authoritative.

Setting `configuration` re-pushes every value into the pass processors (mirrors the
sibling's `didSet`-driven push). Changing `enableCLOD`, `lodBias`, or `enableLODDither`
automatically calls `restartLODSweep()` for you, since those changes invalidate whatever
sweep is in flight.

Modes:

- `.highQualityAverage` — LODSelect → depth → color → reject+resolve → hole fill. All
  visible clouds contribute.
- `.nearestPoint` — single 64-bit packed-atomic winner pass on Apple9+ (`atomic_max` on
  `ulong(depthUint << 32 | lodIndex)`), or a portable two-pass 32-bit fallback; then a
  nearest-mode reject+resolve. Renders only the **first** visible cloud (the winner index
  alone doesn't identify a cloud).

Point size:

```swift
rasteriser.configuration.pointSizeMode = .screenSpace
rasteriser.configuration.minimumPointSize = 1
rasteriser.configuration.maximumPointSize = 5
rasteriser.configuration.pointSizeScale = 5
```

- `.screenSpace`: `pointSize = clamp(pointSizeScale / viewDistance, minimumPointSize, maximumPointSize)`
  pixels — `pointSizeScale` reads as "pixels at one unit of view distance"; FOV doesn't
  affect it (constant under orthographic projection).
- `.worldSpace`: `pointSizeScale` is a world-space sphere radius in scene units, projected to
  pixels; FOV and screen height affect the result.

### `RasteriserCapabilities`

```swift
public struct RasteriserCapabilities: Sendable {
    public let supportsApple9: Bool          // device.supportsFamily(.apple9)
    public let use64BitAtomics: Bool         // defaults to supportsApple9; overridable
    public let gpuFamilyDescription: String  // e.g. "Apple9"

    public init(device: MTLDevice, use64BitAtomics: Bool? = nil)
    public static func verify64BitAtomics(device: MTLDevice) -> Bool
}
```

`verify64BitAtomics` compiles and dispatches a tiny probe kernel doing
`atomic_fetch_max_explicit` on `atomic_ulong` against known inputs and checks the result,
rather than trusting `supportsFamily` alone — catching drivers/hardware that reject 64-bit
buffer atomics despite reporting Apple9. `PointRasteriser.init(context:use64BitAtomics:)`
forwards an explicit override straight through if you want to force the portable fallback
(e.g. for testing).

## Amortization (`lodPointsPerFrame`)

`lodPointsPerFrame = 0` (default) re-runs the full LODSelect sweep every frame — simplest,
no extra memory, but scales with total point count per frame regardless of camera motion.
Setting it `> 0` amortizes selection: at most that many **source points per cloud** are
compacted per frame, resuming from a persistent cursor across frames (a "sweep"), while the
raster passes always read the **front** set — the last fully *completed* sweep — from a
double-buffered LOD set that's lazily allocated the first time amortization turns on.

Tradeoffs:

- **2× LOD memory** once amortization is enabled (the second buffer set is never freed by
  turning the budget back to `0`).
- A sweep in flight can leave **empty space at frame edges** when the camera moves faster
  than a sweep completes — the reference's accepted tradeoff for the throughput win.
- The very first sweep for a freshly added/loaded cloud always runs un-amortized (full
  sweep), so a new cloud isn't blank while its first sweep builds.
- `restartLODSweep()` (on `PointRasteriser`, applies to every cloud, or per-cloud on
  `PointRasteriserPointCloud`) abandons the in-flight back sweep and restarts from batch 0
  next frame; the front set keeps rendering until the new sweep finishes. Call it on a
  camera teleport (fast cuts) so a stale front doesn't linger — selection-affecting config
  changes (`enableCLOD`, `lodBias`, `enableLODDither`) call it for you automatically.
- Poll `cloud.lodSweepProgress` / `lodCount` / `lodOverflow` / `lodOverflowed` for live
  telemetry (all CPU-readable, no GPU stall) — the example app's settings sheet does this
  every frame.

## Loading PLY

```swift
let packed = try PLYPointCloudLoader.load(url: url)
// or:
let packed = try PLYPointCloudLoader.parse(data)
```

Supported: `format ascii 1.0` and `format binary_little_endian 1.0`; required vertex
properties `x`/`y`/`z`; optional `red`/`green`/`blue` (or `r`/`g`/`b`,
`diffuse_red`/`diffuse_green`/`diffuse_blue`), normalized to `[0, 1]` per their declared
scalar type. Points default to white if no color properties are present. Not supported:
`list` properties, big-endian formats.

## Fixture data

```swift
let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 24)

let packed = PackedPointCloudFixtures.pack(
    positions: positions,
    colors: colors,
    pointsPerBatch: pointRasteriserThreadsPerGroup * 80,
    lodLevels: PackedPointCloudFixtures.defaultLODLevels,
    coarseVoxelDivisions: PackedPointCloudFixtures.defaultCoarseVoxelDivisions,
    shuffleBatches: true   // Magnopus inter-wave-contention mitigation; false preserves strict Morton order
)
```

`pack(...)` computes bounds → Morton-sorts → assigns a per-point LOD level → buckets each
batch slice level-ascending → quantizes each point to 30 bits relative to its batch's AABB →
(unless disabled) shuffles the batch order.

## Layout constants

```swift
PointRasteriserLayout.rasterBatchStride    == 64
PointRasteriserLayout.rasterFileStride     == 256
PointRasteriserLayout.rasterPixelStride    == 32
PointRasteriserLayout.visibleBatchStride   == 16
PointRasteriserLayout.dispatchArgsStride   == 12
LODCloudLayout.positionStride              == 12   // packed_float3
LODCloudLayout.colorStride                 == 4    // packed RGBA8
LODCloudLayout.sourceIndexStride           == 4
```

## Runtime notes

- Kernels compile at runtime from the `Pipelines` resource bundle via `MetalFileCompiler`
  (`Bundle.module`), same pattern as the sibling — shader errors surface when a pipeline is
  first built (typically at `PointRasteriser.setup()`/first `encode`), not at `swift test`.
- `PointRasteriser` caches per-viewport GPU resources (pixel buffer, resolve/hole-fill
  textures, nearest-mode scratch buffers) keyed by integer pixel size, LRU-capped at 4, so
  alternating render sizes (e.g. a live drawable vs. an offscreen benchmark target) don't
  reallocate every frame.
- Reverse-Z depth throughout, consistent with Satin's convention.
- The 64-bit nearest fast path requires `MTLGPUFamily.apple9`; everywhere else uses the
  portable 32-bit two-pass fallback automatically.
