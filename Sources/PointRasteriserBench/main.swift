#if os(macOS)
import Foundation
import Metal
import Satin
import SatinPointRasteriser
import simd

// Offscreen GPU benchmark for the Slice-4 fast paths. Renders a ≥10M-point
// fixture under two cameras (a framed view and a Magnopus "View 5" worst-case
// where the whole cloud collapses into a tiny, high-overdraw screen region) for
// each pipeline variant, timing with MTLCommandBuffer gpuStart/EndTime.

guard let device = MTLCreateSystemDefaultDevice() else {
    print("No Metal device."); exit(1)
}

let renderWidth = 1024
let renderHeight = 1024
let warmupFrames = 3
let timedFrames = 12

let caps = RasteriserCapabilities(device: device)
print("Device: \(caps.gpuFamilyDescription), supportsApple9=\(caps.supportsApple9), use64BitAtomics=\(caps.use64BitAtomics)")

// ~10.1M points (216³). Shuffle disabled for a deterministic build.
print("Building fixture (216³ ≈ 10.08M points)…")
let buildStart = Date()
let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 216)
print("  built \(packed.pointCount) points in \(String(format: "%.1f", Date().timeIntervalSince(buildStart)))s")

struct CameraSetup { let name: String; let apply: (PerspectiveCamera) -> Void }
let cameras: [CameraSetup] = [
    CameraSetup(name: "Framed") { c in c.position = [1.5, 1.2, 2.2]; c.lookAt(target: .zero) },
    // View 5: camera far back so the cloud projects into a tiny, extreme-overdraw region.
    CameraSetup(name: "Overdraw") { c in c.position = [0, 0, 40]; c.lookAt(target: .zero) },
]

struct Variant {
    let name: String
    let use64: Bool?
    let configure: (inout PointRasteriserConfiguration) -> Void
}

// All variants disable CLOD so the full 10M-point workload rasterizes every
// frame (a fair, worst-case comparison; CLOD would cull the Overdraw view to a
// handful of coarse points and hide the contention the fast paths target).
let variants: [Variant] = [
    Variant(name: "HQ avg", use64: nil) { c in
        c.renderMode = .highQualityAverage; c.enableSimdAggregation = false; c.enablePointRejection = false
    },
    Variant(name: "HQ avg +simd", use64: nil) { c in
        c.renderMode = .highQualityAverage; c.enableSimdAggregation = true; c.enablePointRejection = false
    },
    Variant(name: "HQ avg +simd +reject", use64: nil) { c in
        c.renderMode = .highQualityAverage; c.enableSimdAggregation = true; c.enablePointRejection = true
    },
    Variant(name: "Nearest 64-bit", use64: true) { c in
        c.renderMode = .nearestPoint; c.enablePointRejection = false
    },
    Variant(name: "Nearest fallback", use64: false) { c in
        c.renderMode = .nearestPoint; c.enablePointRejection = false
    },
]

func percentile(_ values: [Double], _ p: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let rank = p * Double(sorted.count - 1)
    let lo = Int(rank.rounded(.down)), hi = Int(rank.rounded(.up))
    let frac = rank - Double(lo)
    return sorted[lo] * (1 - frac) + sorted[hi] * frac
}

/// Render `warmup + timed` frames; return the timed per-frame GPU ms.
// `forceReselect`: jitter the camera by an imperceptible epsilon each frame so
// the full-sweep LODSelect skip key never matches → LODSelect runs every frame
// (the pre-skip behavior). The rendered workload is otherwise identical, so the
// static ("skip") vs jittered ("select") delta isolates the LODSelect cost.
func benchmark(variant: Variant, camera cameraSetup: CameraSetup, forceReselect: Bool) -> [Double] {
    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)
    let rasteriser = PointRasteriser(context: context, use64BitAtomics: variant.use64)
    rasteriser.setup()
    var config = PointRasteriserConfiguration()
    config.enableCLOD = false
    config.pointSizeMode = .screenSpace
    config.minimumPointSize = 1
    config.maximumPointSize = 1
    config.pointSizeScale = 1
    config.holeFillIterations = 0
    variant.configure(&config)
    rasteriser.configuration = config

    let cloud = PointRasteriserPointCloud(context: context, packed: packed)
    rasteriser.addPointCloud(cloud)
    rasteriser.resize(size: (Float(renderWidth), Float(renderHeight)), scaleFactor: 1)

    let camera = PerspectiveCamera(context: context, position: [0, 0, 2.4], near: 0.01, far: 200, fov: 45)
    camera.aspect = Float(renderWidth) / Float(renderHeight)
    cameraSetup.apply(camera)
    let basePosition = camera.position
    let viewport = simd_float4(0, 0, Float(renderWidth), Float(renderHeight))

    var times: [Double] = []
    for frame in 0 ..< (warmupFrames + timedFrames) {
        if forceReselect {
            camera.position = basePosition + SIMD3<Float>(Float(frame) * 1e-5, 0, 0)
            camera.lookAt(target: .zero)
        }
        guard let cb = context.commandQueue.makeCommandBuffer() else { continue }
        rasteriser.update(renderContext: context, camera: camera, viewport: viewport, index: 0)
        rasteriser.encode(cb)
        cb.commit()
        cb.waitUntilCompleted()
        if frame >= warmupFrames {
            times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000.0)
        }
    }
    return times
}

// Header. Each camera reports two columns: "select" (LODSelect every frame, the
// pre-skip behavior) and "skip" (static-scene, LODSelect reused) — mean ms.
print("\n## PointRasteriser benchmark — \(caps.gpuFamilyDescription), \(renderWidth)×\(renderHeight), \(packed.pointCount) pts, \(timedFrames) timed frames\n")
var header = "| Variant |"
var sep = "|---|"
for cam in cameras { header += " \(cam.name): select → skip ms |"; sep += "---|" }
print(header)
print(sep)

func meanMs(_ variant: Variant, _ cam: CameraSetup, forceReselect: Bool) -> Double {
    let t = benchmark(variant: variant, camera: cam, forceReselect: forceReselect)
    return t.reduce(0, +) / Double(max(t.count, 1))
}

for variant in variants {
    if variant.use64 == true, !caps.supportsApple9 {
        var row = "| \(variant.name) |"
        for _ in cameras { row += " n/a |" }
        print(row)
        continue
    }
    var row = "| \(variant.name) |"
    for cam in cameras {
        let sel = meanMs(variant, cam, forceReselect: true)
        let skip = meanMs(variant, cam, forceReselect: false)
        let pct = sel > 0 ? (sel - skip) / sel * 100 : 0
        row += String(format: " %.2f → %.2f (−%.0f%%) |", sel, skip, pct)
    }
    print(row)
}
print("")
#else
print("PointRasteriserBench is only available on macOS.")
#endif
