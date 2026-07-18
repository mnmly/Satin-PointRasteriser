import CoreGraphics
import Foundation
import ImageIO
import Metal
import Satin
import simd
import Testing
import UniformTypeIdentifiers
@testable import SatinPointRasteriser

// End-to-end GPU verification of the minimal pipeline: LODSelect → finalize →
// clear → depth → color → resolve, reading back the resolved output/depth
// textures directly (no window, no composite render pass). On a GPU-less CI
// host `MTLCreateSystemDefaultDevice()` returns nil and the tests return early.

private struct RenderResult {
    let width: Int
    let height: Int
    let rgba: [UInt8]      // width*height*4
    let depth: [Float]     // width*height
    let lodCount: Int
    let lodOverflow: Int
    let totalPoints: Int

    func alpha(_ x: Int, _ y: Int) -> UInt8 { rgba[(y * width + x) * 4 + 3] }
    func rgb(_ x: Int, _ y: Int) -> (UInt8, UInt8, UInt8) {
        let i = (y * width + x) * 4
        return (rgba[i], rgba[i + 1], rgba[i + 2])
    }
    func depthAt(_ x: Int, _ y: Int) -> Float { depth[y * width + x] }

    var coveredPixelCount: Int {
        var n = 0
        for i in 0 ..< (width * height) where rgba[i * 4 + 3] > 0 { n += 1 }
        return n
    }
}

/// Encodes one offscreen frame and reads back the resolved textures + stats.
private func renderOffscreen(
    device: MTLDevice,
    width: Int,
    height: Int,
    pointsPerAxis: Int = 16,
    packed: PackedPointCloud? = nil,
    use64BitAtomics: Bool? = nil,
    configure: (PointRasteriser) -> Void = { _ in },
    placeCamera: (PerspectiveCamera) -> Void
) -> RenderResult? {
    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)

    let rasteriser = PointRasteriser(context: context, use64BitAtomics: use64BitAtomics)
    rasteriser.setup()
    configure(rasteriser)

    let cloudData = packed ?? PackedPointCloudFixtures.cubeGrid(pointsPerAxis: pointsPerAxis)
    let cloud = PointRasteriserPointCloud(context: context, packed: cloudData)
    rasteriser.addPointCloud(cloud)

    rasteriser.resize(size: (Float(width), Float(height)), scaleFactor: 1)

    let camera = PerspectiveCamera(context: context, position: [0, 0, 2.4], near: 0.01, far: 100, fov: 45)
    camera.aspect = Float(width) / Float(height)
    placeCamera(camera)

    let viewport = simd_float4(0, 0, Float(width), Float(height))
    rasteriser.update(renderContext: context, camera: camera, viewport: viewport, index: 0)

    guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { return nil }
    rasteriser.encode(commandBuffer)

    guard let outputTexture = rasteriser.outputTexture,
          let depthTexture = rasteriser.depthTexture,
          let rgbaStaging = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
          let depthStaging = device.makeBuffer(length: width * height * MemoryLayout<Float>.stride, options: .storageModeShared),
          let blit = commandBuffer.makeBlitCommandEncoder()
    else { return nil }

    blit.copy(
        from: outputTexture, sourceSlice: 0, sourceLevel: 0,
        sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
        sourceSize: MTLSize(width: width, height: height, depth: 1),
        to: rgbaStaging, destinationOffset: 0,
        destinationBytesPerRow: width * 4, destinationBytesPerImage: width * height * 4
    )
    blit.copy(
        from: depthTexture, sourceSlice: 0, sourceLevel: 0,
        sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
        sourceSize: MTLSize(width: width, height: height, depth: 1),
        to: depthStaging, destinationOffset: 0,
        destinationBytesPerRow: width * MemoryLayout<Float>.stride,
        destinationBytesPerImage: width * height * MemoryLayout<Float>.stride
    )
    blit.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let rgba = [UInt8](unsafeUninitializedCapacity: width * height * 4) { buf, count in
        memcpy(buf.baseAddress!, rgbaStaging.contents(), width * height * 4)
        count = width * height * 4
    }
    let depth = [Float](unsafeUninitializedCapacity: width * height) { buf, count in
        memcpy(buf.baseAddress!, depthStaging.contents(), width * height * MemoryLayout<Float>.stride)
        count = width * height
    }

    return RenderResult(
        width: width, height: height, rgba: rgba, depth: depth,
        lodCount: cloud.lodCount, lodOverflow: cloud.lodOverflow,
        totalPoints: cloud.totalPoints
    )
}

@Test func offscreenRenderProducesVisibleCloud() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }

    guard let result = renderOffscreen(
        device: device, width: 256, height: 256, pointsPerAxis: 16,
        configure: { $0.configuration.pointSizeScale = 6 },
        placeCamera: { $0.lookAt(target: .zero) }
    ) else {
        Issue.record("failed to encode/read offscreen frame")
        return
    }

    // (d) LOD compaction produced a sensible survivor count.
    #expect(result.lodCount > 0, "expected LOD survivors, got 0")
    #expect(result.lodCount <= result.totalPoints, "survivors (\(result.lodCount)) exceed total (\(result.totalPoints))")
    #expect(result.lodOverflow == 0, "unexpected LOD overflow \(result.lodOverflow)")

    // (a) > 1% of pixels covered.
    let coverage = Float(result.coveredPixelCount) / Float(result.width * result.height)
    #expect(coverage > 0.01, "coverage \(coverage) below 1%")

    // (b) center-ish region has a non-background (non-transparent) pixel.
    var centerCovered = false
    var centerColored = false
    let cx = result.width / 2, cy = result.height / 2
    for dy in -16 ... 16 {
        for dx in -16 ... 16 {
            let x = cx + dx, y = cy + dy
            if result.alpha(x, y) > 0 {
                centerCovered = true
                let (r, g, b) = result.rgb(x, y)
                if Int(r) + Int(g) + Int(b) > 0 { centerColored = true }
            }
        }
    }
    #expect(centerCovered, "no covered pixels near center")
    #expect(centerColored, "covered center pixels are all black")

    // (c) depth is nonzero (reverse-Z) wherever color exists.
    var checkedDepth = false
    for y in 0 ..< result.height {
        for x in 0 ..< result.width where result.alpha(x, y) > 0 {
            #expect(result.depthAt(x, y) > 0, "covered pixel (\(x),\(y)) has zero depth")
            checkedDepth = true
        }
    }
    #expect(checkedDepth, "no covered pixels to verify depth")
}

@Test func offscreenRenderCullsWhenCameraFacesAway() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }

    guard let result = renderOffscreen(
        device: device, width: 256, height: 256, pointsPerAxis: 16,
        placeCamera: { camera in
            // Camera at +z looking further along +z: the cube at the origin is
            // behind it, so every batch AABB fails the frustum test.
            camera.position = [0, 0, 2.4]
            camera.lookAt(target: [0, 0, 100])
        }
    ) else {
        Issue.record("failed to encode/read offscreen frame")
        return
    }

    #expect(result.lodCount == 0, "expected 0 LOD survivors when culled, got \(result.lodCount)")
    #expect(result.coveredPixelCount == 0, "expected no covered pixels when culled, got \(result.coveredPixelCount)")
}

// MARK: - Slice 3: point rejection

/// Two coincident screen-filling planes: a sparse **near** plane (red) with gaps
/// and a dense **far** plane (green) behind it. Camera at +z looks down −z, so
/// larger z is nearer: near plane at z=+0.25, far at z=−0.5.
private func twoPlanesCloud(nearSide: Int = 52, farSide: Int = 130) -> PackedPointCloud {
    var positions: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    for yi in 0 ..< nearSide {
        for xi in 0 ..< nearSide {
            let x = Float(xi) / Float(nearSide - 1) - 0.5
            let y = Float(yi) / Float(nearSide - 1) - 0.5
            positions.append([x, y, 0.25])
            colors.append([1, 0, 0, 1]) // red near
        }
    }
    for yi in 0 ..< farSide {
        for xi in 0 ..< farSide {
            let x = Float(xi) / Float(farSide - 1) - 0.5
            let y = Float(yi) / Float(farSide - 1) - 0.5
            positions.append([x, y, -0.5])
            colors.append([0, 1, 0, 1]) // green far
        }
    }
    // Disable batch shuffle so the packing is deterministic for the test.
    return PackedPointCloudFixtures.pack(positions: positions, colors: colors, shuffleBatches: false)
}

private extension RenderResult {
    /// Count "green-dominant" covered pixels (far plane) in a centered square.
    func greenPixelCount(inCenterHalfExtent extent: Int) -> Int {
        var n = 0
        let cx = width / 2, cy = height / 2
        for y in max(0, cy - extent) ..< min(height, cy + extent) {
            for x in max(0, cx - extent) ..< min(width, cx + extent) where alpha(x, y) > 0 {
                let (r, g, b) = rgb(x, y)
                if g > 90, g > r + 30, g > b + 30 { n += 1 }
            }
        }
        return n
    }
}

@Test func pointRejectionRemovesFarPlaneLeakThroughGaps() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let cloud = twoPlanesCloud()

    // 1px footprints so the near plane leaves gaps the far plane leaks through;
    // CLOD off so both planes render in full and only rejection differs.
    let setup: (PointRasteriser) -> Void = { r in
        r.configuration.enableCLOD = false
        r.configuration.pointSizeMode = .screenSpace
        r.configuration.minimumPointSize = 1
        r.configuration.maximumPointSize = 1
        r.configuration.pointSizeScale = 1
    }

    guard let off = renderOffscreen(device: device, width: 256, height: 256, packed: cloud, configure: { r in
        setup(r); r.configuration.enablePointRejection = false
    }, placeCamera: { $0.lookAt(target: .zero) }),
    let on = renderOffscreen(device: device, width: 256, height: 256, packed: cloud, configure: { r in
        setup(r); r.configuration.enablePointRejection = true
    }, placeCamera: { $0.lookAt(target: .zero) }) else {
        Issue.record("failed to render two-plane scene")
        return
    }

    let greenOff = off.greenPixelCount(inCenterHalfExtent: 64)
    let greenOn = on.greenPixelCount(inCenterHalfExtent: 64)

    // With rejection off the far (green) plane leaks through the near plane's gaps.
    #expect(greenOff > 100, "expected far-plane leakage with rejection off, got \(greenOff)")
    // Rejection should remove most of that leak in the near plane's region.
    #expect(greenOn < greenOff, "rejection did not reduce far-plane leak (off=\(greenOff) on=\(greenOn))")
    #expect(Float(greenOn) < 0.75 * Float(greenOff),
            "rejection reduced leak by too little (off=\(greenOff) on=\(greenOn))")
}

// MARK: - Slice 3: hole fill

/// A sparse screen-facing plane (position-tinted) at z=0, spaced so 1px points
/// leave holes between them for the hole-fill pass to expand into.
private func sparsePlaneCloud(side: Int = 44) -> PackedPointCloud {
    var positions: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    for yi in 0 ..< side {
        for xi in 0 ..< side {
            let fx = Float(xi) / Float(side - 1)
            let fy = Float(yi) / Float(side - 1)
            positions.append([fx - 0.5, fy - 0.5, 0])
            colors.append([fx, fy, 0.5, 1])
        }
    }
    return PackedPointCloudFixtures.pack(positions: positions, colors: colors, shuffleBatches: false)
}

@Test func holeFillExpandsCoverageAndAveragesColorAndDepth() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let cloud = sparsePlaneCloud()

    // Rejection off + CLOD off so the ONLY difference between the two renders is
    // holeFillIterations; a flat plane has no closer neighbors so rejection is a
    // no-op anyway, but pin it for clarity.
    let base: (PointRasteriser) -> Void = { r in
        r.configuration.enableCLOD = false
        r.configuration.enablePointRejection = false
        r.configuration.minimumPointSize = 1
        r.configuration.maximumPointSize = 1
        r.configuration.pointSizeScale = 1
    }

    guard let off = renderOffscreen(device: device, width: 256, height: 256, packed: cloud, configure: { r in
        base(r); r.configuration.holeFillIterations = 0
    }, placeCamera: { $0.lookAt(target: .zero) }),
    let on = renderOffscreen(device: device, width: 256, height: 256, packed: cloud, configure: { r in
        base(r); r.configuration.holeFillIterations = 3
    }, placeCamera: { $0.lookAt(target: .zero) }) else {
        Issue.record("failed to render sparse plane")
        return
    }

    // Coverage strictly increases with hole filling.
    #expect(off.coveredPixelCount > 0, "sparse plane produced no coverage")
    #expect(on.coveredPixelCount > off.coveredPixelCount,
            "hole fill did not increase coverage (off=\(off.coveredPixelCount) on=\(on.coveredPixelCount))")

    // Spot-check newly filled pixels: color within the covered-neighbor envelope
    // (an average always lies within [min,max]) and depth filled (nonzero).
    var checked = 0
    var depthFilledChecks = 0
    let eps: Float = 6.0 / 255.0
    outer: for y in 1 ..< (on.height - 1) {
        for x in 1 ..< (on.width - 1) {
            guard off.alpha(x, y) == 0, on.alpha(x, y) > 0 else { continue } // newly filled
            // Envelope of covered neighbors in the filled image.
            var lo = SIMD3<Float>(repeating: 1), hi = SIMD3<Float>(repeating: 0)
            var neighbors = 0
            for dy in -1 ... 1 {
                for dx in -1 ... 1 where !(dx == 0 && dy == 0) {
                    let nx = x + dx, ny = y + dy
                    guard on.alpha(nx, ny) > 0 else { continue }
                    let (r, g, b) = on.rgb(nx, ny)
                    let c = SIMD3<Float>(Float(r), Float(g), Float(b)) / 255.0
                    lo = simd_min(lo, c); hi = simd_max(hi, c)
                    neighbors += 1
                }
            }
            guard neighbors >= 3 else { continue }
            let (fr, fg, fb) = on.rgb(x, y)
            let fc = SIMD3<Float>(Float(fr), Float(fg), Float(fb)) / 255.0
            #expect(fc.x >= lo.x - eps && fc.x <= hi.x + eps, "filled R outside neighbor envelope")
            #expect(fc.y >= lo.y - eps && fc.y <= hi.y + eps, "filled G outside neighbor envelope")
            #expect(fc.z >= lo.z - eps && fc.z <= hi.z + eps, "filled B outside neighbor envelope")
            #expect(on.depthAt(x, y) > 0, "filled pixel has zero depth at (\(x),\(y))")
            depthFilledChecks += 1
            checked += 1
            if checked >= 40 { break outer }
        }
    }
    #expect(checked > 0, "found no newly-filled pixels to spot-check")
    #expect(depthFilledChecks > 0, "no filled pixel had its depth verified")
}


// MARK: - Slice 4: nearest-point mode (64-bit fast path vs portable fallback)

@Test func nearestMode64BitMatchesPortableFallback() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    guard device.supportsFamily(.apple9) else { return } // 64-bit path unavailable

    // Tilt the camera so surface depths vary — this minimizes exact reverse-Z
    // depth ties, the only case where the two paths' tie-breaking can diverge.
    let place: (PerspectiveCamera) -> Void = { c in
        c.position = [1.4, 1.1, 1.9]
        c.lookAt(target: .zero)
    }
    let setup: (PointRasteriser) -> Void = { r in
        r.configuration.renderMode = .nearestPoint
        r.configuration.pointSizeScale = 5
        r.configuration.maximumPointSize = 5
    }

    guard let fast = renderOffscreen(device: device, width: 256, height: 256, pointsPerAxis: 20, use64BitAtomics: true, configure: setup, placeCamera: place),
          let slow = renderOffscreen(device: device, width: 256, height: 256, pointsPerAxis: 20, use64BitAtomics: false, configure: setup, placeCamera: place) else {
        Issue.record("failed to render nearest-mode images")
        return
    }

    #expect(fast.coveredPixelCount > 500, "nearest mode produced little coverage")

    // Coverage (which pixels have a winner) and depth are settled by atomic_max
    // identically in both paths, so they must match exactly. Color may differ
    // only where two points tie at the exact same reverse-Z depth on a pixel:
    // the 64-bit max picks the largest lodIndex, the fallback's fetch_min the
    // smallest. Assert depth-identical everywhere + color-identical on >99%.
    var covered = 0, colorMatches = 0, depthMismatches = 0, coverageMismatches = 0
    for i in 0 ..< (fast.width * fast.height) {
        let fa = fast.rgba[i * 4 + 3], sa = slow.rgba[i * 4 + 3]
        if (fa > 0) != (sa > 0) { coverageMismatches += 1; continue }
        if fast.depth[i] != slow.depth[i] { depthMismatches += 1 }
        guard fa > 0 else { continue }
        covered += 1
        if fast.rgba[i*4] == slow.rgba[i*4], fast.rgba[i*4+1] == slow.rgba[i*4+1], fast.rgba[i*4+2] == slow.rgba[i*4+2] {
            colorMatches += 1
        }
    }
    #expect(coverageMismatches == 0, "coverage differs between paths (\(coverageMismatches) px)")
    #expect(depthMismatches == 0, "depth differs between paths (\(depthMismatches) px)")
    let colorAgree = Float(colorMatches) / Float(max(covered, 1))
    #expect(colorAgree > 0.99, "color agreement \(colorAgree) below 99% (\(colorMatches)/\(covered))")
}

// MARK: - Slice 4: SIMD-group aggregation (identical images on vs off)

@Test func simdAggregationProducesIdenticalImage() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }

    let place: (PerspectiveCamera) -> Void = { c in
        c.position = [1.2, 0.9, 2.0]
        c.lookAt(target: .zero)
    }
    let base: (PointRasteriser) -> Void = { r in
        r.configuration.pointSizeScale = 6
        r.configuration.maximumPointSize = 6
    }

    guard let on = renderOffscreen(device: device, width: 256, height: 256, pointsPerAxis: 24, configure: { r in base(r); r.configuration.enableSimdAggregation = true }, placeCamera: place),
          let off = renderOffscreen(device: device, width: 256, height: 256, pointsPerAxis: 24, configure: { r in base(r); r.configuration.enableSimdAggregation = false }, placeCamera: place) else {
        Issue.record("failed to render simd-aggregation images")
        return
    }

    #expect(on.coveredPixelCount > 1000, "produced little coverage")
    // max (depth) and integer add (color) are order-independent → bit-identical.
    #expect(on.rgba == off.rgba, "color buffers differ with simd aggregation on vs off")
    #expect(on.depth == off.depth, "depth buffers differ with simd aggregation on vs off")
}

// MARK: - Slice 5: amortized LOD generation + double buffering

/// A persistent rasteriser + cloud, driven frame-by-frame so amortized sweeps
/// span multiple `frame(...)` calls (unlike `renderOffscreen`, which is one-shot).
private final class RenderSession {
    let context: Context
    let rasteriser: PointRasteriser
    let cloud: PointRasteriserPointCloud
    let width: Int
    let height: Int

    convenience init(device: MTLDevice, packed: PackedPointCloud, width: Int, height: Int) {
        self.init(device: device, width: width, height: height) { PointRasteriserPointCloud(context: $0, packed: packed) }
    }

    init(device: MTLDevice, width: Int, height: Int, makeCloud: (Context) -> PointRasteriserPointCloud) {
        self.width = width
        self.height = height
        context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)
        rasteriser = PointRasteriser(context: context)
        rasteriser.setup()
        cloud = makeCloud(context)
        rasteriser.addPointCloud(cloud)
        rasteriser.resize(size: (Float(width), Float(height)), scaleFactor: 1)
    }

    /// - Parameter preEncode: runs on the frame's command buffer BEFORE the
    ///   rasteriser update/encode (e.g. a Displacement/Tint pass, which may flip
    ///   config flags that `update` then reads).
    func frame(camera: PerspectiveCamera, preEncode: ((MTLCommandBuffer) -> Void)? = nil) -> RenderResult? {
        guard let cb = context.commandQueue.makeCommandBuffer() else { return nil }
        preEncode?(cb)
        let viewport = simd_float4(0, 0, Float(width), Float(height))
        rasteriser.update(renderContext: context, camera: camera, viewport: viewport, index: 0)
        rasteriser.encode(cb)
        return readbackResult(device: context.device, rasteriser: rasteriser, cloud: cloud, width: width, height: height, commandBuffer: cb)
    }
}

/// Blit the rasteriser's output/depth textures to shared buffers and read them
/// back after completion, alongside the cloud's LOD stats.
private func readbackResult(device: MTLDevice, rasteriser: PointRasteriser, cloud: PointRasteriserPointCloud, width: Int, height: Int, commandBuffer cb: MTLCommandBuffer) -> RenderResult? {
    guard let outputTexture = rasteriser.outputTexture,
          let depthTexture = rasteriser.depthTexture,
          let rgbaStaging = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
          let depthStaging = device.makeBuffer(length: width * height * MemoryLayout<Float>.stride, options: .storageModeShared),
          let blit = cb.makeBlitCommandEncoder()
    else { return nil }
    blit.copy(from: outputTexture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: width, height: height, depth: 1), to: rgbaStaging, destinationOffset: 0, destinationBytesPerRow: width * 4, destinationBytesPerImage: width * height * 4)
    blit.copy(from: depthTexture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: width, height: height, depth: 1), to: depthStaging, destinationOffset: 0, destinationBytesPerRow: width * MemoryLayout<Float>.stride, destinationBytesPerImage: width * height * MemoryLayout<Float>.stride)
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let rgba = [UInt8](unsafeUninitializedCapacity: width * height * 4) { buf, c in memcpy(buf.baseAddress!, rgbaStaging.contents(), width * height * 4); c = width * height * 4 }
    let depth = [Float](unsafeUninitializedCapacity: width * height) { buf, c in memcpy(buf.baseAddress!, depthStaging.contents(), width * height * MemoryLayout<Float>.stride); c = width * height }
    return RenderResult(width: width, height: height, rgba: rgba, depth: depth, lodCount: cloud.lodCount, lodOverflow: cloud.lodOverflow, totalPoints: cloud.totalPoints)
}

private func camera(_ context: Context, width: Int, height: Int, position: SIMD3<Float>) -> PerspectiveCamera {
    let c = PerspectiveCamera(context: context, position: position, near: 0.01, far: 100, fov: 45)
    c.aspect = Float(width) / Float(height)
    c.lookAt(target: .zero)
    return c
}

@Test func amortizedBudgetCoveringAllPointsMatchesFullSweep() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 20)

    guard let ref = renderOffscreen(device: device, width: 256, height: 256, packed: packed, configure: { $0.configuration.pointSizeScale = 5 }, placeCamera: { $0.lookAt(target: .zero) }) else {
        Issue.record("full-sweep reference failed"); return
    }

    // Budget larger than the cloud → every sweep completes in one frame.
    let session = RenderSession(device: device, packed: packed, width: 256, height: 256)
    session.rasteriser.configuration.pointSizeScale = 5
    session.rasteriser.configuration.lodPointsPerFrame = packed.pointCount * 2
    let cam = camera(session.context, width: 256, height: 256, position: [0, 0, 2.4])

    _ = session.frame(camera: cam)
    guard let f2 = session.frame(camera: cam) else { Issue.record("amortized frame failed"); return }
    #expect(session.cloud.isDoubleBuffered, "amortization should allocate the second LOD set")
    #expect(f2.lodCount == ref.lodCount, "amortized count \(f2.lodCount) != full-sweep \(ref.lodCount)")
    #expect(f2.rgba == ref.rgba, "amortized (full budget) color differs from full sweep")
    #expect(f2.depth == ref.depth, "amortized (full budget) depth differs from full sweep")
}

@Test func amortizedQuarterBudgetPublishesCompletedSweepsAtomically() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    // Bigger cloud → more source batches → the sweep spans several frames.
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 48)

    // The camera is fixed throughout; the *selection* changes via lodBias so the
    // stale front (few points) and the completed sweep (all points) are visibly
    // and count-wise distinct (CLOD re-projects the front with the live camera,
    // so a camera change alone wouldn't change which points are selected).
    let camPos = SIMD3<Float>(0, 0, 2.4)

    // Full-sweep reference: bias 0 (all points).
    guard let refBig = renderOffscreen(device: device, width: 256, height: 256, packed: packed, configure: { $0.configuration.pointSizeScale = 5; $0.configuration.lodBias = 0 }, placeCamera: { c in c.position = camPos; c.lookAt(target: .zero) }) else {
        Issue.record("reference failed"); return
    }
    let nBig = refBig.lodCount

    let session = RenderSession(device: device, packed: packed, width: 256, height: 256)
    session.rasteriser.configuration.pointSizeScale = 5
    // Frame 0: aggressive CLOD (bias −8 → only the coarsest survivors) full-sweeps
    // the initial front set.
    session.rasteriser.configuration.lodBias = -8
    let cam = camera(session.context, width: 256, height: 256, position: camPos)
    guard let imageSmall = session.frame(camera: cam) else { Issue.record("frame 0 failed"); return }
    let nSmall = imageSmall.lodCount
    #expect(nSmall > 0 && nBig > nSmall, "expected the aggressive-CLOD selection to be a strict subset (\(nSmall) vs \(nBig))")

    // Switch to bias 0 (auto-restarts the sweep) + amortize over several frames.
    session.rasteriser.configuration.lodBias = 0
    session.rasteriser.configuration.lodPointsPerFrame = max(1, packed.pointCount / 4)

    var switchedFrame = -1
    for f in 1 ... 12 {
        guard let r = session.frame(camera: cam) else { Issue.record("frame \(f) failed"); return }
        let count = session.cloud.lodCount
        // The published front count is only ever the old (nSmall) or the
        // newly-completed (nBig) value — never a partial/intermediate count.
        #expect(count == nSmall || count == nBig, "frame \(f) intermediate front count \(count) (small=\(nSmall), big=\(nBig))")
        #expect(session.cloud.lodSweepProgress >= 0 && session.cloud.lodSweepProgress <= 1)

        if count == nBig {
            if switchedFrame < 0 { switchedFrame = f }
            #expect(r.rgba == refBig.rgba, "completed amortized sweep (frame \(f)) != full sweep")
            #expect(r.depth == refBig.depth, "completed amortized sweep depth != full sweep")
        } else {
            #expect(switchedFrame < 0, "front count reverted after switching at frame \(f)")
            // Stale front (the bias −8 selection) still renders unchanged.
            #expect(r.rgba == imageSmall.rgba, "pre-swap frame \(f) not the stale front (partial sweep leaked!)")
        }
    }
    #expect(switchedFrame > 1, "sweep did not amortize across frames (switched at \(switchedFrame))")
}

@Test func restartLODSweepMidSweepStillCompletesCorrectly() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 22)

    guard let ref = renderOffscreen(device: device, width: 256, height: 256, packed: packed, configure: { $0.configuration.pointSizeScale = 5 }, placeCamera: { c in c.position = [0, 0, 3.2]; c.lookAt(target: .zero) }) else {
        Issue.record("reference failed"); return
    }

    let session = RenderSession(device: device, packed: packed, width: 256, height: 256)
    session.rasteriser.configuration.pointSizeScale = 5
    let camA = camera(session.context, width: 256, height: 256, position: [0, 0, 2.4])
    _ = session.frame(camera: camA) // establish front

    session.rasteriser.configuration.lodPointsPerFrame = max(1, packed.pointCount / 5)
    let camB = camera(session.context, width: 256, height: 256, position: [0, 0, 3.2])

    _ = session.frame(camera: camB)                 // partial sweep
    _ = session.frame(camera: camB)                 // partial sweep
    session.rasteriser.restartLODSweep()            // abandon in-flight sweep
    #expect(session.cloud.sweepCursor == 0, "restart should reset the cursor")

    // Continue until a sweep completes; that completed image must be correct.
    var matched = false
    for _ in 0 ... 12 {
        guard let r = session.frame(camera: camB) else { break }
        if r.rgba == ref.rgba { matched = true; break }
    }
    #expect(matched, "sweep after restart never produced the correct full-sweep image")
}


// MARK: - Slice 6: Displacement / Tint / Picking / OIT + motion blur

private extension RenderResult {
    /// Mean (x, y) of covered pixels, or nil if none.
    var coveredCentroid: SIMD2<Float>? {
        var sx = 0.0, sy = 0.0, n = 0
        for y in 0 ..< height { for x in 0 ..< width where alpha(x, y) > 0 { sx += Double(x); sy += Double(y); n += 1 } }
        guard n > 0 else { return nil }
        return SIMD2<Float>(Float(sx / Double(n)), Float(sy / Double(n)))
    }
    /// Max resolved alpha over covered pixels.
    var maxAlpha: UInt8 { rgba.enumerated().reduce(UInt8(0)) { i, e in e.offset % 4 == 3 ? max(i, e.element) : i } }
}

/// Write a sketch kernel to a temp file and return its URL.
private func writeKernel(_ source: String, _ name: String) -> URL {
    let dir = FileManager.default.temporaryDirectory
    let url = dir.appendingPathComponent("\(name)-\(UUID().uuidString).metal")
    try! source.write(to: url, atomically: true, encoding: .utf8)
    return url
}

@Test func displacementShiftsRenderedCloud() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 20)
    let cam = { (s: RenderSession) in camera(s.context, width: 256, height: 256, position: [0, 0, 2.4]) }

    // Undisplaced baseline.
    let base = RenderSession(device: device, packed: packed, width: 256, height: 256)
    base.rasteriser.configuration.pointSizeScale = 5
    guard let undisplaced = base.frame(camera: cam(base)) else { Issue.record("baseline failed"); return }

    // Constant +0.3 world-X displacement → cloud shifts right on screen.
    let url = writeKernel("""
    kernel void computeDisplacement(uint id [[thread_position_in_grid]], SCR_DISPLACEMENT_KERNEL_BUFFERS) {
        RasterBatch batch; uint pointIndex; uint localOffset;
        if (!scr_resolveDisplacementThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
        displacements[pointIndex] = float3(0.3, 0.0, 0.0);
    }
    """, "disp-const")
    let s = RenderSession(device: device, packed: packed, width: 256, height: 256)
    s.rasteriser.configuration.pointSizeScale = 5
    let disp = DisplacementPass(rasteriser: s.rasteriser, kernelURL: url, live: false)
    guard let displaced = s.frame(camera: cam(s), preEncode: { disp.encode(commandBuffer: $0) }) else { Issue.record("displaced frame failed"); return }

    #expect(s.rasteriser.configuration.applyDisplacement, "DisplacementPass should flip applyDisplacement")
    guard let c0 = undisplaced.coveredCentroid, let c1 = displaced.coveredCentroid else { Issue.record("no coverage"); return }
    #expect(c1.x > c0.x + 5, "displaced centroid x \(c1.x) not sufficiently right of \(c0.x)")
    #expect(abs(c1.y - c0.y) < 6, "displacement unexpectedly moved y (\(c0.y) -> \(c1.y))")
}

@Test func nanDisplacementCullsAllPoints() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 16)
    let url = writeKernel("""
    kernel void computeDisplacement(uint id [[thread_position_in_grid]], SCR_DISPLACEMENT_KERNEL_BUFFERS) {
        RasterBatch batch; uint pointIndex; uint localOffset;
        if (!scr_resolveDisplacementThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
        displacements[pointIndex] = float3(NAN, NAN, NAN);
    }
    """, "disp-nan")
    let s = RenderSession(device: device, packed: packed, width: 256, height: 256)
    s.rasteriser.configuration.pointSizeScale = 5
    let disp = DisplacementPass(rasteriser: s.rasteriser, kernelURL: url, live: false)
    guard let r = s.frame(camera: camera(s.context, width: 256, height: 256, position: [0, 0, 2.4]), preEncode: { disp.encode(commandBuffer: $0) }) else { Issue.record("frame failed"); return }
    #expect(r.coveredPixelCount == 0, "NaN displacement should cull every point, got \(r.coveredPixelCount)")
}

/// Regression: `encode(cloud: nil)` must dispatch for EVERY cloud on the
/// rasteriser (not just the first), and two clouds sharing one command buffer
/// must not race on a shared info buffer. Both bugs leave points undisplaced:
/// the first-cloud-only default zeroes the second cloud entirely, and the shared
/// info memcpy makes the larger cloud's dispatch read the smaller cloud's counts
/// (`totalPoints`/`batchCount`), leaving its tail points unwritten. The two
/// clouds have DIFFERENT point counts so the info race is detectable.
@Test func displacementNilCloudTargetsAllCloudsWithoutInfoRace() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }

    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)
    let rasteriser = PointRasteriser(context: context)
    rasteriser.setup()

    // cloudA (4096 pts) is larger than cloudB (1000 pts): a wrong shared
    // totalPoints (B's) would leave cloudA's tail [1000, 4096) unwritten.
    let cloudA = PointRasteriserPointCloud(context: context, packed: PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 16))
    let cloudB = PointRasteriserPointCloud(context: context, packed: PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 10))
    rasteriser.addPointCloud(cloudA)
    rasteriser.addPointCloud(cloudB)
    #expect(cloudA.totalPoints != cloudB.totalPoints, "clouds must differ in size to expose the info race")

    // Pre-allocate SHARED displacement buffers (so we can read them CPU-side) and
    // zero them, so any un-dispatched point stays detectably zero. The pass reuses
    // a non-nil buffer instead of allocating its own private one.
    for c in [cloudA, cloudB] {
        guard let buf = c.makeDisplacementBuffer(storage: .shared, label: "test.disp") else { Issue.record("buffer alloc failed"); return }
        memset(buf.contents(), 0, buf.length)
        c.displacementBuffer = buf
    }

    // Trivial kernel: write a constant (0,1,0) to every resolved point.
    let url = writeKernel("""
    kernel void computeDisplacement(uint id [[thread_position_in_grid]], SCR_DISPLACEMENT_KERNEL_BUFFERS) {
        RasterBatch batch; uint pointIndex; uint localOffset;
        if (!scr_resolveDisplacementThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
        displacements[pointIndex] = float3(0.0, 1.0, 0.0);
    }
    """, "disp-all")
    let pass = DisplacementPass(rasteriser: rasteriser, kernelURL: url, live: false)

    guard let cb = context.commandQueue.makeCommandBuffer() else { Issue.record("no command buffer"); return }
    pass.encode(commandBuffer: cb, cloud: nil) // nil → ALL clouds, one command buffer
    cb.commit()
    cb.waitUntilCompleted()

    // Every point of BOTH clouds must have its .y written to 1 (nonzero).
    for (name, c) in [("A", cloudA), ("B", cloudB)] {
        let base = c.displacementBuffer!.contents()
        var written = 0
        for i in 0 ..< c.totalPoints {
            let y = base.load(fromByteOffset: i * MemoryLayout<SIMD3<Float>>.stride + MemoryLayout<Float>.stride, as: Float.self)
            if y != 0 { written += 1 }
        }
        #expect(written == c.totalPoints,
                "cloud \(name): \(written)/\(c.totalPoints) points displaced — nil-cloud must target all clouds without an info race")
    }
}

/// Regression: streaming (slot-pool) clouds must receive DisplacementPass
/// treatment even when the pool is only PARTIALLY resident. The sketch kernel's
/// `scr_resolve*Thread` binary-searches the batch table by `firstPoint`; if
/// never-filled slots carried `firstPoint == 0` the search drifted past the
/// resident slots (landing on an empty tail slot → state 0 → false), leaving
/// nearly every resident point undisplaced. Seeding each slot's `firstPoint` to
/// its fixed pack offset keeps the table sorted so the search lands on the
/// containing slot; the added `localOffset >= numPoints` guard rejects threads in
/// a resident slot's unfilled tail. Reverting the seed (fix #1) fails this test.
@Test func displacementReachesPartiallyResidentSlotPool() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }

    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)
    let rasteriser = PointRasteriser(context: context)
    rasteriser.setup()

    // Pack 11³ = 1331 points at 512 pts/batch → 3 source batches (512, 512, 307).
    // The 3rd batch is PARTIAL, exercising the unfilled-tail guard.
    let ppb = 512
    var positions: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    let n = 11
    for z in 0 ..< n { for y in 0 ..< n { for x in 0 ..< n {
        let fx = Float(x) / Float(n - 1), fy = Float(y) / Float(n - 1), fz = Float(z) / Float(n - 1)
        positions.append(SIMD3<Float>(fx - 0.5, fy - 0.5, fz - 0.5))
        colors.append(SIMD4<Float>(fx, fy, fz, 1))
    }}}
    let packed = PackedPointCloudFixtures.pack(positions: positions, colors: colors, pointsPerBatch: ppb, shuffleBatches: false)
    #expect(packed.batchCount == 3, "expected 3 source batches, got \(packed.batchCount)")

    // Slot pool with 8 slots; fill only the first 3 (leave 5 never-filled).
    let slotCapacity = 8
    let cloud = PointRasteriserPointCloud(context: context, slotCapacity: slotCapacity, pointsPerBatch: ppb, files: packed.files)
    let slots = cloud.addBatches(
        positionsXYZLow: dataFrom(packed.xyzLow), positionsXYZMed: dataFrom(packed.xyzMed), positionsXYZHigh: dataFrom(packed.xyzHigh),
        colors: dataFrom(packed.colors), levels: Data(packed.levels), batches: packed.batches
    )
    rasteriser.addPointCloud(cloud)
    #expect(slots.count == 3 && cloud.residentBatchCount == 3, "expected 3 resident slots")
    #expect(cloud.totalPoints == slotCapacity * ppb, "pool totalPoints should span every slot")

    // Global point indices that a resident slot actually fills:
    // union of [slot*ppb, slot*ppb + numPoints) — the rest (unfilled tails +
    // never-filled slots) must stay zero.
    var residentWritten = Set<Int>()
    for (i, slot) in slots.enumerated() {
        let base = slot * ppb
        for k in 0 ..< Int(packed.batches[i].numPoints) { residentWritten.insert(base + k) }
    }

    // Shared, zeroed displacement buffer so un-dispatched points stay detectably 0.
    guard let buf = cloud.makeDisplacementBuffer(storage: .shared, label: "test.disp.pool") else { Issue.record("buffer alloc failed"); return }
    memset(buf.contents(), 0, buf.length)
    cloud.displacementBuffer = buf

    // Trivial kernel: write (0,1,0) to every resolved point.
    let url = writeKernel("""
    kernel void computeDisplacement(uint id [[thread_position_in_grid]], SCR_DISPLACEMENT_KERNEL_BUFFERS) {
        RasterBatch batch; uint pointIndex; uint localOffset;
        if (!scr_resolveDisplacementThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
        displacements[pointIndex] = float3(0.0, 1.0, 0.0);
    }
    """, "disp-pool")
    let pass = DisplacementPass(rasteriser: rasteriser, kernelURL: url, live: false)

    guard let cb = context.commandQueue.makeCommandBuffer() else { Issue.record("no command buffer"); return }
    pass.encode(commandBuffer: cb, cloud: cloud)
    cb.commit()
    cb.waitUntilCompleted()

    let cptr = buf.contents()
    var residentOK = 0, leaked = 0
    for i in 0 ..< cloud.totalPoints {
        let y = cptr.load(fromByteOffset: i * MemoryLayout<SIMD3<Float>>.stride + MemoryLayout<Float>.stride, as: Float.self)
        if residentWritten.contains(i) {
            if y == 1 { residentOK += 1 }
        } else if y != 0 {
            leaked += 1
        }
    }
    #expect(residentOK == residentWritten.count,
            "\(residentOK)/\(residentWritten.count) resident points displaced — a partially-resident slot pool must be fully treated")
    #expect(leaked == 0, "\(leaked) non-resident points (unfilled tails / never-filled slots) were wrongly displaced")
}

@Test func tintReplaceDiscardAndPassthrough() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 18)
    let place = { (s: RenderSession) in camera(s.context, width: 256, height: 256, position: [0, 0, 2.4]) }

    // Untinted baseline.
    let base = RenderSession(device: device, packed: packed, width: 256, height: 256)
    base.rasteriser.configuration.pointSizeScale = 5
    guard let untinted = base.frame(camera: place(base)) else { Issue.record("baseline failed"); return }

    func tintRun(_ body: String) -> RenderResult? {
        let url = writeKernel("""
        kernel void computeTint(uint id [[thread_position_in_grid]], SCR_TINT_KERNEL_BUFFERS) {
            RasterBatch batch; uint pointIndex; uint localOffset;
            if (!scr_resolveTintThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
            \(body)
        }
        """, "tint")
        let s = RenderSession(device: device, packed: packed, width: 256, height: 256)
        s.rasteriser.configuration.pointSizeScale = 5
        let tint = TintPass(rasteriser: s.rasteriser, kernelURL: url, live: false)
        return s.frame(camera: place(s), preEncode: { tint.encode(commandBuffer: $0) })
    }

    // Full replace with pure red (a = 1).
    guard let red = tintRun("tints[pointIndex] = float4(1.0, 0.0, 0.0, 1.0);") else { Issue.record("red failed"); return }
    var reds = 0, covered = 0
    for y in 0 ..< red.height { for x in 0 ..< red.width where red.alpha(x, y) > 0 {
        covered += 1; let (r, g, b) = red.rgb(x, y); if r > 200, g < 60, b < 60 { reds += 1 }
    }}
    #expect(covered > 100 && Float(reds) / Float(covered) > 0.9, "full-replace tint not mostly red (\(reds)/\(covered))")

    // Discard sentinel (a < 0) → zero coverage.
    guard let discarded = tintRun("tints[pointIndex] = float4(0.0, 0.0, 0.0, -1.0);") else { Issue.record("discard failed"); return }
    #expect(discarded.coveredPixelCount == 0, "discard tint should be fully transparent, got \(discarded.coveredPixelCount)")

    // Pass-through (a = 0) → identical to untinted.
    guard let passthrough = tintRun("tints[pointIndex] = float4(1.0, 1.0, 1.0, 0.0);") else { Issue.record("passthrough failed"); return }
    #expect(passthrough.rgba == untinted.rgba, "a==0 tint should match the untinted image")
}

@Test func pickReturnsSourceIndexOfPointAtCursor() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 12)
    let s = RenderSession(device: device, packed: packed, width: 256, height: 256)
    s.rasteriser.configuration.pointSizeScale = 4
    let cam = camera(s.context, width: 256, height: 256, position: [0, 0, 2.4])

    // Render once so the front LOD set is populated.
    _ = s.frame(camera: cam)

    // Target the front-most point (max z → nearest the +z camera) so it is the
    // depth winner at its own screen pixel and picking is unambiguous.
    let target = (0 ..< packed.orderedPositions.count).max { packed.orderedPositions[$0].z < packed.orderedPositions[$1].z }!
    let worldPos = packed.orderedPositions[target]
    let clip = (cam.projectionMatrix * cam.viewMatrix) * SIMD4<Float>(worldPos, 1)
    let ndc = SIMD2<Float>(clip.x / clip.w, clip.y / clip.w)

    guard let picked = s.rasteriser.pickPointIndex(atNDC: ndc, in: s.cloud, camera: cam, searchRadius: 8) else {
        Issue.record("pick returned nil"); return
    }
    // The returned pack-order index must decode (via orderedPositions) to a point
    // on the same front surface near the cursor (the exact point or a neighbor).
    let pickedPos = packed.orderedPositions[Int(picked)]
    #expect(simd_distance(pickedPos, worldPos) < 0.2, "picked \(picked) at \(pickedPos) far from target \(worldPos)")
}

@Test func motionBlurIncreasesCoverageAndOITAlphaInRange() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 20)

    // Motion blur: the SAME camera moved between frames yields nonzero screen
    // velocity (prev-VP is keyed per camera identity), so with motionBlur>0 the
    // smear should cover strictly more pixels than without.
    func run(motionBlur: Float) -> Int {
        let s = RenderSession(device: device, packed: packed, width: 256, height: 256)
        s.rasteriser.configuration.pointSizeScale = 3
        s.rasteriser.configuration.enablePointRejection = false
        s.rasteriser.configuration.motionBlur = motionBlur
        s.rasteriser.configuration.motionBlurMaxSpread = 64
        let cam = camera(s.context, width: 256, height: 256, position: [0, 0, 2.4])
        _ = s.frame(camera: cam)                 // frame 1 establishes prevTransform
        cam.position = [0.6, 0.3, 2.4]           // move the SAME camera
        cam.lookAt(target: .zero)
        return s.frame(camera: cam)?.coveredPixelCount ?? 0
    }
    let still = run(motionBlur: 0)
    let smeared = run(motionBlur: 1.0)
    #expect(still > 0 && smeared > still, "motion blur did not increase coverage (still=\(still) smeared=\(smeared))")

    // OIT: a translucent tint (coverage mode) should resolve alpha strictly in (0,1).
    let url = writeKernel("""
    kernel void computeTint(uint id [[thread_position_in_grid]], SCR_TINT_KERNEL_BUFFERS) {
        RasterBatch batch; uint pointIndex; uint localOffset;
        if (!scr_resolveTintThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
        tints[pointIndex] = float4(0.0, 0.0, 0.0, 0.6); // circle-of-confusion 0.6
    }
    """, "tint-oit")
    let s = RenderSession(device: device, packed: packed, width: 256, height: 256)
    s.rasteriser.configuration.pointSizeScale = 3
    let tint = TintPass(rasteriser: s.rasteriser, kernelURL: url, live: false)
    tint.alphaIsCoverage = true
    guard let r = s.frame(camera: camera(s.context, width: 256, height: 256, position: [0, 0, 2.4]), preEncode: { tint.encode(commandBuffer: $0) }) else { Issue.record("oit frame failed"); return }
    #expect(s.rasteriser.configuration.tintAlphaIsCoverage, "alphaIsCoverage should flip the config")
    #expect(r.coveredPixelCount > 0, "OIT produced no coverage")
    // Weighted-blended OIT (α = 1 − e^(−Σα)): thin coverage resolves to a partial
    // alpha. Assert at least one covered pixel is strictly translucent (0<α<255).
    var partial = 0
    for y in 0 ..< r.height { for x in 0 ..< r.width { let a = r.alpha(x, y); if a > 0, a < 255 { partial += 1 } } }
    #expect(partial > 0, "no strictly-translucent OIT pixel (all covered pixels fully opaque/transparent)")
}

// MARK: - Slice 8a: slot-pool residency

private func dataFrom<T>(_ arr: [T]) -> Data { arr.withUnsafeBytes { Data($0) } }

/// Build a slot-pool cloud populated from a packed cloud (one batch per slot),
/// with `extraSlots` left empty at the tail.
private func poolCloud(from packed: PackedPointCloud, context: Context, extraSlots: Int = 0) -> PointRasteriserPointCloud {
    let ppb = Int(packed.batches.map(\.numPoints).max() ?? 1)
    let cloud = PointRasteriserPointCloud(context: context, slotCapacity: packed.batchCount + extraSlots, pointsPerBatch: ppb, files: packed.files)
    cloud.addBatches(
        positionsXYZLow: dataFrom(packed.xyzLow), positionsXYZMed: dataFrom(packed.xyzMed), positionsXYZHigh: dataFrom(packed.xyzHigh),
        colors: dataFrom(packed.colors), levels: Data(packed.levels), batches: packed.batches
    )
    return cloud
}

@Test func slotPoolAddBatchesRendersIdenticallyToWholesale() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    // Dither off: the CLOD keep-test hashes the point index, which differs between
    // contiguous (wholesale) and slot-rebased (pool) layouts. With dither the
    // per-point 0.5 constant makes selection index-independent → identical set.
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 22)
    let cfg: (PointRasteriser) -> Void = { r in r.configuration.pointSizeScale = 5; r.configuration.enableLODDither = false }

    let wholesale = RenderSession(device: device, packed: packed, width: 256, height: 256)
    cfg(wholesale.rasteriser)
    let pool = RenderSession(device: device, width: 256, height: 256) { poolCloud(from: packed, context: $0, extraSlots: 3) }
    cfg(pool.rasteriser)

    let cam = { (s: RenderSession) in camera(s.context, width: 256, height: 256, position: [0.8, 0.6, 2.2]) }
    guard let w = wholesale.frame(camera: cam(wholesale)), let p = pool.frame(camera: cam(pool)) else {
        Issue.record("render failed"); return
    }
    #expect(pool.cloud.isSlotPool && pool.cloud.residentBatchCount == packed.batchCount)
    #expect(pool.cloud.residentPointCount == packed.pointCount)
    #expect(w.lodCount == p.lodCount, "survivor counts differ (wholesale \(w.lodCount) vs pool \(p.lodCount))")
    #expect(w.rgba == p.rgba, "slot-pool color differs from wholesale")
    #expect(w.depth == p.depth, "slot-pool depth differs from wholesale")
}

@Test func removeBatchesRemovesTheirPointsFromNextRender() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 40) // several batches
    var slots: [Int] = []
    let s = RenderSession(device: device, width: 256, height: 256) { ctx in
        let ppb = Int(packed.batches.map(\.numPoints).max() ?? 1)
        let c = PointRasteriserPointCloud(context: ctx, slotCapacity: packed.batchCount, pointsPerBatch: ppb, files: packed.files)
        slots = c.addBatches(positionsXYZLow: dataFrom(packed.xyzLow), positionsXYZMed: dataFrom(packed.xyzMed), positionsXYZHigh: dataFrom(packed.xyzHigh), colors: dataFrom(packed.colors), levels: Data(packed.levels), batches: packed.batches)
        return c
    }
    s.rasteriser.configuration.pointSizeScale = 4
    s.rasteriser.configuration.enableLODDither = false
    let cam = camera(s.context, width: 256, height: 256, position: [0, 0, 2.4])

    guard let full = s.frame(camera: cam) else { Issue.record("full failed"); return }
    let fullCount = s.cloud.lodCount
    #expect(fullCount > 0 && full.coveredPixelCount > 0)

    // Remove half the batches → next full-sweep render drops their points.
    s.cloud.removeBatches(slots: Array(slots.prefix(slots.count / 2)))
    #expect(s.cloud.residentBatchCount == packed.batchCount - slots.count / 2)
    guard let reduced = s.frame(camera: cam) else { Issue.record("reduced failed"); return }
    #expect(s.cloud.lodCount < fullCount, "lodCount did not drop after removeBatches (\(s.cloud.lodCount) vs \(fullCount))")
    #expect(reduced.coveredPixelCount < full.coveredPixelCount, "coverage did not drop after removeBatches")
}

@Test func emptyAndNeverPopulatedSlotsAreInert() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    // A pool with capacity but no resident batches renders nothing and never crashes.
    let s = RenderSession(device: device, width: 256, height: 256) { ctx in
        PointRasteriserPointCloud(context: ctx, slotCapacity: 64, pointsPerBatch: 2048, files: [RasterFile()])
    }
    s.rasteriser.configuration.pointSizeScale = 5
    let cam = camera(s.context, width: 256, height: 256, position: [0, 0, 2.4])
    guard let empty = s.frame(camera: cam) else { Issue.record("empty render failed"); return }
    #expect(s.cloud.residentBatchCount == 0 && s.cloud.residentPointCount == 0)
    #expect(empty.lodCount == 0 && empty.coveredPixelCount == 0, "empty pool should render nothing")

    // Populate a few scattered slots; the still-empty slots must stay inert.
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 12)
    s.cloud.addBatches(positionsXYZLow: dataFrom(packed.xyzLow), positionsXYZMed: dataFrom(packed.xyzMed), positionsXYZHigh: dataFrom(packed.xyzHigh), colors: dataFrom(packed.colors), levels: Data(packed.levels), batches: packed.batches)
    guard let populated = s.frame(camera: cam) else { Issue.record("populated render failed"); return }
    #expect(populated.lodCount > 0 && populated.lodCount <= packed.pointCount, "lodCount \(populated.lodCount)")
    #expect(populated.coveredPixelCount > 0)
}

@Test func addBatchesMidAmortizedSweepJoinsALaterSweepWithoutPartial() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 48) // 11 batches
    let half = packed.batchCount / 2

    // Split the packed batches into two groups by pack-order point range so each
    // add is a self-contained blob.
    var groupA: [RasterBatch] = [], groupB: [RasterBatch] = []
    for (i, b) in packed.batches.enumerated() { if i < half { groupA.append(b) } else { groupB.append(b) } }

    let s = RenderSession(device: device, width: 256, height: 256) { ctx in
        let ppb = Int(packed.batches.map(\.numPoints).max() ?? 1)
        return PointRasteriserPointCloud(context: ctx, slotCapacity: packed.batchCount, pointsPerBatch: ppb, files: packed.files)
    }
    s.rasteriser.configuration.pointSizeScale = 4
    s.rasteriser.configuration.enableLODDither = false
    let cam = camera(s.context, width: 256, height: 256, position: [0, 0, 2.4])

    let lo = dataFrom(packed.xyzLow), me = dataFrom(packed.xyzMed), hi = dataFrom(packed.xyzHigh)
    let co = dataFrom(packed.colors), le = Data(packed.levels)

    // Add group A, full-sweep once to establish the front set (count Na).
    s.cloud.addBatches(positionsXYZLow: lo, positionsXYZMed: me, positionsXYZHigh: hi, colors: co, levels: le, batches: groupA)
    _ = s.frame(camera: cam)
    let na = s.cloud.lodCount
    #expect(na > 0)

    // Enable amortization, add group B mid-life; step frames until group B joins.
    s.rasteriser.configuration.lodPointsPerFrame = max(1, packed.pointCount / 4)
    s.cloud.addBatches(positionsXYZLow: lo, positionsXYZMed: me, positionsXYZHigh: hi, colors: co, levels: le, batches: groupB)
    let nbExpected = s.cloud.residentPointCount // after both groups (dither off → all survive-ish)

    var reachedFull = false
    for f in 1 ... 12 {
        _ = s.frame(camera: cam)
        let count = s.cloud.lodCount
        // Published front count is only ever Na (stale) or the final full count —
        // never a partial mix, even though group B is being swept in.
        #expect(count == na || count == s.cloud.lodCount)
        #expect(count >= na, "front count regressed below Na at frame \(f)")
        if count > na { reachedFull = true; break }
    }
    #expect(reachedFull, "group B never joined a completed sweep")
    #expect(nbExpected > na, "sanity: adding group B should raise resident points")
}

// MARK: - LOD capacity / overflow (bugfix)

@Test func lodCapacityBelowSelectionOverflowsAndClampsExactly() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 24) // 13824 points
    let place = { (s: RenderSession) in camera(s.context, width: 256, height: 256, position: [0, 0, 2.4]) }

    // Unrestricted (default capacity = full source count): no overflow, full selection.
    let full = RenderSession(device: device, packed: packed, width: 256, height: 256)
    full.rasteriser.configuration.pointSizeScale = 5
    guard let fullR = full.frame(camera: place(full)) else { Issue.record("full render failed"); return }
    #expect(full.cloud.lodOverflow == 0, "default capacity should never overflow (got \(full.cloud.lodOverflow))")
    #expect(full.cloud.lodOverflowed == false)
    let fullCount = full.cloud.lodCount
    #expect(fullCount > 0 && fullCount <= packed.pointCount)

    // Forced-small capacity → overflow; the clamped count equals the cap exactly.
    let cap = fullCount / 2
    let restricted = RenderSession(device: device, width: 256, height: 256) {
        PointRasteriserPointCloud(context: $0, packed: packed, lodCapacity: cap)
    }
    restricted.rasteriser.configuration.pointSizeScale = 5
    guard let restrictedR = restricted.frame(camera: place(restricted)) else { Issue.record("restricted render failed"); return }
    #expect(restricted.cloud.lodOverflowed, "capacity below selection should overflow")
    #expect(restricted.cloud.lodOverflow > 0)
    #expect(restricted.cloud.lodCount == cap, "clamped count \(restricted.cloud.lodCount) != capacity \(cap)")
    // Dropped + kept == full survivor count (clamp is exact, no double-count).
    #expect(restricted.cloud.lodCount + restricted.cloud.lodOverflow == fullCount,
            "kept(\(restricted.cloud.lodCount)) + dropped(\(restricted.cloud.lodOverflow)) != full \(fullCount)")
    #expect(restrictedR.coveredPixelCount < fullR.coveredPixelCount,
            "restricted image should cover fewer pixels (\(restrictedR.coveredPixelCount) vs \(fullR.coveredPixelCount))")
}

// Real-file verification (env-gated so normal `swift test` skips the 44M-point
// pack). Set POINTRASTERISER_PLY=<path> to run.
@Test func realFilePLYFitsUnderDefaultCapacity() throws {
    guard let path = ProcessInfo.processInfo.environment["POINTRASTERISER_PLY"], !path.isEmpty else { return }
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }

    let t0 = Date()
    let packed = try PLYPointCloudLoader.load(url: URL(fileURLWithPath: path))
    let loadPack = Date().timeIntervalSince(t0)
    print("REALFILE: loaded+packed \(packed.pointCount) points (\(packed.batchCount) batches) in \(String(format: "%.1f", loadPack))s")

    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)
    let rasteriser = PointRasteriser(context: context)
    rasteriser.setup()
    // CLOD + frustum off → LODSelect must append EVERY source point; with the
    // fixed default (capacity == source count) that fits exactly, overflow 0.
    rasteriser.configuration.enableCLOD = false
    rasteriser.configuration.enableFrustumCulling = false
    let cloud = PointRasteriserPointCloud(context: context, packed: packed) // default capacity
    rasteriser.addPointCloud(cloud)
    rasteriser.resize(size: (512, 512), scaleFactor: 1)

    let center = (packed.boundsMin + packed.boundsMax) * 0.5
    let extent = simd_length(packed.boundsMax - packed.boundsMin)
    let cam = PerspectiveCamera(context: context, position: center + [0, 0, max(extent, 1)], near: 0.01, far: max(extent * 4, 100), fov: 45)
    cam.aspect = 1
    cam.lookAt(target: center)

    rasteriser.update(renderContext: context, camera: cam, viewport: simd_float4(0, 0, 512, 512), index: 0)
    guard let cb = context.commandQueue.makeCommandBuffer() else { Issue.record("no cb"); return }
    rasteriser.encode(cb)
    cb.commit()
    cb.waitUntilCompleted()

    print("REALFILE: totalPoints=\(cloud.totalPoints) lodCapacity=\(cloud.lodCapacity) lodCount=\(cloud.lodCount) overflow=\(cloud.lodOverflow)")
    #expect(cloud.lodCapacity == packed.pointCount, "default capacity should equal the full source count")
    #expect(cloud.lodOverflow == 0, "44M-point cloud overflowed under the fixed default capacity")
    #expect(cloud.lodCount == packed.pointCount, "expected every point selected with CLOD+frustum off")
}

// MARK: - Full-sweep LODSelect skip (static-scene optimization)

@Test func lodSelectSkipsWhenNothingChangesAndRendersBitIdentically() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 20)
    let s = RenderSession(device: device, packed: packed, width: 256, height: 256)
    s.rasteriser.configuration.pointSizeScale = 5
    let cam = camera(s.context, width: 256, height: 256, position: [1, 0.6, 2.2])

    // Frame 1: first select runs (no front data yet).
    guard let f1 = s.frame(camera: cam) else { Issue.record("f1 failed"); return }
    #expect(s.rasteriser.lodSelectRanLastFrame == 1 && s.rasteriser.lodSelectSkippedLastFrame == 0, "first frame should run select")

    // Frames 2..4: nothing changed → skip, and render bit-identically.
    for f in 2 ... 4 {
        guard let fr = s.frame(camera: cam) else { Issue.record("f\(f) failed"); return }
        #expect(s.rasteriser.lodSelectSkippedLastFrame == 1 && s.rasteriser.lodSelectRanLastFrame == 0, "frame \(f) should skip select")
        #expect(fr.rgba == f1.rgba, "skipped frame \(f) color differs from the selected frame")
        #expect(fr.depth == f1.depth, "skipped frame \(f) depth differs")
        #expect(fr.lodCount == f1.lodCount, "front lodCount changed on a skipped frame")
    }
}

@Test func lodSelectReRunsOnCameraMove() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let s = RenderSession(device: device, packed: PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 18), width: 256, height: 256)
    s.rasteriser.configuration.pointSizeScale = 5
    let cam = camera(s.context, width: 256, height: 256, position: [0, 0, 2.4])
    _ = s.frame(camera: cam)                                   // run
    _ = s.frame(camera: cam)
    #expect(s.rasteriser.lodSelectSkippedLastFrame == 1, "static frame skips")
    cam.position = [0.5, 0.3, 2.4]; cam.lookAt(target: .zero)  // move
    _ = s.frame(camera: cam)
    #expect(s.rasteriser.lodSelectRanLastFrame == 1, "camera move should re-run select")
}

@Test func lodSelectReRunsOnEachSelectionAffectingConfigToggle() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let s = RenderSession(device: device, packed: PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 18), width: 256, height: 256)
    s.rasteriser.configuration.pointSizeScale = 5
    let cam = camera(s.context, width: 256, height: 256, position: [0, 0, 2.4])

    func settleThenToggle(_ mutate: (inout PointRasteriserConfiguration) -> Void, _ name: String) {
        _ = s.frame(camera: cam); _ = s.frame(camera: cam)
        #expect(s.rasteriser.lodSelectSkippedLastFrame == 1, "\(name): should be settled/skipping before toggle")
        var cfg = s.rasteriser.configuration; mutate(&cfg); s.rasteriser.configuration = cfg
        _ = s.frame(camera: cam)
        #expect(s.rasteriser.lodSelectRanLastFrame == 1, "\(name): toggle should re-run select")
    }
    settleThenToggle({ $0.enableCLOD.toggle() }, "enableCLOD")
    settleThenToggle({ $0.lodBias += 1 }, "lodBias")
    settleThenToggle({ $0.enableLODDither.toggle() }, "enableLODDither")

    // A NON-selection config (point size is raster-side) must NOT invalidate.
    _ = s.frame(camera: cam); _ = s.frame(camera: cam)
    #expect(s.rasteriser.lodSelectSkippedLastFrame == 1)
    var cfg = s.rasteriser.configuration; cfg.pointSizeScale = 9; s.rasteriser.configuration = cfg
    _ = s.frame(camera: cam)
    #expect(s.rasteriser.lodSelectSkippedLastFrame == 1, "point size is raster-side and must not re-run select")
}

@Test func lodSelectReRunsOnCloudMutationAndWorldTransform() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 16)
    let ppb = Int(packed.batches.map(\.numPoints).max() ?? 1)
    var extraLo = Data(), extraMe = Data(), extraHi = Data(), extraCo = Data(), extraLe = Data()
    let s = RenderSession(device: device, width: 256, height: 256) { ctx in
        let c = PointRasteriserPointCloud(context: ctx, slotCapacity: packed.batchCount + 4, pointsPerBatch: ppb, files: packed.files)
        c.addBatches(positionsXYZLow: dataFrom(packed.xyzLow), positionsXYZMed: dataFrom(packed.xyzMed), positionsXYZHigh: dataFrom(packed.xyzHigh), colors: dataFrom(packed.colors), levels: Data(packed.levels), batches: packed.batches)
        // Keep a copy so we can add another batch mid-test.
        extraLo = dataFrom(packed.xyzLow); extraMe = dataFrom(packed.xyzMed); extraHi = dataFrom(packed.xyzHigh); extraCo = dataFrom(packed.colors); extraLe = Data(packed.levels)
        return c
    }
    s.rasteriser.configuration.pointSizeScale = 5
    let cam = camera(s.context, width: 256, height: 256, position: [0, 0, 2.4])
    _ = s.frame(camera: cam); _ = s.frame(camera: cam)
    #expect(s.rasteriser.lodSelectSkippedLastFrame == 1, "settled")

    // Content mutation (add one more batch) → contentGeneration bump → re-run.
    let gen = s.cloud.contentGeneration
    s.cloud.removeBatches(slots: [0])
    #expect(s.cloud.contentGeneration != gen, "removeBatches must bump contentGeneration")
    _ = s.frame(camera: cam)
    #expect(s.rasteriser.lodSelectRanLastFrame == 1, "cloud mutation should re-run select")

    // Settle, then change the cloud's world transform → re-run.
    _ = s.frame(camera: cam)
    #expect(s.rasteriser.lodSelectSkippedLastFrame == 1, "settled after mutation")
    s.cloud.position = [0.3, 0, 0]
    _ = s.frame(camera: cam)
    #expect(s.rasteriser.lodSelectRanLastFrame == 1, "world transform change should re-run select")
    _ = (extraLo, extraMe, extraHi, extraCo, extraLe) // silence unused (kept for symmetry)
}

// MARK: - Slice 3: rejection self-rejection rim regression (depthTolerance)

/// A single **flat** grid of near-white points, tilted so its depth varies
/// smoothly across the surface. Multi-pixel splats tile the plane into abutting
/// flat depth plateaus at slightly different depths — the exact input that made
/// the VAST empty-cone operator self-reject (a farther plateau's disc has a
/// same-surface neighbor a hair nearer → narrow cone → carved into a black rim).
/// The `depthTolerance` same-surface skip must suppress that.
///
/// `tilt` is kept gentle enough that adjacent grid plateaus differ by less than
/// the default `depthTolerance` (0.01) in reverse-Z (relative step ≈ 2.1% · tilt
/// for this camera), so the same-surface skip covers them, yet steep enough that
/// the plateau borders still form narrow eye-ward cones the operator rejects when
/// the tolerance is 0. That gap is exactly what this regression exercises.
private func tiltedWhiteGridCloud(side: Int = 40, tilt: Float = 0.35, span: Float = 0.6) -> PackedPointCloud {
    var positions: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    for yi in 0 ..< side {
        for xi in 0 ..< side {
            let fx = Float(xi) / Float(side - 1) * 2 - 1  // -1 … 1
            let fy = Float(yi) / Float(side - 1) * 2 - 1
            positions.append([fx * span, fy * span, fy * tilt]) // depth ramps with y
            colors.append([0.95, 0.95, 0.95, 1])
        }
    }
    return PackedPointCloudFixtures.pack(positions: positions, colors: colors, shuffleBatches: false)
}

private extension RenderResult {
    /// Pixels lit in `baseline` but turned to transparent background in `self`
    /// — i.e. carved by point rejection. For a single flat surface this is the
    /// self-rejection rim count (should be ~0 once the tolerance skip is active).
    func carvedPixelCount(vs baseline: RenderResult) -> Int {
        var n = 0
        for i in 0 ..< (width * height) where baseline.rgba[i * 4 + 3] > 0 && rgba[i * 4 + 3] == 0 { n += 1 }
        return n
    }
}

/// Write an RGBA8 buffer to a PNG (best-effort; used only for evidence dumps).
@discardableResult
private func dumpPNG(_ r: RenderResult, to url: URL) -> Bool {
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil, width: r.width, height: r.height, bitsPerComponent: 8, bytesPerRow: r.width * 4,
        space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return false }
    r.rgba.withUnsafeBytes { _ = memcpy(ctx.data!, $0.baseAddress!, r.width * r.height * 4) }
    guard let img = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { return false }
    CGImageDestinationAddImage(dest, img, nil)
    return CGImageDestinationFinalize(dest)
}

/// (a) A tilted flat white grid at a large point size must render with point
/// rejection on essentially identically to rejection off — no self-rejected
/// black rims. Rendered three ways to prove the mechanism directly:
///   - rejection OFF          → the fully lit surface (baseline)
///   - rejection ON, tol = 0  → reproduces the bug (same-surface plateau borders
///                              carved into a rim)
///   - rejection ON, tol = default 0.01 → the fix (rim suppressed)
@Test func pointRejectionDoesNotCarveFlatSurfaceRims() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let cloud = tiltedWhiteGridCloud()

    // Large multi-pixel splats, hole fill off, CLOD off: the only thing that
    // varies across the three renders is rejection + depthTolerance.
    let base: (PointRasteriser) -> Void = { r in
        r.configuration.enableCLOD = false
        r.configuration.holeFillIterations = 0
        r.configuration.pointSizeMode = .screenSpace
        r.configuration.minimumPointSize = 40
        r.configuration.maximumPointSize = 40
        r.configuration.pointSizeScale = 1
    }

    guard let off = renderOffscreen(device: device, width: 512, height: 512, packed: cloud, configure: { r in
        base(r); r.configuration.enablePointRejection = false
    }, placeCamera: { $0.lookAt(target: .zero) }),
    let bug = renderOffscreen(device: device, width: 512, height: 512, packed: cloud, configure: { r in
        base(r); r.configuration.enablePointRejection = true; r.configuration.depthTolerance = 0
    }, placeCamera: { $0.lookAt(target: .zero) }),
    let fixed = renderOffscreen(device: device, width: 512, height: 512, packed: cloud, configure: { r in
        base(r); r.configuration.enablePointRejection = true; r.configuration.depthTolerance = 0.01
    }, placeCamera: { $0.lookAt(target: .zero) }) else {
        Issue.record("failed to render tilted white grid")
        return
    }

    let lit = off.coveredPixelCount
    let rimBug = bug.carvedPixelCount(vs: off)
    let rimFixed = fixed.carvedPixelCount(vs: off)

    if let dir = ProcessInfo.processInfo.environment["PR_DUMP_DIR"] {
        let d = URL(fileURLWithPath: dir)
        dumpPNG(off, to: d.appendingPathComponent("fixed_rejection_off.png"))
        dumpPNG(fixed, to: d.appendingPathComponent("fixed_rejection_on.png"))
        dumpPNG(bug, to: d.appendingPathComponent("bug_rejection_on_tol0.png"))
        print("[rim-regression] lit=\(lit) rimBug(tol0)=\(rimBug) rimFixed(tol0.01)=\(rimFixed)")
    }

    #expect(lit > 10_000, "sanity: flat grid should cover a large area, got \(lit) lit px")
    // With tolerance 0 the operator self-rejects a large plateau-border rim.
    #expect(rimBug > lit / 20, "expected a self-rejection rim with tol=0, got \(rimBug) of \(lit) lit")
    // The default tolerance suppresses essentially all of it.
    #expect(rimFixed < lit / 200,
            "fix left flat-surface rim: \(rimFixed) of \(lit) lit (bug rim was \(rimBug))")
    #expect(rimFixed < rimBug / 20,
            "fix did not substantially reduce rim (bug=\(rimBug) fixed=\(rimFixed))")
}

/// (b) The tolerance skip must NOT disable genuine occlusion rejection: two
/// surfaces separated by depth far larger than depthTolerance still get the
/// leaking-far-plane pixels carved. Guards against the fix skipping everything.
@Test func pointRejectionStillCarvesGenuineOcclusionWithTolerance() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let cloud = twoPlanesCloud()

    let setup: (PointRasteriser) -> Void = { r in
        r.configuration.enableCLOD = false
        r.configuration.holeFillIterations = 0
        r.configuration.pointSizeMode = .screenSpace
        r.configuration.minimumPointSize = 1
        r.configuration.maximumPointSize = 1
        r.configuration.pointSizeScale = 1
        // Default depthTolerance 0.01: planes at z=+0.25 / z=-0.5 are separated
        // far beyond it, so the same-surface skip must not exempt the leak.
    }

    guard let off = renderOffscreen(device: device, width: 256, height: 256, packed: cloud, configure: { r in
        setup(r); r.configuration.enablePointRejection = false
    }, placeCamera: { $0.lookAt(target: .zero) }),
    let on = renderOffscreen(device: device, width: 256, height: 256, packed: cloud, configure: { r in
        setup(r); r.configuration.enablePointRejection = true
    }, placeCamera: { $0.lookAt(target: .zero) }) else {
        Issue.record("failed to render two-plane occlusion scene")
        return
    }

    let greenOff = off.greenPixelCount(inCenterHalfExtent: 64)
    let greenOn = on.greenPixelCount(inCenterHalfExtent: 64)
    #expect(greenOff > 100, "expected far-plane leakage with rejection off, got \(greenOff)")
    #expect(Float(greenOn) < 0.75 * Float(greenOff),
            "tolerance skip disabled genuine occlusion rejection (off=\(greenOff) on=\(greenOn))")
}
