#if canImport(Metal)
import Foundation
import Metal
import Satin
import simd
import Testing
@testable import SatinPointRasteriser

// In-place wholesale replace (identity / grow / shrink / GPU==CPU / amortized)
// and the sub-region viewport composite. GPU-less CI hosts return early.

private struct RGBAImage {
    let width: Int
    let height: Int
    let rgba: [UInt8]

    func alpha(_ x: Int, _ y: Int) -> UInt8 { rgba[(y * width + x) * 4 + 3] }
    var coveredPixelCount: Int {
        var n = 0
        for i in 0 ..< (width * height) where rgba[i * 4 + 3] > 0 { n += 1 }
        return n
    }
    /// Covered pixels whose dominant channel is red / green (solid-colored clouds).
    func dominantCounts() -> (red: Int, green: Int) {
        var red = 0, green = 0
        for i in 0 ..< (width * height) where rgba[i * 4 + 3] > 0 {
            let r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2]
            if r > g && r > b { red += 1 }
            else if g > r && g > b { green += 1 }
        }
        return (red, green)
    }
}

/// A solid-colored `pointsPerAxis³` grid spanning `[-span/2, span/2]`.
private func solidGrid(pointsPerAxis n: Int, color: SIMD4<Float>, span: Float = 1) -> ([SIMD3<Float>], [SIMD4<Float>]) {
    var positions: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    let count = max(n, 2)
    for z in 0 ..< count {
        for y in 0 ..< count {
            for x in 0 ..< count {
                let fx = Float(x) / Float(count - 1) - 0.5
                let fy = Float(y) / Float(count - 1) - 0.5
                let fz = Float(z) / Float(count - 1) - 0.5
                positions.append(SIMD3<Float>(fx, fy, fz) * span)
                colors.append(color)
            }
        }
    }
    return (positions, colors)
}

private let red = SIMD4<Float>(1, 0, 0, 1)
private let green = SIMD4<Float>(0, 1, 0, 1)

/// Minimal offscreen harness that renders a caller-owned cloud so the same
/// instance can be re-rendered after an in-place replace.
private final class ReplaceHarness {
    let context: Context
    let device: MTLDevice
    let rasteriser: PointRasteriser
    let width = 256, height = 256
    let camera: PerspectiveCamera

    init(device: MTLDevice, configure: (inout PointRasteriserConfiguration) -> Void = { _ in }) {
        self.device = device
        context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)
        rasteriser = PointRasteriser(context: context)
        rasteriser.setup()
        var config = PointRasteriserConfiguration()
        config.enableCLOD = false
        config.enableFrustumCulling = false
        config.holeFillIterations = 0
        config.pointSizeScale = 6
        configure(&config)
        rasteriser.configuration = config
        rasteriser.resize(size: (Float(width), Float(height)), scaleFactor: 1)
        camera = PerspectiveCamera(context: context, position: [0, 0, 2.4], near: 0.01, far: 100, fov: 45)
        camera.aspect = Float(width) / Float(height)
        camera.lookAt(target: .zero)
    }

    func render() -> RGBAImage? {
        let viewport = simd_float4(0, 0, Float(width), Float(height))
        rasteriser.update(renderContext: context, camera: camera, viewport: viewport, index: 0)
        guard let cb = context.commandQueue.makeCommandBuffer() else { return nil }
        rasteriser.encode(cb)
        guard let out = rasteriser.outputTexture,
              let staging = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let blit = cb.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: out, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: staging, destinationOffset: 0, destinationBytesPerRow: width * 4, destinationBytesPerImage: width * height * 4)
        blit.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        let rgba = [UInt8](unsafeUninitializedCapacity: width * height * 4) { buf, c in
            memcpy(buf.baseAddress!, staging.contents(), width * height * 4); c = width * height * 4
        }
        return RGBAImage(width: width, height: height, rgba: rgba)
    }
}

// (a) identity — replace renders new data on the SAME instance; gen strictly ↑.
@Test func replaceKeepsIdentityAndRendersNewContent() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let h = ReplaceHarness(device: device)

    let (rp, rc) = solidGrid(pointsPerAxis: 16, color: red)
    let cloud = PointRasteriserPointCloud(context: h.context, packed: PackedPointCloudFixtures.pack(positions: rp, colors: rc))
    let identity = ObjectIdentifier(cloud)
    h.rasteriser.addPointCloud(cloud)

    guard let before = h.render() else { Issue.record("render 1 failed"); return }
    let gen0 = cloud.contentGeneration
    #expect(before.dominantCounts().red > 0, "expected red before replace")

    let (gp, gc) = solidGrid(pointsPerAxis: 16, color: green)
    cloud.replacePackedPointCloud(PackedPointCloudFixtures.pack(positions: gp, colors: gc))

    #expect(cloud.contentGeneration > gen0, "replace must bump contentGeneration")
    #expect(ObjectIdentifier(cloud) == identity, "replace must preserve object identity")

    guard let after = h.render() else { Issue.record("render 2 failed"); return }
    let counts = after.dominantCounts()
    #expect(counts.green > 0, "expected green after replace")
    #expect(counts.red == 0, "no red (old) pixels may survive an identity replace")
}

// (b) grow — larger replace grows source + LOD buffers; overflow 0; all selected.
@Test func replaceGrowsBuffersWithoutOverflow() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let h = ReplaceHarness(device: device)

    let (sp, sc) = solidGrid(pointsPerAxis: 8, color: green)      // 512 pts
    let small = PackedPointCloudFixtures.pack(positions: sp, colors: sc)
    let cloud = PointRasteriserPointCloud(context: h.context, packed: small)
    h.rasteriser.addPointCloud(cloud)
    _ = h.render()
    let smallCap = cloud.lodCapacity

    let (lp, lc) = solidGrid(pointsPerAxis: 48, color: red)       // 110_592 pts
    let large = PackedPointCloudFixtures.pack(positions: lp, colors: lc)
    cloud.replacePackedPointCloud(large)
    #expect(cloud.totalPoints == large.pointCount)
    #expect(cloud.lodCapacity >= large.pointCount, "lodCapacity must grow to cover new points")
    #expect(cloud.lodCapacity > smallCap, "expected LOD growth")

    guard let img = h.render() else { Issue.record("render failed"); return }
    #expect(cloud.lodOverflow == 0, "grow replace must never overflow LOD capacity")
    #expect(cloud.lodCount == cloud.totalPoints, "CLOD+frustum disabled → every point selected (\(cloud.lodCount) of \(cloud.totalPoints))")
    #expect(img.dominantCounts().red > 0 && img.dominantCounts().green == 0, "only the new (red) content renders")
}

// (c) shrink — smaller replace reuses buffers, no stale points from old content.
@Test func replaceShrinksWithoutStalePoints() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let h = ReplaceHarness(device: device)

    let (lp, lc) = solidGrid(pointsPerAxis: 40, color: red)       // 64_000 pts, full cube
    let large = PackedPointCloudFixtures.pack(positions: lp, colors: lc)
    let cloud = PointRasteriserPointCloud(context: h.context, packed: large)
    h.rasteriser.addPointCloud(cloud)
    guard let before = h.render() else { Issue.record("render 1 failed"); return }
    #expect(before.dominantCounts().red > 0)

    // A tiny green cluster near the center — a strict spatial + count subset.
    let (sp, sc) = solidGrid(pointsPerAxis: 6, color: green, span: 0.2)
    cloud.replacePackedPointCloud(PackedPointCloudFixtures.pack(positions: sp, colors: sc))
    #expect(cloud.totalPoints < large.pointCount)

    guard let after = h.render() else { Issue.record("render 2 failed"); return }
    let counts = after.dominantCounts()
    #expect(counts.green > 0, "small cloud must render")
    #expect(counts.red == 0, "no stale red points from the previous larger content")
    #expect(after.coveredPixelCount < before.coveredPixelCount, "shrunk content covers fewer pixels")
}

// (d) GPU replace bit-matches CPU replace for the same input.
@Test func gpuReplaceBitMatchesCPUReplace() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)
    let packer = try GPUPacker(device: device)

    // Deterministic randomized cloud, many batches (exercises shuffle + LOD).
    var state: UInt64 = 0xD00D
    func next() -> Float { state = state &* 6364136223846793005 &+ 1442695040888963407; return Float(state >> 40) / Float(1 << 24) }
    let count = 120_000
    var positions: [SIMD3<Float>] = []; var colors: [SIMD4<Float>] = []
    for _ in 0 ..< count {
        positions.append(SIMD3<Float>(next() * 8 - 4, next() * 8 - 4, next() * 8 - 4))
        colors.append(SIMD4<Float>(next(), next(), next(), 1))
    }
    let ppb = 10240

    let cpuCloud = PointRasteriserPointCloud(context: context, gpuPackPointCount: count, pointsPerBatch: ppb)
    cpuCloud.replacePackedPointCloud(PackedPointCloudFixtures.pack(positions: positions, colors: colors, pointsPerBatch: ppb))

    let gpuCloud = PointRasteriserPointCloud(context: context, gpuPackPointCount: count, pointsPerBatch: ppb)
    let posBuf = device.makeBuffer(bytes: positions, length: count * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
    let colBuf = device.makeBuffer(bytes: colors, length: count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!
    gpuCloud.replacePackedPointCloud(packer: packer, queue: context.commandQueue, positions: posBuf, colors: colBuf, count: count)

    #expect(gpuCloud.batchCount == cpuCloud.batchCount)
    #expect(gpuCloud.totalPoints == cpuCloud.totalPoints)
    func diff<T: Equatable>(_ a: MTLBuffer?, _ b: MTLBuffer?, _ n: Int, _ t: T.Type) -> Int {
        guard let a, let b else { return -1 }
        let pa = a.contents().bindMemory(to: T.self, capacity: n)
        let pb = b.contents().bindMemory(to: T.self, capacity: n)
        var d = 0; for i in 0 ..< n where pa[i] != pb[i] { d += 1 }; return d
    }
    let n = count
    #expect(diff(gpuCloud.xyzLowBuffer, cpuCloud.xyzLowBuffer, n, UInt32.self) == 0, "xyzLow differs")
    #expect(diff(gpuCloud.xyzMedBuffer, cpuCloud.xyzMedBuffer, n, UInt32.self) == 0, "xyzMed differs")
    #expect(diff(gpuCloud.xyzHighBuffer, cpuCloud.xyzHighBuffer, n, UInt32.self) == 0, "xyzHigh differs")
    #expect(diff(gpuCloud.colorsBuffer, cpuCloud.colorsBuffer, n, UInt32.self) == 0, "colors differ")
    #expect(diff(gpuCloud.levelsBuffer, cpuCloud.levelsBuffer, n, UInt8.self) == 0, "levels differ")
}

// (e) amortized mid-sweep replace → next completed sweep is the new content only.
@Test func replaceMidAmortizedSweepPublishesOnlyNewContent() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let h = ReplaceHarness(device: device) { cfg in
        cfg.lodPointsPerFrame = 2000 // small budget → amortized, multi-frame sweeps
    }

    let (lp, lc) = solidGrid(pointsPerAxis: 40, color: red) // 64_000 pts → many amortized frames
    let cloud = PointRasteriserPointCloud(context: h.context, packed: PackedPointCloudFixtures.pack(positions: lp, colors: lc))
    h.rasteriser.addPointCloud(cloud)
    // Frame 1 is a forced full sweep (red published); frame 2 begins a partial
    // amortized sweep (double-buffered), left deliberately mid-flight.
    _ = h.render()
    _ = h.render()

    // Replace mid-sweep → resets the sweep to a fresh full sweep of the new data.
    let (gp, gc) = solidGrid(pointsPerAxis: 16, color: green)
    cloud.replacePackedPointCloud(PackedPointCloudFixtures.pack(positions: gp, colors: gc))

    guard let after = h.render() else { Issue.record("render failed"); return }
    let counts = after.dominantCounts()
    #expect(counts.green > 0, "new content must be published")
    #expect(counts.red == 0, "a mid-sweep replace must never publish a stale or mixed (old red) front")
}

// GAP 1 — sub-region viewport composite: cloud confined to its half, other half untouched.
@Test func drawViewportCompositesIntoSubRegionOnly() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .bgra8Unorm, depthPixelFormat: .depth32Float)
    let rasteriser = PointRasteriser(context: context)
    rasteriser.setup()
    var cfg = PointRasteriserConfiguration()
    cfg.enableCLOD = false; cfg.enableFrustumCulling = false; cfg.holeFillIterations = 0; cfg.pointSizeScale = 6
    rasteriser.configuration = cfg

    let renderW = 256, renderH = 128
    rasteriser.resize(size: (Float(renderW), Float(renderH)), scaleFactor: 1)
    let (rp, rc) = solidGrid(pointsPerAxis: 20, color: red)
    let cloud = PointRasteriserPointCloud(context: context, packed: PackedPointCloudFixtures.pack(positions: rp, colors: rc))
    rasteriser.addPointCloud(cloud)

    // Composite target twice as wide as the rasteriser output; draw into the LEFT half.
    let targetW = 512, targetH = 128
    let camera = PerspectiveCamera(context: context, position: [0, 0, 2.4], near: 0.01, far: 100, fov: 45)
    camera.aspect = Float(renderW) / Float(renderH)
    camera.lookAt(target: .zero)
    rasteriser.update(renderContext: context, camera: camera, viewport: simd_float4(0, 0, Float(renderW), Float(renderH)), index: 0)

    let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: targetW, height: targetH, mipmapped: false)
    desc.usage = [.renderTarget, .shaderRead]
    desc.storageMode = .private
    guard let target = device.makeTexture(descriptor: desc),
          let cb = context.commandQueue.makeCommandBuffer() else { Issue.record("setup failed"); return }

    rasteriser.encode(cb)

    let rpd = MTLRenderPassDescriptor()
    rpd.colorAttachments[0].texture = target
    rpd.colorAttachments[0].loadAction = .clear
    rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    rpd.colorAttachments[0].storeAction = .store
    // Left half of the wider target.
    let leftViewport = MTLViewport(originX: 0, originY: 0, width: Double(targetW) / 2, height: Double(targetH), znear: 0, zfar: 1)
    rasteriser.draw(renderPassDescriptor: rpd, commandBuffer: cb, viewport: leftViewport)

    guard let staging = device.makeBuffer(length: targetW * targetH * 4, options: .storageModeShared),
          let blit = cb.makeBlitCommandEncoder() else { Issue.record("staging failed"); return }
    blit.copy(from: target, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: targetW, height: targetH, depth: 1),
              to: staging, destinationOffset: 0, destinationBytesPerRow: targetW * 4, destinationBytesPerImage: targetW * targetH * 4)
    blit.endEncoding()
    cb.commit(); cb.waitUntilCompleted()

    let px = staging.contents().bindMemory(to: UInt8.self, capacity: targetW * targetH * 4)
    var leftCovered = 0, rightCovered = 0
    for y in 0 ..< targetH {
        for x in 0 ..< targetW {
            let a = px[(y * targetW + x) * 4 + 3]
            if a > 0 { if x < targetW / 2 { leftCovered += 1 } else { rightCovered += 1 } }
        }
    }
    #expect(leftCovered > 0, "cloud must composite into the left viewport half")
    #expect(rightCovered == 0, "the right half must be untouched (\(rightCovered) covered)")
}
#endif
