#if canImport(Metal)
import Foundation
import Metal
import Satin
import simd
import Testing
@testable import SatinPointRasteriser

/// GPU packer ⇄ CPU packer equivalence.
///
/// The GPU packer is a faithful port of ``PackedPointCloudFixtures/pack(positions:colors:pointsPerBatch:lodLevels:coarseVoxelDivisions:shuffleBatches:)``.
/// Because the CPU reference now tie-breaks equal Morton keys by original index
/// (matching the GPU's stable radix sort), the two are expected to be
/// **bit-identical**: same batch count, same per-batch AABB / numPoints /
/// firstPoint / cumulative LOD counts, and same per-point `xyzLow`/`xyzMed`/
/// `xyzHigh` / colors / levels — including after the whole-batch shuffle.
private struct GPUPackerEquivalence {
    let lodLevels = 4
    let coarseVoxelDivisions = 64

    /// Deterministic pseudo-random cloud (LCG).
    static func jitteredCloud(count: Int, seed: UInt64, span: Float = 10) -> ([SIMD3<Float>], [SIMD4<Float>]) {
        var state = seed
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24)
        }
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD4<Float>] = []
        positions.reserveCapacity(count)
        colors.reserveCapacity(count)
        for _ in 0 ..< count {
            positions.append(SIMD3<Float>(next() * span - span / 2, next() * span - span / 2, next() * span - span / 2))
            colors.append(SIMD4<Float>(next(), next(), next(), 1))
        }
        return (positions, colors)
    }

    static func cubeGrid(pointsPerAxis n: Int) -> ([SIMD3<Float>], [SIMD4<Float>]) {
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD4<Float>] = []
        for z in 0 ..< n {
            for y in 0 ..< n {
                for x in 0 ..< n {
                    let fx = Float(x) / Float(n - 1)
                    let fy = Float(y) / Float(n - 1)
                    let fz = Float(z) / Float(n - 1)
                    positions.append(SIMD3<Float>(fx - 0.5, fy - 0.5, fz - 0.5))
                    colors.append(SIMD4<Float>(fx, fy, fz, 1.0))
                }
            }
        }
        return (positions, colors)
    }

    func makeContext() -> Context? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        return Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)
    }

    /// Assert GPU pack == CPU pack for the given input, returning a short label on
    /// success (for the test to report), or triggering `#expect` failures.
    func assertEquivalent(
        _ positions: [SIMD3<Float>],
        _ colors: [SIMD4<Float>],
        pointsPerBatch: Int,
        shuffle: Bool,
        context: Context,
        packer: GPUPacker
    ) {
        let cpu = PackedPointCloudFixtures.pack(
            positions: positions, colors: colors,
            pointsPerBatch: pointsPerBatch, lodLevels: lodLevels,
            coarseVoxelDivisions: coarseVoxelDivisions, shuffleBatches: shuffle
        )
        let cloud = PointRasteriserPointCloud.gpuPacked(
            context: context, packer: packer, queue: context.commandQueue,
            positions: positions, colors: colors,
            pointsPerBatch: pointsPerBatch, shuffle: shuffle
        )

        #expect(cloud.contentGeneration > 0, "GPU pack must bump contentGeneration")
        #expect(cloud.pointCount == cpu.pointCount)
        #expect(cloud.batchCount == cpu.batchCount)
        #expect(cloud.sourceBoundsMin == cpu.boundsMin)
        #expect(cloud.sourceBoundsMax == cpu.boundsMax)
        guard cloud.batchCount == cpu.batchCount, cloud.pointCount == cpu.pointCount else { return }

        let n = cpu.pointCount
        // Per-batch records.
        let gpuBatches = cloud.batchesBuffer!.contents().bindMemory(to: RasterBatch.self, capacity: cloud.batchCount)
        var batchMismatch = 0
        for b in 0 ..< cpu.batchCount {
            let cb = cpu.batches[b]
            let gb = gpuBatches[b]
            if cb.numPoints != gb.numPoints || cb.firstPoint != gb.firstPoint
                || cb.minX != gb.minX || cb.minY != gb.minY || cb.minZ != gb.minZ
                || cb.maxX != gb.maxX || cb.maxY != gb.maxY || cb.maxZ != gb.maxZ
                || cb.lodCumulativeCounts != gb.lodCumulativeCounts {
                batchMismatch += 1
            }
        }
        #expect(batchMismatch == 0, "\(batchMismatch)/\(cpu.batchCount) batch records differ")

        // Per-point buffers, bit-for-bit.
        func mismatches<T: Equatable>(_ buffer: MTLBuffer, _ reference: [T], as type: T.Type) -> Int {
            let ptr = buffer.contents().bindMemory(to: T.self, capacity: n)
            var diff = 0
            for i in 0 ..< n where ptr[i] != reference[i] { diff += 1 }
            return diff
        }
        #expect(mismatches(cloud.xyzLowBuffer!, cpu.xyzLow, as: UInt32.self) == 0, "xyzLow differs")
        #expect(mismatches(cloud.xyzMedBuffer!, cpu.xyzMed, as: UInt32.self) == 0, "xyzMed differs")
        #expect(mismatches(cloud.xyzHighBuffer!, cpu.xyzHigh, as: UInt32.self) == 0, "xyzHigh differs")
        #expect(mismatches(cloud.colorsBuffer!, cpu.colors, as: UInt32.self) == 0, "colors differ")
        #expect(mismatches(cloud.levelsBuffer!, cpu.levels, as: UInt8.self) == 0, "levels differ")
    }
}

@Test func gpuPackMatchesCPUBitIdenticalOnFixtures() throws {
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let harness = GPUPackerEquivalence()
    guard let context = harness.makeContext() else { return } // no GPU (CI) — skip
    let packer = try GPUPacker(device: context.device, lodLevels: harness.lodLevels, coarseVoxelDivisions: harness.coarseVoxelDivisions)

    // cubeGrid (unique Morton keys) — single- and multi-batch, shuffled + not.
    let (gp8, gc8) = GPUPackerEquivalence.cubeGrid(pointsPerAxis: 8)
    harness.assertEquivalent(gp8, gc8, pointsPerBatch: 256, shuffle: false, context: context, packer: packer)
    harness.assertEquivalent(gp8, gc8, pointsPerBatch: 64, shuffle: true, context: context, packer: packer)

    let (gp32, gc32) = GPUPackerEquivalence.cubeGrid(pointsPerAxis: 32) // 32768 pts
    harness.assertEquivalent(gp32, gc32, pointsPerBatch: 1024, shuffle: true, context: context, packer: packer)

    // Randomized clouds — several sizes, both shuffle modes. pointsPerBatch=256
    // yields many batches so the shuffle permutation is exercised.
    for (count, seed) in [(4000, 0x5EED), (60000, 0xBEEF), (250_000, 0xC0DE)] {
        let (p, c) = GPUPackerEquivalence.jitteredCloud(count: count, seed: UInt64(seed))
        harness.assertEquivalent(p, c, pointsPerBatch: 256, shuffle: true, context: context, packer: packer)
        harness.assertEquivalent(p, c, pointsPerBatch: 10240, shuffle: false, context: context, packer: packer)
    }
}

@Test func gpuPackMatchesCPUOnRoughlyOneMillionPoints() throws {
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let harness = GPUPackerEquivalence()
    guard let context = harness.makeContext() else { return }
    let packer = try GPUPacker(device: context.device, lodLevels: harness.lodLevels, coarseVoxelDivisions: harness.coarseVoxelDivisions)
    let (p, c) = GPUPackerEquivalence.jitteredCloud(count: 1_000_003, seed: 0xA11CE)
    harness.assertEquivalent(p, c, pointsPerBatch: 10240, shuffle: true, context: context, packer: packer)
}

/// Env-gated perf smoke test: `PR_GPU_PACK_BENCH=1 swift test --filter gpuPackPerfSmoke`.
/// Reports GPU pack time for ~10M synthetic points. Never asserts (informational).
@Test func gpuPackPerfSmoke() throws {
    guard ProcessInfo.processInfo.environment["PR_GPU_PACK_BENCH"] != nil else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let harness = GPUPackerEquivalence()
    guard let context = harness.makeContext() else { return }
    let packer = try GPUPacker(device: context.device)

    let count = 10_000_000
    let (positions, colors) = GPUPackerEquivalence.jitteredCloud(count: count, seed: 0xF00D)
    // Upload once so the timed section is pure pack (matches the loader flow,
    // which uploads then packs).
    let posBuf = context.device.makeBuffer(bytes: positions, length: count * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
    let colBuf = context.device.makeBuffer(bytes: colors, length: count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!
    let cloud = PointRasteriserPointCloud(context: context, gpuPackPointCount: count)

    // Warm up (compiles pipelines already done; this warms buffers/caches).
    do {
        let cb = context.commandQueue.makeCommandBuffer()!
        packer.pack(positions: posBuf, colors: colBuf, count: count, into: cloud, commandBuffer: cb)
        cb.commit(); cb.waitUntilCompleted()
    }
    var gpuMs: [Double] = []
    var wallMs: [Double] = []
    for _ in 0 ..< 5 {
        let cb = context.commandQueue.makeCommandBuffer()!
        let wallStart = Date()
        packer.pack(positions: posBuf, colors: colBuf, count: count, into: cloud, commandBuffer: cb)
        cb.commit(); cb.waitUntilCompleted()
        wallMs.append(Date().timeIntervalSince(wallStart) * 1000)
        gpuMs.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
    }
    let minGPU = gpuMs.min() ?? 0
    let minWall = wallMs.min() ?? 0
    print("[gpuPackPerfSmoke] \(count) pts, \(cloud.batchCount) batches: GPU \(String(format: "%.2f", minGPU))ms, wall \(String(format: "%.2f", minWall))ms (best of 5)")
}

/// End-to-end timing for a real PLY through the GPU fast path. Point it at a
/// file with `PR_SANMARCO_PLY=/path/to.ply swift test -c release --filter gpuPackRealPLYEndToEnd`.
/// Reports parse vs upload+GPU-pack vs total. Never asserts (informational).
@Test func gpuPackRealPLYEndToEnd() throws {
    guard let path = ProcessInfo.processInfo.environment["PR_SANMARCO_PLY"] else { return }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let harness = GPUPackerEquivalence()
    guard let context = harness.makeContext() else { return }

    let url = URL(fileURLWithPath: path)
    let parseStart = Date()
    let (positions, colors) = try PLYPointCloudLoader.loadArrays(url: url)
    let parseSeconds = Date().timeIntervalSince(parseStart)

    let packStart = Date()
    let packer = try GPUPacker(device: context.device)
    let cloud = PointRasteriserPointCloud.gpuPacked(
        context: context, packer: packer, queue: context.commandQueue,
        positions: positions, colors: colors
    )
    let packSeconds = Date().timeIntervalSince(packStart)

    print("""
    [gpuPackRealPLYEndToEnd] \(url.lastPathComponent)
      points:  \(cloud.pointCount)  batches: \(cloud.batchCount)
      parse:   \(String(format: "%.2f", parseSeconds))s
      pack:    \(String(format: "%.3f", packSeconds))s (includes upload + GPU pack + waitUntilCompleted, first call also compiles kernels)
      total:   \(String(format: "%.2f", parseSeconds + packSeconds))s
      bounds:  \(cloud.sourceBoundsMin) .. \(cloud.sourceBoundsMax)
    """)
}
#endif
