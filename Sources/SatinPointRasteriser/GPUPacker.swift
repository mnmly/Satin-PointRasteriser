#if canImport(Metal)
import Foundation
import Metal
import simd

/// GPU implementation of ``PackedPointCloudFixtures/pack(positions:colors:pointsPerBatch:lodLevels:coarseVoxelDivisions:shuffleBatches:)``.
///
/// Packs GPU-resident `positions`/`colors` `MTLBuffer`s straight into a
/// wholesale ``PointRasteriserPointCloud``'s source buffers — no CPU pack, no
/// Swift arrays. Replaces the ~2-minute CPU pack for large PLYs with a handful
/// of compute dispatches.
///
/// The output is **decode-identical** to the CPU ``PackedPointCloudFixtures/pack(positions:colors:pointsPerBatch:lodLevels:coarseVoxelDivisions:shuffleBatches:)``:
/// same Morton order (the CPU reference tie-breaks equal keys by original index,
/// matching this stable radix sort), same per-batch AABB + 30-bit quantization,
/// same density LOD levels, same stable level-ascending bucketing with the
/// cumulative counts in ``RasterBatch/lodCumulativeCounts``, **and** the same
/// whole-batch shuffle (folded into the finalize stage, no extra pass).
///
/// Stages (compute kernels): global bounds → Morton key → LSD radix sort →
/// gather positions → LOD voxel occupancy → per-batch AABB → per-batch stable
/// level bucketing → shuffled finalize (gather + quantize + batch record).
///
/// - Note: Apple-platform only. The CPU `pack()` remains the cross-platform
///   reference/fallback.
public final class GPUPacker {
    /// Per-tile element count for the radix sort. One thread processes a whole
    /// tile (histogram + stable scatter in identical order).
    public static let radixTileSize = 1024

    /// Maximum dense LOD-grid cell count before GPU LOD is skipped (levels left
    /// at the finest). 2^28 cells ≈ 1 GB of uint cells.
    public static let lodCellCap = 268_435_456

    private struct PackParams {
        var count: UInt32 = 0
        var pointsPerBatch: UInt32 = 0
        var numBatches: UInt32 = 0
        var numTiles: UInt32 = 0
        var tileSize: UInt32 = 0
        var shift: UInt32 = 0
        var maxLevel: UInt32 = 0
        var level: UInt32 = 0
        var coarseVoxelDivisions: Float = 0
        var lodVoxelScale: Float = 0
        var slotBase: UInt32 = 0
    }

    private let device: MTLDevice
    private let lodLevels: Int
    private let coarseVoxelDivisions: Int
    private let lodGridCellCount: Int

    private let pBoundsInit, pBounds, pMorton: MTLComputePipelineState
    private let pHistogram, pScanPerDigit, pDigitBase, pScatter: MTLComputePipelineState
    private let pGatherPos, pBatchAABB, pBucketLOD, pFinalize: MTLComputePipelineState
    private let pLODInit, pLODClaim, pLODAssign: MTLComputePipelineState

    private let scratch = PackScratch()

    /// - Parameters:
    ///   - device: the Metal device.
    ///   - lodLevels: number of LOD levels (clamped to `1...8`); matches
    ///     ``PackedPointCloudFixtures/defaultLODLevels``.
    ///   - coarseVoxelDivisions: coarsest-level voxel divisions along the longest
    ///     axis; matches ``PackedPointCloudFixtures/defaultCoarseVoxelDivisions``.
    public init(
        device: MTLDevice,
        lodLevels: Int = PackedPointCloudFixtures.defaultLODLevels,
        coarseVoxelDivisions: Int = PackedPointCloudFixtures.defaultCoarseVoxelDivisions
    ) throws {
        self.device = device
        let clampedLevels = max(1, min(lodLevels, 8))
        let clampedDivs = max(1, coarseVoxelDivisions)
        self.lodLevels = clampedLevels
        self.coarseVoxelDivisions = clampedDivs

        // Per-axis cell count ≤ coarseVoxelDivisions·2^(lodLevels-2) (voxel size
        // scales with the longest axis). +2 guards the floor()+1 dim formula.
        if clampedLevels > 1 {
            let maxDim = clampedDivs * (1 << (clampedLevels - 2)) + 2
            let cells = maxDim * maxDim * maxDim
            self.lodGridCellCount = cells <= Self.lodCellCap ? cells : 0
            if cells > Self.lodCellCap {
                print("[GPUPacker] LOD grid \(cells) cells exceeds cap \(Self.lodCellCap); GPU LOD disabled (levels left at finest).")
            }
        } else {
            self.lodGridCellCount = 0
        }

        let url = PointRasteriser.pipelinesURL
            .appendingPathComponent("GPUPacker")
            .appendingPathComponent("Shaders.metal")
        let source = try String(contentsOf: url, encoding: .utf8)
        // Disable fast math: the pack must be bit-identical to the CPU reference,
        // which uses IEEE-correctly-rounded division. Fast math makes GPU
        // division/reciprocal approximate (±1 ULP), which flips the low bit of
        // the 30-bit quantization (and, at voxel boundaries, a few LOD cells).
        let options = MTLCompileOptions()
        if #available(macOS 15.0, iOS 18.0, *) {
            options.mathMode = .safe
        } else {
            options.fastMathEnabled = false
        }
        let library = try device.makeLibrary(source: source, options: options)

        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw GPUPackerError.kernelNotFound(name)
            }
            return try device.makeComputePipelineState(function: fn)
        }
        pBoundsInit = try pipeline("packBoundsInit")
        pBounds = try pipeline("packBounds")
        pMorton = try pipeline("packMorton")
        pHistogram = try pipeline("radixHistogram")
        pScanPerDigit = try pipeline("radixScanPerDigit")
        pDigitBase = try pipeline("radixDigitBase")
        pScatter = try pipeline("radixScatter")
        pGatherPos = try pipeline("packGatherPos")
        pBatchAABB = try pipeline("packBatchAABB")
        pBucketLOD = try pipeline("packBucketLOD")
        pFinalize = try pipeline("packShuffleFinalize")
        pLODInit = try pipeline("packLODInit")
        pLODClaim = try pipeline("packLODClaim")
        pLODAssign = try pipeline("packLODAssign")
    }

    /// Number of batches `count` points pack into at `pointsPerBatch`.
    public static func batchCount(count: Int, pointsPerBatch: Int) -> Int {
        max(1, (count + pointsPerBatch - 1) / pointsPerBatch)
    }

    /// Encode all pack stages onto `commandBuffer`, writing the quantized layout
    /// into `cloud`'s wholesale source buffers. The caller commits (and typically
    /// waits, then calls ``PointRasteriserPointCloud/adoptGPUBatchBounds()``).
    ///
    /// - Parameters:
    ///   - positions: `float3` positions, **16-byte stride** (`SIMD3<Float>`).
    ///   - colors: `float4` RGBA in `[0,1]`, 16-byte stride.
    ///   - count: number of points.
    ///   - shuffle: apply the whole-batch shuffle (default `true`, matching the
    ///     CPU path); pass `false` to keep strict Morton batch order.
    ///   - cloud: destination wholesale cloud (sized for ≥ `count` points).
    public func pack(
        positions: MTLBuffer,
        colors: MTLBuffer,
        count: Int,
        shuffle: Bool = true,
        into cloud: PointRasteriserPointCloud,
        commandBuffer: MTLCommandBuffer
    ) {
        guard count > 0 else { return }
        let ppb = cloud.pointsPerBatch
        precondition(ppb <= 65535, "pointsPerBatch must fit the uint16 LOD prefix counts")
        precondition(cloud.totalPoints >= count, "cloud too small for \(count) points")
        let numBatches = Self.batchCount(count: count, pointsPerBatch: ppb)
        precondition(cloud.batchCount >= numBatches, "cloud has too few batch slots")

        guard let xyzLow = cloud.xyzLowBuffer, let xyzMed = cloud.xyzMedBuffer,
              let xyzHigh = cloud.xyzHighBuffer, let colorsOut = cloud.colorsBuffer,
              let levels = cloud.levelsBuffer, let batches = cloud.batchesBuffer
        else { return }

        let numTiles = (count + Self.radixTileSize - 1) / Self.radixTileSize
        scratch.ensure(device: device, count: count, numBatches: numBatches,
                       radixTileSize: Self.radixTileSize, lodGridCellCount: lodGridCellCount)
        guard let keysA = scratch.keysA, let keysB = scratch.keysB,
              let indicesA = scratch.indicesA, let indicesB = scratch.indicesB,
              let sortedPos = scratch.sortedPos, let boundsBuf = scratch.boundsBuf,
              let histBuf = scratch.histBuf, let tileOffsetBuf = scratch.tileOffsetBuf,
              let digitTotalBuf = scratch.digitTotalBuf, let digitBaseBuf = scratch.digitBaseBuf,
              let oldBatches = scratch.oldBatches, let shuffleMap = scratch.shuffleMap
        else { return }

        // Whole-batch shuffle: computed on CPU (deterministic, batch-count only)
        // and folded into the finalize stage. shuffleMap[j] = (oldBatch, newFirst).
        writeShuffleMap(into: shuffleMap, numBatches: numBatches, pointsPerBatch: ppb, count: count, shuffle: shuffle)

        var params = PackParams(
            count: UInt32(count),
            pointsPerBatch: UInt32(ppb),
            numBatches: UInt32(numBatches),
            numTiles: UInt32(numTiles),
            tileSize: UInt32(Self.radixTileSize),
            shift: 0,
            maxLevel: UInt32(lodLevels - 1),
            level: 0,
            coarseVoxelDivisions: Float(coarseVoxelDivisions),
            lodVoxelScale: 1.0,
            slotBase: 0
        )

        // 1. Global bounds.
        encode(commandBuffer, pBoundsInit, threads: 6) { e in
            e.setBuffer(boundsBuf, offset: 0, index: 0)
        }
        encodeGroups(commandBuffer, pBounds, groups: min(numTiles, 2048), threadsPerGroup: 256) { e in
            e.setBuffer(positions, offset: 0, index: 0)
            e.setBuffer(boundsBuf, offset: 0, index: 1)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 2)
        }

        // 2. Morton key + identity indices.
        encode(commandBuffer, pMorton, threads: count) { e in
            e.setBuffer(positions, offset: 0, index: 0)
            e.setBuffer(keysA, offset: 0, index: 1)
            e.setBuffer(indicesA, offset: 0, index: 2)
            e.setBuffer(boundsBuf, offset: 0, index: 3)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 4)
        }

        // 3. LSD radix sort — 4×8-bit passes, ping-ponging A→B→A→B→A.
        var (kIn, kOut, iIn, iOut) = (keysA, keysB, indicesA, indicesB)
        for pass in 0 ..< 4 {
            params.shift = UInt32(pass * 8)
            encode(commandBuffer, pHistogram, threads: numTiles) { e in
                e.setBuffer(kIn, offset: 0, index: 0)
                e.setBuffer(histBuf, offset: 0, index: 1)
                e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 2)
            }
            encodeGroups(commandBuffer, pScanPerDigit, groups: 256, threadsPerGroup: 256) { e in
                e.setBuffer(histBuf, offset: 0, index: 0)
                e.setBuffer(tileOffsetBuf, offset: 0, index: 1)
                e.setBuffer(digitTotalBuf, offset: 0, index: 2)
                e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 3)
            }
            encode(commandBuffer, pDigitBase, threads: 1) { e in
                e.setBuffer(digitTotalBuf, offset: 0, index: 0)
                e.setBuffer(digitBaseBuf, offset: 0, index: 1)
            }
            encode(commandBuffer, pScatter, threads: numTiles) { e in
                e.setBuffer(kIn, offset: 0, index: 0)
                e.setBuffer(iIn, offset: 0, index: 1)
                e.setBuffer(kOut, offset: 0, index: 2)
                e.setBuffer(iOut, offset: 0, index: 3)
                e.setBuffer(tileOffsetBuf, offset: 0, index: 4)
                e.setBuffer(digitBaseBuf, offset: 0, index: 5)
                e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 6)
            }
            swap(&kIn, &kOut)
            swap(&iIn, &iOut)
        }
        // After 4 swaps the sorted indices are back in indicesA (== iIn); keysA
        // holds dead sorted keys, keysB/indicesB dead. Reuse them:
        //   keysB     → Morton-order LOD levels (uchar/point)
        //   indicesB  → bucketed source-index permutation
        //   keysA     → bucketed (final-order) levels (uchar/point)
        let sortedIndices = iIn
        let mortonLevels = keysB
        let bucketedIndices = indicesB
        let bucketedLevels = keysA

        // 4. Gather positions into Morton order (for LOD + per-batch AABB).
        encode(commandBuffer, pGatherPos, threads: count) { e in
            e.setBuffer(positions, offset: 0, index: 0)
            e.setBuffer(sortedIndices, offset: 0, index: 1)
            e.setBuffer(sortedPos, offset: 0, index: 2)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 3)
        }

        // 5. LOD voxel occupancy → Morton-order levels. Runs on the PRE-bucket
        //    order: atomic_min "lowest sorted index per voxel wins" mirrors the
        //    CPU occupancy walk over sorted order.
        encode(commandBuffer, pLODInit, threads: count) { e in
            e.setBuffer(mortonLevels, offset: 0, index: 0)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 1)
        }
        if lodLevels > 1, lodGridCellCount > 0, let lodGrid = scratch.lodGrid {
            for level in 0 ..< (lodLevels - 1) {
                params.level = UInt32(level)
                params.lodVoxelScale = powf(0.5, Float(level))
                if let blit = commandBuffer.makeBlitCommandEncoder() {
                    blit.label = "GPUPacker.LODGridFill"
                    blit.fill(buffer: lodGrid, range: 0 ..< (lodGridCellCount * 4), value: 0xff)
                    blit.endEncoding()
                }
                encode(commandBuffer, pLODClaim, threads: count) { e in
                    e.setBuffer(sortedPos, offset: 0, index: 0)
                    e.setBuffer(mortonLevels, offset: 0, index: 1)
                    e.setBuffer(lodGrid, offset: 0, index: 2)
                    e.setBuffer(boundsBuf, offset: 0, index: 3)
                    e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 4)
                }
                encode(commandBuffer, pLODAssign, threads: count) { e in
                    e.setBuffer(sortedPos, offset: 0, index: 0)
                    e.setBuffer(mortonLevels, offset: 0, index: 1)
                    e.setBuffer(lodGrid, offset: 0, index: 2)
                    e.setBuffer(boundsBuf, offset: 0, index: 3)
                    e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 4)
                }
            }
        }

        // 6. Per-batch AABB → scratch batch records (Morton/old order).
        encodeGroups(commandBuffer, pBatchAABB, groups: numBatches, threadsPerGroup: 256) { e in
            e.setBuffer(sortedPos, offset: 0, index: 0)
            e.setBuffer(oldBatches, offset: 0, index: 1)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 2)
        }

        // 7. Stable per-batch level bucketing → bucketed indices/levels + cum counts.
        encode(commandBuffer, pBucketLOD, threads: numBatches) { e in
            e.setBuffer(mortonLevels, offset: 0, index: 0)
            e.setBuffer(sortedIndices, offset: 0, index: 1)
            e.setBuffer(bucketedIndices, offset: 0, index: 2)
            e.setBuffer(bucketedLevels, offset: 0, index: 3)
            e.setBuffer(oldBatches, offset: 0, index: 4)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 5)
        }

        // 8. Shuffled finalize: gather + quantize + emit final batch records at
        //    the shuffled destinations, in one pass.
        encodeGroups(commandBuffer, pFinalize, groups: numBatches, threadsPerGroup: 256) { e in
            e.setBuffer(positions, offset: 0, index: 0)
            e.setBuffer(colors, offset: 0, index: 1)
            e.setBuffer(bucketedIndices, offset: 0, index: 2)
            e.setBuffer(bucketedLevels, offset: 0, index: 3)
            e.setBuffer(oldBatches, offset: 0, index: 4)
            e.setBuffer(shuffleMap, offset: 0, index: 5)
            e.setBuffer(xyzLow, offset: 0, index: 6)
            e.setBuffer(xyzMed, offset: 0, index: 7)
            e.setBuffer(xyzHigh, offset: 0, index: 8)
            e.setBuffer(colorsOut, offset: 0, index: 9)
            e.setBuffer(levels, offset: 0, index: 10)
            e.setBuffer(batches, offset: 0, index: 11)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 12)
        }
    }

    /// Read the global bounds computed by the most recent ``pack(positions:colors:count:shuffle:into:commandBuffer:)``
    /// (valid after the command buffer completes). Returns `nil` if no pack ran.
    public func lastBounds() -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        guard let boundsBuf = scratch.boundsBuf else { return nil }
        let b = boundsBuf.contents().bindMemory(to: UInt32.self, capacity: 6)
        func unflip(_ u: UInt32) -> Float {
            let mask = ((u >> 31) &- 1) | 0x80000000
            return Float(bitPattern: u ^ mask)
        }
        return (SIMD3<Float>(unflip(b[0]), unflip(b[1]), unflip(b[2])),
                SIMD3<Float>(unflip(b[3]), unflip(b[4]), unflip(b[5])))
    }

    // MARK: - Shuffle map

    /// Mirror of ``shuffleBatches(_:)``'s permutation, as `(oldBatch, newFirst)`
    /// per final batch. `newFirst[j] = Σ numPoints(newOrder[k]) for k < j`.
    private func writeShuffleMap(into buffer: MTLBuffer, numBatches: Int, pointsPerBatch ppb: Int, count: Int, shuffle: Bool) {
        var newOrder = Array(0 ..< numBatches)
        if shuffle {
            var i = 1
            while i < numBatches - 1 - i {
                newOrder.swapAt(i, numBatches - 1 - i)
                i += 2
            }
        }
        func numPoints(_ b: Int) -> Int { min(ppb, count - b * ppb) }
        let map = buffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: numBatches)
        var cursor = 0
        for j in 0 ..< numBatches {
            let oldB = newOrder[j]
            map[j] = SIMD2<UInt32>(UInt32(oldB), UInt32(cursor))
            cursor += numPoints(oldB)
        }
    }

    // MARK: - Encode helpers

    private func encode(
        _ commandBuffer: MTLCommandBuffer,
        _ pipeline: MTLComputePipelineState,
        threads: Int,
        _ bind: (MTLComputeCommandEncoder) -> Void
    ) {
        guard threads > 0, let e = commandBuffer.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(pipeline)
        bind(e)
        let tew = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tew, height: 1, depth: 1))
        e.endEncoding()
    }

    private func encodeGroups(
        _ commandBuffer: MTLCommandBuffer,
        _ pipeline: MTLComputePipelineState,
        groups: Int,
        threadsPerGroup: Int,
        _ bind: (MTLComputeCommandEncoder) -> Void
    ) {
        guard groups > 0, let e = commandBuffer.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(pipeline)
        bind(e)
        let tpg = min(pipeline.maxTotalThreadsPerThreadgroup, threadsPerGroup)
        e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        e.endEncoding()
    }
}

/// GPU-private scratch for a single wholesale pack: radix keys/indices, sorted
/// positions, histogram/scan temporaries, the dense LOD voxel grid, the
/// old-order batch records, and the CPU-written shuffle map. Grows to the
/// largest pack it has seen.
private final class PackScratch {
    var capacityPoints = 0
    var capacityBatches = 0
    var keysA, keysB, indicesA, indicesB: MTLBuffer?
    var sortedPos, boundsBuf: MTLBuffer?
    var histBuf, tileOffsetBuf, digitTotalBuf, digitBaseBuf: MTLBuffer?
    var lodGrid: MTLBuffer?
    var oldBatches, shuffleMap: MTLBuffer?

    func ensure(device: MTLDevice, count: Int, numBatches: Int, radixTileSize: Int, lodGridCellCount: Int) {
        if count > capacityPoints {
            let uintLen = count * MemoryLayout<UInt32>.stride
            let numTiles = (count + radixTileSize - 1) / radixTileSize
            keysA = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            keysB = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            indicesA = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            indicesB = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            sortedPos = device.makeBuffer(length: count * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)
            histBuf = device.makeBuffer(length: numTiles * 256 * 4, options: .storageModePrivate)
            tileOffsetBuf = device.makeBuffer(length: numTiles * 256 * 4, options: .storageModePrivate)
            capacityPoints = count
        }
        if numBatches > capacityBatches {
            oldBatches = device.makeBuffer(length: numBatches * MemoryLayout<RasterBatch>.stride, options: .storageModePrivate)
            shuffleMap = device.makeBuffer(length: numBatches * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)
            capacityBatches = numBatches
        }
        if boundsBuf == nil {
            boundsBuf = device.makeBuffer(length: 6 * 4, options: .storageModeShared)
            digitTotalBuf = device.makeBuffer(length: 256 * 4, options: .storageModePrivate)
            digitBaseBuf = device.makeBuffer(length: 256 * 4, options: .storageModePrivate)
        }
        if lodGridCellCount > 0, lodGrid == nil {
            lodGrid = device.makeBuffer(length: lodGridCellCount * 4, options: .storageModePrivate)
        }
    }
}

public enum GPUPackerError: LocalizedError {
    case kernelNotFound(String)
    public var errorDescription: String? {
        switch self {
        case let .kernelNotFound(name): return "GPUPacker kernel '\(name)' not found."
        }
    }
}
#endif
