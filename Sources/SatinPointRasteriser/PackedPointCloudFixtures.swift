import Foundation
import simd

/// CPU reference packer: bounds → Morton sort → per-point LOD level → per-batch
/// level bucketing → 30-bit quantize → batch shuffle. Cross-implementation
/// contract shared with GPU packers and streaming loaders — the Morton,
/// quantize, and LOD formulas here are pinned and must not drift.
public enum PackedPointCloudFixtures {
    /// A `pointsPerAxis`³ grid of points spanning `[-0.5, 0.5]` per axis, colored by
    /// normalized position. Handy fixture for tests and example apps.
    public static func cubeGrid(pointsPerAxis: Int = 24) -> PackedPointCloud {
        let count = max(pointsPerAxis, 2)
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD4<Float>] = []
        positions.reserveCapacity(count * count * count)
        colors.reserveCapacity(count * count * count)

        for z in 0 ..< count {
            for y in 0 ..< count {
                for x in 0 ..< count {
                    let fx = Float(x) / Float(count - 1)
                    let fy = Float(y) / Float(count - 1)
                    let fz = Float(z) / Float(count - 1)
                    positions.append(SIMD3<Float>(fx - 0.5, fy - 0.5, fz - 0.5))
                    colors.append(SIMD4<Float>(fx, fy, fz, 1.0))
                }
            }
        }

        return pack(positions: positions, colors: colors)
    }

    public static let defaultLODLevels: Int = 4
    public static let defaultCoarseVoxelDivisions: Int = 64

    /// Packs `positions`/`colors` into a ``PackedPointCloud``: computes bounds,
    /// Morton-sorts, assigns LOD levels, buckets each batch slice level-ascending,
    /// quantizes each point to 30 bits relative to its batch's AABB, and — unless
    /// disabled — shuffles the resulting batch order (see ``shuffleBatches(_:)``).
    ///
    /// - Parameter shuffleBatches: When `true` (the default), swaps every-other
    ///   batch with its mirror from the end after packing, per the Magnopus
    ///   inter-wave-contention mitigation. Pass `false` to preserve strict
    ///   Morton batch order (e.g. for tests that assert on it).
    public static func pack(
        positions: [SIMD3<Float>],
        colors: [SIMD4<Float>],
        pointsPerBatch: Int = pointRasteriserThreadsPerGroup * 80,
        lodLevels: Int = defaultLODLevels,
        coarseVoxelDivisions: Int = defaultCoarseVoxelDivisions,
        shuffleBatches shouldShuffleBatches: Bool = true
    ) -> PackedPointCloud {
        precondition(positions.count == colors.count, "positions and colors must have the same count")
        guard !positions.isEmpty else {
            return PackedPointCloud(
                batches: [],
                files: [RasterFile()],
                xyzLow: [],
                xyzMed: [],
                xyzHigh: [],
                colors: [],
                levels: [],
                boundsMin: .zero,
                boundsMax: .zero
            )
        }

        var boundsMin = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var boundsMax = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        for position in positions {
            boundsMin = simd_min(boundsMin, position)
            boundsMax = simd_max(boundsMax, position)
        }

        var order = mortonOrder(positions: positions, boundsMin: boundsMin, boundsMax: boundsMax)
        var sortedPositions = order.map { positions[$0] }
        var levels = computeLODLevels(
            positions: sortedPositions,
            boundsMin: boundsMin,
            boundsMax: boundsMax,
            lodLevels: max(1, min(lodLevels, 8)),
            coarseVoxelDivisions: max(1, coarseVoxelDivisions)
        )

        // Bucket each batch slice level-ascending (stable, so Morton order is
        // preserved within a level) and compose the permutation into `order`,
        // so colors — gathered below — plus `sourceIndices`/`orderedPositions`
        // all follow the same final point order. The cull pass uses the
        // per-batch cumulative counts to bound the draw loops.
        let batchStride = max(pointsPerBatch, 1)
        precondition(batchStride <= 65535, "pointsPerBatch must fit the uint16 LOD prefix counts")
        bucketSortBatchSlicesByLevel(
            order: &order,
            positions: &sortedPositions,
            levels: &levels,
            batchStride: batchStride
        )

        let sortedColorsSrc = order.map { colors[$0] }

        var batches: [RasterBatch] = []
        var xyzLow = Array(repeating: UInt32(0), count: sortedPositions.count)
        var xyzMed = Array(repeating: UInt32(0), count: sortedPositions.count)
        var xyzHigh = Array(repeating: UInt32(0), count: sortedPositions.count)

        var first = 0
        while first < sortedPositions.count {
            let end = min(first + max(pointsPerBatch, 1), sortedPositions.count)
            let slice = sortedPositions[first ..< end]
            var batchMin = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
            var batchMax = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)

            for position in slice {
                batchMin = simd_min(batchMin, position)
                batchMax = simd_max(batchMax, position)
            }

            let size = max(batchMax - batchMin, SIMD3<Float>(repeating: 0.000001))

            for pointIndex in first ..< end {
                let normalized = simd_clamp((sortedPositions[pointIndex] - batchMin) / size, .zero, SIMD3<Float>(repeating: 0.99999994))
                let q = SIMD3<UInt32>(
                    UInt32(normalized.x * Float(pointRasteriserSteps30Bit - 1)),
                    UInt32(normalized.y * Float(pointRasteriserSteps30Bit - 1)),
                    UInt32(normalized.z * Float(pointRasteriserSteps30Bit - 1))
                )

                let xLow = (q.x >> 20) & pointRasteriserMask10Bit
                let yLow = (q.y >> 20) & pointRasteriserMask10Bit
                let zLow = (q.z >> 20) & pointRasteriserMask10Bit
                let xMed = (q.x >> 10) & pointRasteriserMask10Bit
                let yMed = (q.y >> 10) & pointRasteriserMask10Bit
                let zMed = (q.z >> 10) & pointRasteriserMask10Bit
                let xHigh = q.x & pointRasteriserMask10Bit
                let yHigh = q.y & pointRasteriserMask10Bit
                let zHigh = q.z & pointRasteriserMask10Bit

                xyzLow[pointIndex] = xLow | (yLow << 10) | (zLow << 20)
                xyzMed[pointIndex] = xMed | (yMed << 10) | (zMed << 20)
                xyzHigh[pointIndex] = xHigh | (yHigh << 10) | (zHigh << 20)
            }

            var batch = RasterBatch(
                min: batchMin,
                max: batchMax,
                numPoints: UInt32(end - first),
                firstPoint: UInt32(first),
                fileIndex: 0
            )
            var cumulative = [Int](repeating: 0, count: 8)
            for pointIndex in first ..< end {
                cumulative[Int(levels[pointIndex]) & 7] += 1
            }
            for level in 1 ..< 8 {
                cumulative[level] += cumulative[level - 1]
            }
            batch.setLODCumulativeCounts(cumulative)
            batches.append(batch)
            first = end
        }

        let packedColors = sortedColorsSrc.map { color -> UInt32 in
            let r = UInt32(simd_clamp(color.x, 0.0, 1.0) * 255.0)
            let g = UInt32(simd_clamp(color.y, 0.0, 1.0) * 255.0)
            let b = UInt32(simd_clamp(color.z, 0.0, 1.0) * 255.0)
            let a = UInt32(simd_clamp(color.w, 0.0, 1.0) * 255.0)
            return r | (g << 8) | (b << 16) | (a << 24)
        }

        var result = PackedPointCloud(
            batches: batches,
            files: [RasterFile()],
            xyzLow: xyzLow,
            xyzMed: xyzMed,
            xyzHigh: xyzHigh,
            colors: packedColors,
            levels: levels,
            boundsMin: boundsMin,
            boundsMax: boundsMax,
            orderedPositions: sortedPositions,
            sourceIndices: order.map { UInt32($0) }
        )

        if shouldShuffleBatches {
            shuffleBatches(&result)
        }

        return result
    }

    /// Decodes a single point's position at the finest (30-bit, level 0)
    /// precision from `packed`'s `xyzLow`/`xyzMed`/`xyzHigh` buffers, relative
    /// to `batch`'s AABB. Mirrors the depth/color kernels' level-0 decode path
    /// (see `Common.metal` in the sibling `Satin-ComputeRasteriser`); kept here
    /// as the CPU-side round-trip reference for packer tests.
    public static func decodePosition30Bit(pointIndex: Int, batch: RasterBatch, packed: PackedPointCloud) -> SIMD3<Float> {
        let batchMin = SIMD3<Float>(batch.minX, batch.minY, batch.minZ)
        let batchMax = SIMD3<Float>(batch.maxX, batch.maxY, batch.maxZ)
        let batchSize = max(batchMax - batchMin, SIMD3<Float>(repeating: 0.000001))

        let low = packed.xyzLow[pointIndex]
        let med = packed.xyzMed[pointIndex]
        let high = packed.xyzHigh[pointIndex]
        let x = (unpack10(low, shift: 0) << 20) | (unpack10(med, shift: 0) << 10) | unpack10(high, shift: 0)
        let y = (unpack10(low, shift: 10) << 20) | (unpack10(med, shift: 10) << 10) | unpack10(high, shift: 10)
        let z = (unpack10(low, shift: 20) << 20) | (unpack10(med, shift: 20) << 10) | unpack10(high, shift: 20)
        return SIMD3<Float>(Float(x), Float(y), Float(z)) * (batchSize / Float(pointRasteriserSteps30Bit)) + batchMin
    }

    private static func unpack10(_ encoded: UInt32, shift: UInt32) -> UInt32 {
        (encoded >> shift) & pointRasteriserMask10Bit
    }
}

/// Swaps every-other batch with its mirror from the end of the batch array
/// (batch 1 ↔ batch N-2, batch 3 ↔ batch N-4, …), per the Magnopus
/// write-up's inter-wave-contention mitigation: adjacent threadgroups no
/// longer draw batches with adjacent Morton keys, so their atomic writes
/// spread across the pixel buffer instead of colliding.
///
/// Implemented as a whole-batch reorder: batches are variable-length, so
/// rather than swapping unequal-sized in-place slices, this computes the new
/// batch order, then rebuilds every per-point array (`xyzLow`/`xyzMed`/
/// `xyzHigh`/`colors`/`levels`/`orderedPositions`/`sourceIndices`) by
/// concatenating each batch's original point range in the new order, and
/// rewrites each ``RasterBatch/firstPoint`` to its new offset. O(pointCount).
public func shuffleBatches(_ cloud: inout PackedPointCloud) {
    let n = cloud.batches.count
    guard n >= 2 else { return }

    var newOrder = Array(0 ..< n)
    var i = 1
    while i < n - 1 - i {
        newOrder.swapAt(i, n - 1 - i)
        i += 2
    }
    guard newOrder != Array(0 ..< n) else { return }

    let oldBatches = cloud.batches
    let totalPoints = cloud.pointCount
    let hasOrderedPositions = !cloud.orderedPositions.isEmpty
    let hasSourceIndices = !cloud.sourceIndices.isEmpty

    var newBatches: [RasterBatch] = []
    newBatches.reserveCapacity(n)
    var newXyzLow = [UInt32](); newXyzLow.reserveCapacity(totalPoints)
    var newXyzMed = [UInt32](); newXyzMed.reserveCapacity(totalPoints)
    var newXyzHigh = [UInt32](); newXyzHigh.reserveCapacity(totalPoints)
    var newColors = [UInt32](); newColors.reserveCapacity(totalPoints)
    var newLevels = [UInt8](); newLevels.reserveCapacity(totalPoints)
    var newOrderedPositions: [SIMD3<Float>] = []
    if hasOrderedPositions { newOrderedPositions.reserveCapacity(totalPoints) }
    var newSourceIndices: [UInt32] = []
    if hasSourceIndices { newSourceIndices.reserveCapacity(totalPoints) }

    var cursor: UInt32 = 0
    for oldBatchIndex in newOrder {
        var batch = oldBatches[oldBatchIndex]
        let first = Int(batch.firstPoint)
        let count = Int(batch.numPoints)
        let range = first ..< (first + count)

        newXyzLow.append(contentsOf: cloud.xyzLow[range])
        newXyzMed.append(contentsOf: cloud.xyzMed[range])
        newXyzHigh.append(contentsOf: cloud.xyzHigh[range])
        newColors.append(contentsOf: cloud.colors[range])
        newLevels.append(contentsOf: cloud.levels[range])
        if hasOrderedPositions { newOrderedPositions.append(contentsOf: cloud.orderedPositions[range]) }
        if hasSourceIndices { newSourceIndices.append(contentsOf: cloud.sourceIndices[range]) }

        batch.firstPoint = cursor
        newBatches.append(batch)
        cursor += UInt32(count)
    }

    cloud.batches = newBatches
    cloud.xyzLow = newXyzLow
    cloud.xyzMed = newXyzMed
    cloud.xyzHigh = newXyzHigh
    cloud.colors = newColors
    cloud.levels = newLevels
    if hasOrderedPositions { cloud.orderedPositions = newOrderedPositions }
    if hasSourceIndices { cloud.sourceIndices = newSourceIndices }
}

// Stable per-batch-slice 8-bucket counting sort by LOD level: within each
// [first, first+batchStride) slice, points are reordered level-ascending while
// preserving Morton order inside each level. The identical permutation is
// applied to `order` so every array derived from it stays consistent with the
// permuted positions/levels.
private func bucketSortBatchSlicesByLevel(
    order: inout [Int],
    positions: inout [SIMD3<Float>],
    levels: inout [UInt8],
    batchStride: Int
) {
    let count = positions.count
    var first = 0
    while first < count {
        let end = min(first + batchStride, count)

        var cursors = [Int](repeating: 0, count: 9)
        for i in first ..< end {
            cursors[(Int(levels[i]) & 7) + 1] += 1
        }
        for level in 1 ..< 9 {
            cursors[level] += cursors[level - 1]
        }

        // permutation[j] = source index (in the whole array) of the point
        // that lands at slice-relative position j.
        var permutation = [Int](repeating: 0, count: end - first)
        for i in first ..< end {
            let level = Int(levels[i]) & 7
            permutation[cursors[level]] = i
            cursors[level] += 1
        }

        let orderSlice = permutation.map { order[$0] }
        let positionSlice = permutation.map { positions[$0] }
        let levelSlice = permutation.map { levels[$0] }
        for j in 0 ..< permutation.count {
            order[first + j] = orderSlice[j]
            positions[first + j] = positionSlice[j]
            levels[first + j] = levelSlice[j]
        }

        first = end
    }
}

// Z-order points so consecutive entries are spatially close. Tighter per-batch
// AABBs (better frustum culling and precision selection) and better cache
// coherency for neighbouring threadgroup reads in the rasteriser passes.
private func mortonOrder(
    positions: [SIMD3<Float>],
    boundsMin: SIMD3<Float>,
    boundsMax: SIMD3<Float>
) -> [Int] {
    let count = positions.count
    let extent = simd_max(boundsMax - boundsMin, SIMD3<Float>(repeating: 0.000001))
    let scale = SIMD3<Float>(repeating: 1023.0) / extent

    var keys = [UInt32](repeating: 0, count: count)
    for i in 0 ..< count {
        let normalized = simd_clamp((positions[i] - boundsMin) * scale, .zero, SIMD3<Float>(repeating: 1023.0))
        let qx = UInt32(normalized.x)
        let qy = UInt32(normalized.y)
        let qz = UInt32(normalized.z)
        keys[i] = (mortonSpread10(qx) << 2) | (mortonSpread10(qy) << 1) | mortonSpread10(qz)
    }

    // Stable tie-break by original index: equal Morton keys keep ascending
    // input order. This makes the CPU order deterministic and — crucially —
    // bit-identical to the GPU packer's stable LSD radix sort (which preserves
    // identity order across equal keys). Swift's `sort` is not stable, so the
    // explicit `$0 < $1` tie-break is required, not incidental.
    var indices = Array(0 ..< count)
    indices.sort { keys[$0] != keys[$1] ? keys[$0] < keys[$1] : $0 < $1 }
    return indices
}

// Density-aware LOD: for each level (coarsest first) assign points that
// occupy a previously-empty voxel. Coarsest level gets the most spatially
// representative subset; finest level catches the remainder. Levels stored
// in top 3 bits of the packed color word (0 = coarsest visible at distance).
private func computeLODLevels(
    positions: [SIMD3<Float>],
    boundsMin: SIMD3<Float>,
    boundsMax: SIMD3<Float>,
    lodLevels: Int,
    coarseVoxelDivisions: Int
) -> [UInt8] {
    let count = positions.count
    let maxLevel = UInt8(lodLevels - 1)
    var levels = [UInt8](repeating: maxLevel, count: count)
    guard lodLevels > 1 else { return levels }

    let extent = simd_max(boundsMax - boundsMin, SIMD3<Float>(repeating: 0.000001))
    let longestAxis = max(extent.x, max(extent.y, extent.z))
    let baseVoxel = longestAxis / Float(coarseVoxelDivisions)

    // Voxel-occupancy dedup keyed by a single UInt64 (21 bits/axis) rather than
    // SIMD3<Int32> — hashing one integer is far cheaper than Swift's per-
    // component Hasher combine. Cells are non-negative (positions >= boundsMin)
    // and node-local, so the pack is collision-free over the values that occur.
    @inline(__always)
    func cellKey(_ local: SIMD3<Float>) -> UInt64 {
        let cx = UInt64(max(0, Int32(local.x.rounded(.down)))) & 0x1F_FFFF
        let cy = UInt64(max(0, Int32(local.y.rounded(.down)))) & 0x1F_FFFF
        let cz = UInt64(max(0, Int32(local.z.rounded(.down)))) & 0x1F_FFFF
        return (cx << 42) | (cy << 21) | cz
    }

    // Open-addressing hash set of occupied voxel keys, replacing Swift's
    // `Set<UInt64>` (its SipHash + CoW/uniqueness overhead dominates on large
    // clouds). One `[UInt64]` table, sized once for the whole call and reused
    // across levels, with Fibonacci hashing and linear probing in an unsafe
    // buffer (no bounds/uniqueness checks in the probe).
    //
    // Sentinel is `UInt64.max`: cellKey packs 21 bits/axis into bits 0..62, so
    // bit 63 is always 0 and a real key can never equal the sentinel.
    //
    // Capacity = next power of two ≥ 2·count (load factor ≤ 0.5). Sizing by
    // the total count — not by per-level unclaimed counts — is deliberate: the
    // eligible set shrinks with depth but the number of distinct occupied
    // voxels (the actual live entries) grows, and `count` bounds both at every
    // level, so one allocation stays under 0.5 load throughout.
    let sentinel = UInt64.max
    var capacity = 1
    while capacity < count * 2 { capacity <<= 1 }
    let mask = capacity - 1
    let shift = UInt64(64 - capacity.trailingZeroBitCount)

    var table = [UInt64](repeating: sentinel, count: capacity)
    levels.withUnsafeMutableBufferPointer { lv in
        positions.withUnsafeBufferPointer { pos in
            table.withUnsafeMutableBufferPointer { t in
                for level in 0 ..< (lodLevels - 1) {
                    let voxelSize = baseVoxel * powf(0.5, Float(level))
                    let invVoxel = 1.0 / max(voxelSize, 0.000001)
                    // Level 0 uses the freshly sentinel-filled table; refill for
                    // each subsequent level.
                    if level != 0 {
                        for j in 0 ..< capacity { t[j] = sentinel }
                    }
                    for i in 0 ..< count {
                        if lv[i] != maxLevel { continue }
                        let local = (pos[i] - boundsMin) * invVoxel
                        let key = cellKey(local)
                        var slot = Int((key &* 0x9E37_79B9_7F4A_7C15) >> shift)
                        while true {
                            let cur = t[slot]
                            if cur == sentinel {
                                t[slot] = key
                                lv[i] = UInt8(level)
                                break
                            }
                            if cur == key { break }
                            slot = (slot + 1) & mask
                        }
                    }
                }
            }
        }
    }
    return levels
}

private func mortonSpread10(_ value: UInt32) -> UInt32 {
    var x = value & 0x3ff
    x = (x | (x << 16)) & 0x030000ff
    x = (x | (x << 8))  & 0x0300f00f
    x = (x | (x << 4))  & 0x030c30c3
    x = (x | (x << 2))  & 0x09249249
    return x
}
