import Foundation
import simd
import Testing
@testable import SatinPointRasteriser

/// Deterministic pseudo-random cloud (LCG) — multiple batches, multiple levels.
private func jitteredCloud(count: Int = 4000, seed: UInt64 = 0x5EED) -> ([SIMD3<Float>], [SIMD4<Float>]) {
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
        positions.append(SIMD3<Float>(next() * 10 - 5, next() * 10 - 5, next() * 10 - 5))
        colors.append(SIMD4<Float>(next(), next(), next(), 1))
    }
    return (positions, colors)
}

@Test func packRoundTripsPositionsAtLevel0WithinQuantizationEpsilon() {
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 8)
    #expect(!packed.batches.isEmpty)

    for batch in packed.batches {
        let batchMin = SIMD3<Float>(batch.minX, batch.minY, batch.minZ)
        let batchMax = SIMD3<Float>(batch.maxX, batch.maxY, batch.maxZ)
        let batchSize = max(batchMax - batchMin, SIMD3<Float>(repeating: 0.000001))
        // One 30-bit quantization step at the batch's scale, plus float slop.
        let epsilon = simd_length(batchSize) / Float(pointRasteriserSteps30Bit) * 2 + 1e-5

        for localIndex in 0 ..< Int(batch.numPoints) {
            let pointIndex = Int(batch.firstPoint) + localIndex
            let decoded = PackedPointCloudFixtures.decodePosition30Bit(pointIndex: pointIndex, batch: batch, packed: packed)
            let original = packed.orderedPositions[pointIndex]
            let error = simd_length(decoded - original)
            #expect(error <= epsilon, "point \(pointIndex) decoded error \(error) exceeds epsilon \(epsilon)")
        }
    }
}

@Test func packPerBatchAABBsContainTheirPoints() {
    let (positions, colors) = jitteredCloud()
    let packed = PackedPointCloudFixtures.pack(
        positions: positions, colors: colors, pointsPerBatch: 256, lodLevels: 4, coarseVoxelDivisions: 8
    )
    let slop: Float = 1e-4
    for batch in packed.batches {
        let batchMin = SIMD3<Float>(batch.minX, batch.minY, batch.minZ) - slop
        let batchMax = SIMD3<Float>(batch.maxX, batch.maxY, batch.maxZ) + slop
        for localIndex in 0 ..< Int(batch.numPoints) {
            let pointIndex = Int(batch.firstPoint) + localIndex
            let p = packed.orderedPositions[pointIndex]
            #expect(p.x >= batchMin.x && p.x <= batchMax.x)
            #expect(p.y >= batchMin.y && p.y <= batchMax.y)
            #expect(p.z >= batchMin.z && p.z <= batchMax.z)
        }
    }
}

@Test func packLODCumulativeCountsAreConsistent() {
    let (positions, colors) = jitteredCloud()
    let packed = PackedPointCloudFixtures.pack(
        positions: positions, colors: colors, pointsPerBatch: 256, lodLevels: 4, coarseVoxelDivisions: 8
    )
    for batch in packed.batches {
        let first = Int(batch.firstPoint)
        let slice = packed.levels[first ..< first + Int(batch.numPoints)]
        var recount = [Int](repeating: 0, count: 8)
        for level in slice { recount[Int(level) & 7] += 1 }
        for level in 1 ..< 8 { recount[level] += recount[level - 1] }
        let stored = batch.lodCumulativeCounts.map(Int.init)
        #expect(stored == recount)
        #expect(stored[7] == Int(batch.numPoints))
    }
}

@Test func packPreservesPointCountAndColorMultisetUnderMortonOrdering() {
    let positions: [SIMD3<Float>] = [
        SIMD3<Float>(0, 0, 0),
        SIMD3<Float>(1, 0, 0),
        SIMD3<Float>(0, 1, 0),
        SIMD3<Float>(0, 0, 1),
        SIMD3<Float>(1, 1, 1),
    ]
    let colors: [SIMD4<Float>] = [
        SIMD4<Float>(1, 0, 0, 1),
        SIMD4<Float>(0, 1, 0, 1),
        SIMD4<Float>(0, 0, 1, 1),
        SIMD4<Float>(1, 1, 0, 1),
        SIMD4<Float>(1, 1, 1, 1),
    ]

    let packed = PackedPointCloudFixtures.pack(positions: positions, colors: colors, pointsPerBatch: 2)
    #expect(packed.pointCount == 5)

    let expected: [UInt32: Int] = [
        0xff0000ff: 1,
        0xff00ff00: 1,
        0xffff0000: 1,
        0xff00ffff: 1,
        0xffffffff: 1,
    ]
    var actual: [UInt32: Int] = [:]
    for c in packed.colors { actual[c, default: 0] += 1 }
    #expect(actual == expected)
}

/// (position, color) pair keyed by exact float bits so shuffled arrays compare
/// as an unordered multiset — the shuffle only relocates whole point blocks,
/// never re-derives values, so bit-exact equality is the correct check.
private struct PositionColorKey: Hashable {
    let x: Float, y: Float, z: Float
    let color: UInt32
}

private func multiset(_ cloud: PackedPointCloud) -> [PositionColorKey: Int] {
    var m: [PositionColorKey: Int] = [:]
    for i in 0 ..< cloud.pointCount {
        let p = cloud.orderedPositions[i]
        let key = PositionColorKey(x: p.x, y: p.y, z: p.z, color: cloud.colors[i])
        m[key, default: 0] += 1
    }
    return m
}

@Test func shuffleBatchesPreservesPointBatchAssociationAndPermutesOrder() {
    let (positions, colors) = jitteredCloud(count: 6000, seed: 0xB0CE7)
    let unshuffled = PackedPointCloudFixtures.pack(
        positions: positions, colors: colors, pointsPerBatch: 256, lodLevels: 4, coarseVoxelDivisions: 8, shuffleBatches: false
    )
    #expect(unshuffled.batchCount >= 6, "fixture must produce >= 6 batches to exercise the shuffle")

    var shuffled = unshuffled
    shuffleBatches(&shuffled)

    // Batch order actually changed for N >= 6 batches.
    let unshuffledMins = unshuffled.batches.map { SIMD3<Float>($0.minX, $0.minY, $0.minZ) }
    let shuffledMins = shuffled.batches.map { SIMD3<Float>($0.minX, $0.minY, $0.minZ) }
    #expect(unshuffledMins != shuffledMins, "shuffle should permute batch order for N >= 6 batches")

    #expect(shuffled.pointCount == unshuffled.pointCount)
    #expect(shuffled.batchCount == unshuffled.batchCount)
    #expect(Set(shuffled.sourceIndices).count == shuffled.pointCount, "sourceIndices must stay a permutation after shuffle")

    // Every batch's AABB still contains its (decoded/ordered) points after shuffle.
    let slop: Float = 1e-4
    for batch in shuffled.batches {
        let batchMin = SIMD3<Float>(batch.minX, batch.minY, batch.minZ) - slop
        let batchMax = SIMD3<Float>(batch.maxX, batch.maxY, batch.maxZ) + slop
        for localIndex in 0 ..< Int(batch.numPoints) {
            let pointIndex = Int(batch.firstPoint) + localIndex
            let p = shuffled.orderedPositions[pointIndex]
            #expect(p.x >= batchMin.x && p.x <= batchMax.x)
            #expect(p.y >= batchMin.y && p.y <= batchMax.y)
            #expect(p.z >= batchMin.z && p.z <= batchMax.z)

            // decodePosition30Bit round-trips against the *shuffled* batch/point data too.
            let decoded = PackedPointCloudFixtures.decodePosition30Bit(pointIndex: pointIndex, batch: batch, packed: shuffled)
            #expect(simd_length(decoded - p) <= simd_length(batchMax - batchMin) / Float(pointRasteriserSteps30Bit) * 2 + 1e-4)
        }
    }

    // The multiset of (position, color) pairs is unchanged by the shuffle.
    #expect(multiset(unshuffled) == multiset(shuffled))
}

@Test func packShufflesBatchesByDefault() {
    let (positions, colors) = jitteredCloud(count: 6000, seed: 0x1234)
    let unshuffled = PackedPointCloudFixtures.pack(
        positions: positions, colors: colors, pointsPerBatch: 256, lodLevels: 4, coarseVoxelDivisions: 8, shuffleBatches: false
    )
    let shuffledByDefault = PackedPointCloudFixtures.pack(
        positions: positions, colors: colors, pointsPerBatch: 256, lodLevels: 4, coarseVoxelDivisions: 8
    )
    #expect(unshuffled.batchCount >= 6)

    let unshuffledMins = unshuffled.batches.map { SIMD3<Float>($0.minX, $0.minY, $0.minZ) }
    let shuffledMins = shuffledByDefault.batches.map { SIMD3<Float>($0.minX, $0.minY, $0.minZ) }
    #expect(unshuffledMins != shuffledMins, "pack() should shuffle batches by default")
    #expect(multiset(unshuffled) == multiset(shuffledByDefault))
}
