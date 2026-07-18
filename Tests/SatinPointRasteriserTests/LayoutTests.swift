import Foundation
import Testing
@testable import SatinPointRasteriser

@Test func sharedTypeStridesMatchMetalExpectations() {
    #expect(PointRasteriserLayout.rasterBatchStride == 64)
    #expect(PointRasteriserLayout.rasterFileStride == 256)
    #expect(PointRasteriserLayout.rasterPixelStride == 32)
    #expect(PointRasteriserLayout.visibleBatchStride == 16)
    #expect(PointRasteriserLayout.dispatchArgsStride == 12)
    #expect(MemoryLayout<UInt64>.stride == 8)
}

// Guards the cross-package memcpy from a streaming source's RasterBatch into
// this package's RasterBatch. Field offsets here must match exactly;
// mismatches would silently corrupt batch metadata.
@Test func rasterBatchFieldOffsetsMatchStreamingLayout() {
    #expect(MemoryLayout<RasterBatch>.offset(of: \.state) == 0)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.minX) == 4)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.maxZ) == 24)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.numPoints) == 28)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.firstPoint) == 32)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.fileIndex) == 36)
}

@Test func rasterBatchDefaultsToResident() {
    let batch = RasterBatch(min: .zero, max: .one, numPoints: 1, firstPoint: 0)
    #expect(batch.state == 1)
}

// Full byte-layout guard for the slot-pool `addBatches` memcpy path (Slice 8a).
// A streaming source hands over chunk metadata as blobs of its own
// `StreamingRasterBatch`; this package memcpy's them into `RasterBatch`, so the
// two must be byte-identical. Offsets/stride are hardcoded (mirrors the sibling's
// LayoutTests approach) — do NOT import the streaming target here.
@Test func rasterBatchIsByteIdenticalToStreamingContract() {
    #expect(MemoryLayout<RasterBatch>.stride == 64)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.state) == 0)      // int32
    #expect(MemoryLayout<RasterBatch>.offset(of: \.minX) == 4)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.minY) == 8)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.minZ) == 12)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.maxX) == 16)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.maxY) == 20)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.maxZ) == 24)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.numPoints) == 28) // uint32
    #expect(MemoryLayout<RasterBatch>.offset(of: \.firstPoint) == 32)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.fileIndex) == 36)
    // padding3..8 (the LOD cumulative-count words) occupy 40..63.
    #expect(MemoryLayout<RasterBatch>.offset(of: \.padding3) == 40)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.padding8) == 60)
}

@Test func lodCloudLayoutStridesMatchPackedFloat3AndUInt() {
    #expect(LODCloudLayout.positionStride == 12)
    #expect(LODCloudLayout.colorStride == 4)
    #expect(LODCloudLayout.sourceIndexStride == 4)
}

@Test func fixturePackingProducesConsistentCounts() {
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 4)
    #expect(packed.pointCount == 64)
    #expect(packed.colors.count == packed.pointCount)
    #expect(packed.xyzLow.count == packed.pointCount)
    #expect(packed.xyzMed.count == packed.pointCount)
    #expect(packed.xyzHigh.count == packed.pointCount)
    #expect(!packed.batches.isEmpty)
    #expect(packed.files.count == 1)
}
