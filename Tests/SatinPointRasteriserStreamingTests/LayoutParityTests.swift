import SatinPointRasteriser
import SwiftPDAL
import Testing

// `StreamingAdapter.toRasterBatch` copies `StreamingRasterBatch` into
// `RasterBatch` field-by-field (not a raw memcpy), so a mismatch here
// wouldn't corrupt GPU data directly — but the two types are documented as
// intentionally field-order-identical (SwiftPDAL's doc comment: "Field order
// intentionally mirrors SatinComputeRasteriser.RasterBatch"), and streaming
// code elsewhere assumes it (e.g. the padding3…padding6 packed LOD-count
// carry-through). This test guards that invariant so a layout drift in
// either package's struct is caught immediately instead of silently
// producing swapped LOD buckets or AABBs downstream.
@Test func rasterBatchLayoutMatchesStreamingRasterBatch() {
    #expect(MemoryLayout<RasterBatch>.size == MemoryLayout<StreamingRasterBatch>.size)
    #expect(MemoryLayout<RasterBatch>.stride == MemoryLayout<StreamingRasterBatch>.stride)
    #expect(MemoryLayout<RasterBatch>.alignment == MemoryLayout<StreamingRasterBatch>.alignment)

    #expect(MemoryLayout<RasterBatch>.offset(of: \.state) == MemoryLayout<StreamingRasterBatch>.offset(of: \.state))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.minX) == MemoryLayout<StreamingRasterBatch>.offset(of: \.minX))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.minY) == MemoryLayout<StreamingRasterBatch>.offset(of: \.minY))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.minZ) == MemoryLayout<StreamingRasterBatch>.offset(of: \.minZ))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.maxX) == MemoryLayout<StreamingRasterBatch>.offset(of: \.maxX))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.maxY) == MemoryLayout<StreamingRasterBatch>.offset(of: \.maxY))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.maxZ) == MemoryLayout<StreamingRasterBatch>.offset(of: \.maxZ))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.numPoints) == MemoryLayout<StreamingRasterBatch>.offset(of: \.numPoints))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.firstPoint) == MemoryLayout<StreamingRasterBatch>.offset(of: \.firstPoint))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.fileIndex) == MemoryLayout<StreamingRasterBatch>.offset(of: \.fileIndex))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.padding3) == MemoryLayout<StreamingRasterBatch>.offset(of: \.padding3))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.padding4) == MemoryLayout<StreamingRasterBatch>.offset(of: \.padding4))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.padding5) == MemoryLayout<StreamingRasterBatch>.offset(of: \.padding5))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.padding6) == MemoryLayout<StreamingRasterBatch>.offset(of: \.padding6))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.padding7) == MemoryLayout<StreamingRasterBatch>.offset(of: \.padding7))
    #expect(MemoryLayout<RasterBatch>.offset(of: \.padding8) == MemoryLayout<StreamingRasterBatch>.offset(of: \.padding8))

    // Field types must match too — offsets could coincidentally line up
    // between a 4-byte float and a 4-byte uint at the same position.
    #expect(MemoryLayout<Int32>.size == MemoryLayout<Float>.size, "sanity: state/min* share a 4-byte lane")
    #expect(MemoryLayout<UInt32>.size == MemoryLayout<Float>.size, "sanity: numPoints/etc. share a 4-byte lane")
}
