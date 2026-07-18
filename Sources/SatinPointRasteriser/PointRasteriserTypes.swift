import Foundation
import simd

/// Threadgroup width used by the rasteriser's compute kernels.
public let pointRasteriserThreadsPerGroup = 128
/// Quantization step count for the finest (30-bit) per-point position encoding.
public let pointRasteriserSteps30Bit: UInt32 = 1_073_741_824
/// Mask isolating one 10-bit quantized coordinate component.
public let pointRasteriserMask10Bit: UInt32 = 1_023

/// A batch (chunk) of points sharing one axis-aligned bounding box, level-of-detail
/// bucketing, and source file. Byte-identical to the Metal `RasterBatch` in
/// `PointRasteriserTypes.h` — a proven cross-ABI contract carried over from
/// Satin-ComputeRasteriser; do not reorder or resize fields without updating both sides.
public struct RasterBatch: Sendable {
    public var state: Int32
    public var minX: Float
    public var minY: Float
    public var minZ: Float
    public var maxX: Float
    public var maxY: Float
    public var maxZ: Float
    public var numPoints: UInt32
    public var firstPoint: UInt32
    public var fileIndex: UInt32
    public var padding3: UInt32
    public var padding4: UInt32
    public var padding5: UInt32
    public var padding6: UInt32
    public var padding7: UInt32
    public var padding8: UInt32

    /// Creates a batch record covering `[min, max]` with `numPoints` points starting at `firstPoint`.
    public init(
        min: SIMD3<Float>,
        max: SIMD3<Float>,
        numPoints: UInt32,
        firstPoint: UInt32,
        fileIndex: UInt32 = 0,
        state: Int32 = 1
    ) {
        self.state = state
        self.minX = min.x
        self.minY = min.y
        self.minZ = min.z
        self.maxX = max.x
        self.maxY = max.y
        self.maxZ = max.z
        self.numPoints = numPoints
        self.firstPoint = firstPoint
        self.fileIndex = fileIndex
        self.padding3 = 0
        self.padding4 = 0
        self.padding5 = 0
        self.padding6 = 0
        self.padding7 = 0
        self.padding8 = 0
    }

    /// Cumulative LOD level counts for a level-bucketed batch.
    ///
    /// `counts[L]` is the number of points in the batch with level ≤ `L`, so
    /// `counts[7] == numPoints` for a bucketed batch. The counts live packed
    /// two-per-word in ``padding3``…``padding6`` (low level in the low half);
    /// an all-zero result — specifically `counts[7] == 0` — is the legacy
    /// sentinel meaning the batch's points are *not* stored level-ascending
    /// and the cull pass must draw the full `numPoints` range.
    public var lodCumulativeCounts: [UInt16] {
        [
            UInt16(truncatingIfNeeded: padding3), UInt16(truncatingIfNeeded: padding3 >> 16),
            UInt16(truncatingIfNeeded: padding4), UInt16(truncatingIfNeeded: padding4 >> 16),
            UInt16(truncatingIfNeeded: padding5), UInt16(truncatingIfNeeded: padding5 >> 16),
            UInt16(truncatingIfNeeded: padding6), UInt16(truncatingIfNeeded: padding6 >> 16),
        ]
    }

    /// Stores the 8 cumulative LOD level counts into ``padding3``…``padding6``.
    ///
    /// Callers (packers that store batch points level-ascending) must pass
    /// `counts[L]` = number of points with level ≤ `L`, with
    /// `counts[7] == numPoints`; a non-zero final word is what marks the batch
    /// as bucketed for the cull kernel (see ``lodCumulativeCounts``).
    ///
    /// - Parameter counts: Exactly 8 non-decreasing cumulative counts, each
    ///   ≤ 65535 (batches must therefore hold at most 65535 points).
    public mutating func setLODCumulativeCounts(_ counts: [Int]) {
        precondition(counts.count == 8, "expected 8 cumulative level counts, got \(counts.count)")
        for count in counts {
            precondition(count >= 0 && count <= 65535, "cumulative count \(count) does not fit uint16")
        }
        padding3 = UInt32(counts[0]) | (UInt32(counts[1]) << 16)
        padding4 = UInt32(counts[2]) | (UInt32(counts[3]) << 16)
        padding5 = UInt32(counts[4]) | (UInt32(counts[5]) << 16)
        padding6 = UInt32(counts[6]) | (UInt32(counts[7]) << 16)
    }
}

/// Per-file transforms bound alongside a batch's points. Byte-identical to the Metal `RasterFile`.
public struct RasterFile: Sendable {
    public var transform: simd_float4x4
    public var transformFrustum: simd_float4x4
    public var world: simd_float4x4
    /// Previous frame's `transform` (view-projection · world), used for
    /// per-point screen-space velocity (e.g. motion blur).
    public var prevTransform: simd_float4x4

    /// Creates a file record; `prevTransform` defaults to `transform`.
    public init(
        transform: simd_float4x4 = matrix_identity_float4x4,
        transformFrustum: simd_float4x4 = matrix_identity_float4x4,
        world: simd_float4x4 = matrix_identity_float4x4
    ) {
        self.transform = transform
        self.transformFrustum = transformFrustum
        self.world = world
        self.prevTransform = transform
    }
}

/// One screen-sized accumulation cell (depth + color sum + count), atomically
/// updated by the depth/color passes. Byte-identical to the Metal `RasterPixel`.
public struct RasterPixel: Sendable {
    public var depth: UInt32
    public var red: UInt32
    public var green: UInt32
    public var blue: UInt32
    public var count: UInt32
    /// Σ(coverage·255) accumulated when translucent-defocus is on; 0 otherwise.
    public var weight: UInt32
    public var padding: SIMD2<UInt32>

    /// Creates a zeroed (or explicitly valued) accumulation cell.
    public init(depth: UInt32 = 0, red: UInt32 = 0, green: UInt32 = 0, blue: UInt32 = 0, count: UInt32 = 0, weight: UInt32 = 0) {
        self.depth = depth
        self.red = red
        self.green = green
        self.blue = blue
        self.count = count
        self.weight = weight
        self.padding = .zero
    }
}

/// A CPU- or GPU-packed point cloud: batch metadata, per-file transforms, and
/// the quantized 30-bit position/color/level buffers produced by
/// ``PackedPointCloudFixtures/pack(positions:colors:pointsPerBatch:lodLevels:coarseVoxelDivisions:shuffleBatches:)``.
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
    /// Original `[SIMD3<Float>]` reordered into pack order (Morton-sorted, then
    /// batch-shuffled). Empty when a loader didn't preserve them.
    public var orderedPositions: [SIMD3<Float>]
    /// The pack permutation: `sourceIndices[packedIndex]` is the index of that
    /// point in the loader's **original** (pre-pack) input arrays. Lets a
    /// caller map a rasteriser `pointIndex` back to the source point.
    public var sourceIndices: [UInt32]

    /// Total number of points across all batches.
    public var pointCount: Int { colors.count }
    /// Total number of batches.
    public var batchCount: Int { batches.count }

    /// Creates a packed point cloud from its constituent buffers.
    public init(
        batches: [RasterBatch],
        files: [RasterFile],
        xyzLow: [UInt32],
        xyzMed: [UInt32],
        xyzHigh: [UInt32],
        colors: [UInt32],
        levels: [UInt8],
        boundsMin: SIMD3<Float>,
        boundsMax: SIMD3<Float>,
        orderedPositions: [SIMD3<Float>] = [],
        sourceIndices: [UInt32] = []
    ) {
        self.batches = batches
        self.files = files
        self.xyzLow = xyzLow
        self.xyzMed = xyzMed
        self.xyzHigh = xyzHigh
        self.colors = colors
        self.levels = levels
        self.boundsMin = boundsMin
        self.boundsMax = boundsMax
        self.orderedPositions = orderedPositions
        self.sourceIndices = sourceIndices
    }
}

/// A batch that survived frustum culling for the current frame, with its
/// selected precision level and LOD survivor-prefix length. Byte-identical to
/// the Metal `VisibleBatch`.
public struct VisibleBatch: Sendable {
    public var batchIndex: UInt32
    public var level: Int32
    public var lodThreshold: Float
    /// Number of points at the start of the batch the draw passes iterate —
    /// the LOD survivor prefix the cull kernel computes from
    /// ``RasterBatch/lodCumulativeCounts``. Equals the batch's full
    /// `numPoints` for legacy (unbucketed) batches.
    public var activePoints: UInt32

    /// Creates a visible-batch record; `activePoints` defaults to 0 (filled in by the cull kernel).
    public init(batchIndex: UInt32 = 0, level: Int32 = 0, lodThreshold: Float = 0) {
        self.batchIndex = batchIndex
        self.level = level
        self.lodThreshold = lodThreshold
        self.activePoints = 0
    }
}

/// Indirect-dispatch threadgroup counts written by a cull/finalize pass and
/// consumed by `dispatchThreadgroups(indirectBuffer:...)`. Byte-identical to the Metal `CRDispatchArgs`.
public struct CRDispatchArgs: Sendable {
    public var threadgroupsX: UInt32
    public var threadgroupsY: UInt32
    public var threadgroupsZ: UInt32

    /// Creates a dispatch-args record; `threadgroupsY`/`threadgroupsZ` default to 1.
    public init(threadgroupsX: UInt32 = 0, threadgroupsY: UInt32 = 1, threadgroupsZ: UInt32 = 1) {
        self.threadgroupsX = threadgroupsX
        self.threadgroupsY = threadgroupsY
        self.threadgroupsZ = threadgroupsZ
    }
}

/// `MemoryLayout` strides for the shared Swift/Metal structs, asserted against
/// the Metal side's `sizeof(...)` in tests.
public enum PointRasteriserLayout {
    public static let rasterBatchStride = MemoryLayout<RasterBatch>.stride
    public static let rasterFileStride = MemoryLayout<RasterFile>.stride
    public static let rasterPixelStride = MemoryLayout<RasterPixel>.stride
    public static let visibleBatchStride = MemoryLayout<VisibleBatch>.stride
    public static let dispatchArgsStride = MemoryLayout<CRDispatchArgs>.stride
}

/// Layout documentation for the compacted, amortized **LOD point cloud** the
/// (future) LODSelect pass emits: the CLOD survivors for the current frame,
/// stored SoA rather than as one AoS struct so kernels that only need
/// positions (e.g. the depth pass) don't pull colors/indices through cache.
///
/// Three parallel buffers, one entry per surviving point, same index `i` across all three:
/// - `positions[i]: packed_float3` — object-space position, 12 B stride, no padding.
/// - `colors[i]: uint` — packed RGBA8 color, 4 B stride.
/// - `sourceIndices[i]: uint` — index of this point in its cloud's pack order
///   (i.e. into `PackedPointCloud.orderedPositions`/`sourceIndices`), 4 B stride.
///
/// No Swift/Metal struct models this directly — allocate three `MTLBuffer`s
/// (or three `[T]`s pre-GPU) sized to the frame's survivor count.
public enum LODCloudLayout {
    /// Stride of one `packed_float3` position entry, in bytes.
    public static let positionStride = 12
    /// Stride of one packed-color entry, in bytes.
    public static let colorStride = MemoryLayout<UInt32>.stride
    /// Stride of one source-index entry, in bytes.
    public static let sourceIndexStride = MemoryLayout<UInt32>.stride
}
