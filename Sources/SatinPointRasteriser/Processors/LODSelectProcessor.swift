import Metal
import Satin
import simd

/// LOD compaction pass: one threadgroup per source batch, appending CLOD
/// survivors into a cloud's compacted SoA LOD buffers. See `LODSelect/Shaders.metal`.
open class LODSelectProcessor: BasePointRasteriserProcessor {
    /// Number of source batches to process this dispatch (one threadgroup each).
    /// Per-cloud, so it drives the dispatch width but is not a uniform.
    public var batchCount: Int = 0

    /// Index of the first source batch to process (amortized sweep cursor).
    /// Bound via `setBytes` at raw buffer index 13.
    public var firstBatch: UInt32 = 0

    /// Per-cloud LOD capacity, bound via `setBytes` at raw buffer index 12
    /// (shared per-frame uniforms can't carry a per-cloud value).
    public var lodCapacity: UInt32 = 0

    public var screenSize: SIMD2<UInt32> = .zero {
        didSet { set("screenSize", simd_int2(Int32(screenSize.x), Int32(screenSize.y))) }
    }
    public var viewMatrix: simd_float4x4 = matrix_identity_float4x4 {
        didSet { set("viewMatrix", viewMatrix) }
    }
    public var projectionMatrix: simd_float4x4 = matrix_identity_float4x4 {
        didSet { set("projectionMatrix", projectionMatrix) }
    }
    public var enableFrustumCulling: Bool = true {
        didSet { set("enableFrustumCulling", enableFrustumCulling ? 1 : 0) }
    }
    public var lodBias: Int = 0 {
        didSet { set("lodBias", lodBias) }
    }
    public var enableCLOD: Bool = true {
        didSet { set("enableCLOD", enableCLOD ? 1 : 0) }
    }
    public var lodDither: Bool = true {
        didSet { set("lodDither", lodDither ? 1 : 0) }
    }

    public var batchesBuffer: MTLBuffer? { didSet { set(batchesBuffer, index: .Custom0) } }
    public var xyzLowBuffer: MTLBuffer? { didSet { set(xyzLowBuffer, index: .Custom1) } }
    public var xyzMedBuffer: MTLBuffer? { didSet { set(xyzMedBuffer, index: .Custom2) } }
    public var xyzHighBuffer: MTLBuffer? { didSet { set(xyzHighBuffer, index: .Custom3) } }
    public var filesBuffer: MTLBuffer? { didSet { set(filesBuffer, index: .Custom4) } }
    /// Byte offset into ``filesBuffer`` for the current ring slot.
    public var filesBufferOffset: Int = 0
    public var colorsBuffer: MTLBuffer? { didSet { set(colorsBuffer, index: .Custom5) } }
    public var levelsBuffer: MTLBuffer? { didSet { set(levelsBuffer, index: .Custom6) } }
    public var lodPositionsBuffer: MTLBuffer? { didSet { set(lodPositionsBuffer, index: .Custom7) } }
    public var lodColorsBuffer: MTLBuffer? { didSet { set(lodColorsBuffer, index: .Custom8) } }
    public var lodSourceIndicesBuffer: MTLBuffer? { didSet { set(lodSourceIndicesBuffer, index: .Custom9) } }
    public var lodStatsBuffer: MTLBuffer? { didSet { set(lodStatsBuffer, index: .Custom10) } }

    private static let lodCapacityIndex = 12
    private static let firstBatchIndex = 13

    open override func applyAdditionalBindings(_ computeEncoder: MTLComputeCommandEncoder) {
        if let filesBuffer {
            computeEncoder.setBuffer(filesBuffer, offset: filesBufferOffset, index: ComputeBufferIndex.Custom4.rawValue)
        }
        var capacity = lodCapacity
        computeEncoder.setBytes(&capacity, length: MemoryLayout<UInt32>.stride, index: Self.lodCapacityIndex)
        var first = firstBatch
        computeEncoder.setBytes(&first, length: MemoryLayout<UInt32>.stride, index: Self.firstBatchIndex)
    }

    open override func setup() {
        super.setup()
        set("screenSize", simd_int2(Int32(screenSize.x), Int32(screenSize.y)))
        set("viewMatrix", viewMatrix)
        set("projectionMatrix", projectionMatrix)
        set("enableFrustumCulling", enableFrustumCulling ? 1 : 0)
        set("lodBias", lodBias)
        set("enableCLOD", enableCLOD ? 1 : 0)
        set("lodDither", lodDither ? 1 : 0)
    }

    var isEncodeReady: Bool {
        batchCount > 0
            && screenSize.x > 0 && screenSize.y > 0
            && batchesBuffer != nil
            && xyzLowBuffer != nil && xyzMedBuffer != nil && xyzHighBuffer != nil
            && filesBuffer != nil
            && colorsBuffer != nil && levelsBuffer != nil
            && lodPositionsBuffer != nil && lodColorsBuffer != nil
            && lodSourceIndicesBuffer != nil && lodStatsBuffer != nil
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: isEncodeReady)
    }

    // One threadgroup per batch, PR_THREADS_PER_GROUP threads each. Both
    // dispatch entry points route here (this pass is always threadgroup-per-batch).
#if os(macOS) || os(iOS) || os(visionOS)
    open override func dispatchThreads(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        dispatchThreadgroups(computeEncoder: computeEncoder, pipeline: pipeline, iteration: iteration)
    }
#endif

    open override func dispatchThreadgroups(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        guard batchCount > 0 else { return }
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: batchCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: pointRasteriserThreadsPerGroup, height: 1, depth: 1)
        )
    }
}
