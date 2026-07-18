import Metal
import Satin
import simd

/// Depth pass over the compacted LOD point cloud (one thread per LOD point,
/// indirect dispatch). See `DepthPass/Shaders.metal`.
open class DepthPassProcessor: BasePointRasteriserProcessor {
    public var screenSize: SIMD2<UInt32> = .zero {
        didSet { set("screenSize", simd_int2(Int32(screenSize.x), Int32(screenSize.y))) }
    }
    public var viewMatrix: simd_float4x4 = matrix_identity_float4x4 {
        didSet { set("viewMatrix", viewMatrix) }
    }
    public var projectionMatrix: simd_float4x4 = matrix_identity_float4x4 {
        didSet { set("projectionMatrix", projectionMatrix) }
    }
    public var pointSizeMode: PointSizeMode = .screenSpace {
        didSet { set("pointSizeMode", Int(pointSizeMode.rawValue)) }
    }
    public var minimumPointSize: Float = 1.0 {
        didSet { set("minimumPointSize", minimumPointSize) }
    }
    public var maximumPointSize: Float = 1.0 {
        didSet { set("maximumPointSize", maximumPointSize) }
    }
    public var pointSizeScale: Float = 1.0 {
        didSet { set("pointSizeScale", pointSizeScale) }
    }
    public var applyDisplacement: Bool = false {
        didSet { set("applyDisplacement", applyDisplacement ? 1 : 0) }
    }
    public var applyTint: Bool = false {
        didSet { set("applyTint", applyTint ? 1 : 0) }
    }
    public var tintAlphaIsCoverage: Bool = false {
        didSet { set("tintAlphaIsCoverage", tintAlphaIsCoverage ? 1 : 0) }
    }

    public var lodPositionsBuffer: MTLBuffer? { didSet { set(lodPositionsBuffer, index: .Custom0) } }
    public var filesBuffer: MTLBuffer? { didSet { set(filesBuffer, index: .Custom1) } }
    /// Byte offset into ``filesBuffer`` for the current ring slot.
    public var filesBufferOffset: Int = 0
    public var pixelBuffer: MTLBuffer? { didSet { set(pixelBuffer, index: .Custom2) } }
    public var lodStatsBuffer: MTLBuffer? { didSet { set(lodStatsBuffer, index: .Custom3) } }
    /// Pack-order source index per LOD point (displacement/tint lookup key).
    public var lodSourceIndicesBuffer: MTLBuffer? { didSet { set(lodSourceIndicesBuffer, index: .Custom4) } }
    /// Per-point displacement (`float3`); must be bound (stand-in when unused).
    public var displacementBuffer: MTLBuffer? { didSet { set(displacementBuffer, index: .Custom5) } }
    /// Per-point tint (`float4`); must be bound (stand-in when unused).
    public var tintBuffer: MTLBuffer? { didSet { set(tintBuffer, index: .Custom6) } }

    /// Per-cloud indirect dispatch args (threadgroup count) written by the
    /// finalize pass; drives the point-count-driven dispatch.
    public var indirectArgsBuffer: MTLBuffer?
    public var indirectArgsBufferOffset: Int = 0

    open override func applyAdditionalBindings(_ computeEncoder: MTLComputeCommandEncoder) {
        if let filesBuffer {
            computeEncoder.setBuffer(filesBuffer, offset: filesBufferOffset, index: ComputeBufferIndex.Custom1.rawValue)
        }
    }

    open override func setup() {
        super.setup()
        set("screenSize", simd_int2(Int32(screenSize.x), Int32(screenSize.y)))
        set("viewMatrix", viewMatrix)
        set("projectionMatrix", projectionMatrix)
        set("pointSizeMode", Int(pointSizeMode.rawValue))
        set("minimumPointSize", minimumPointSize)
        set("maximumPointSize", maximumPointSize)
        set("pointSizeScale", pointSizeScale)
        set("applyDisplacement", applyDisplacement ? 1 : 0)
        set("applyTint", applyTint ? 1 : 0)
        set("tintAlphaIsCoverage", tintAlphaIsCoverage ? 1 : 0)
    }

    var isEncodeReady: Bool {
        screenSize.x > 0 && screenSize.y > 0
            && lodPositionsBuffer != nil
            && filesBuffer != nil
            && pixelBuffer != nil
            && lodStatsBuffer != nil
            && lodSourceIndicesBuffer != nil
            && displacementBuffer != nil
            && tintBuffer != nil
            && indirectArgsBuffer != nil
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: isEncodeReady)
    }

#if os(macOS) || os(iOS) || os(visionOS)
    open override func dispatchThreads(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        dispatchThreadgroups(computeEncoder: computeEncoder, pipeline: pipeline, iteration: iteration)
    }
#endif

    open override func dispatchThreadgroups(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        guard let indirectArgsBuffer else { return }
        computeEncoder.dispatchThreadgroups(
            indirectBuffer: indirectArgsBuffer,
            indirectBufferOffset: indirectArgsBufferOffset,
            threadsPerThreadgroup: MTLSize(width: pointRasteriserThreadsPerGroup, height: 1, depth: 1)
        )
    }
}
