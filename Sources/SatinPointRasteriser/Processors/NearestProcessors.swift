import Metal
import Satin
import simd

/// Shared projection/point-size uniforms + files-ring binding for the
/// nearest-mode rasterization passes (winner64 / depth / index).
open class NearestRasterProcessor: BasePointRasteriserProcessor {
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

    public var lodPositionsBuffer: MTLBuffer? { didSet { set(lodPositionsBuffer, index: .Custom0) } }
    public var filesBuffer: MTLBuffer? { didSet { set(filesBuffer, index: .Custom1) } }
    public var filesBufferOffset: Int = 0
    public var indirectArgsBuffer: MTLBuffer?

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
            indirectBufferOffset: 0,
            threadsPerThreadgroup: MTLSize(width: pointRasteriserThreadsPerGroup, height: 1, depth: 1)
        )
    }
}

/// Apple9+ single-pass 64-bit winner selection. See `NearestWinner64/Shaders.metal`.
open class NearestWinner64Processor: NearestRasterProcessor {
    /// 64-bit packed depth|index winner buffer (`atomic_ulong` per pixel).
    public var winnerBuffer: MTLBuffer? { didSet { set(winnerBuffer, index: .Custom2) } }
    public var lodStatsBuffer: MTLBuffer? { didSet { set(lodStatsBuffer, index: .Custom3) } }

    var isEncodeReady: Bool {
        screenSize.x > 0 && lodPositionsBuffer != nil && filesBuffer != nil
            && winnerBuffer != nil && lodStatsBuffer != nil && indirectArgsBuffer != nil
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: isEncodeReady)
    }
}

/// Portable fallback pass 1: 32-bit nearest depth. See `NearestDepth/Shaders.metal`.
open class NearestDepthProcessor: NearestRasterProcessor {
    public var depthsBuffer: MTLBuffer? { didSet { set(depthsBuffer, index: .Custom2) } }
    public var lodStatsBuffer: MTLBuffer? { didSet { set(lodStatsBuffer, index: .Custom3) } }

    var isEncodeReady: Bool {
        screenSize.x > 0 && lodPositionsBuffer != nil && filesBuffer != nil
            && depthsBuffer != nil && lodStatsBuffer != nil && indirectArgsBuffer != nil
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: isEncodeReady)
    }
}

/// Portable fallback pass 2: winning lodIndex. See `NearestIndex/Shaders.metal`.
open class NearestIndexProcessor: NearestRasterProcessor {
    public var depthsBuffer: MTLBuffer? { didSet { set(depthsBuffer, index: .Custom2) } }
    public var indicesBuffer: MTLBuffer? { didSet { set(indicesBuffer, index: .Custom3) } }
    public var lodStatsBuffer: MTLBuffer? { didSet { set(lodStatsBuffer, index: .Custom4) } }

    var isEncodeReady: Bool {
        screenSize.x > 0 && lodPositionsBuffer != nil && filesBuffer != nil
            && depthsBuffer != nil && indicesBuffer != nil && lodStatsBuffer != nil && indirectArgsBuffer != nil
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: isEncodeReady)
    }
}

/// Unpack the 64-bit winner buffer into uint depth + index buffers. One thread
/// per pixel. See `NearestSplit/Shaders.metal`.
open class NearestSplitProcessor: BasePointRasteriserProcessor {
    public var pixelCount: Int = 0 {
        didSet { set("pixelCount", pixelCount) }
    }
    public var winnerBuffer: MTLBuffer? { didSet { set(winnerBuffer, index: .Custom0) } }
    public var depthsBuffer: MTLBuffer? { didSet { set(depthsBuffer, index: .Custom1) } }
    public var indicesBuffer: MTLBuffer? { didSet { set(indicesBuffer, index: .Custom2) } }

    open override func setup() {
        super.setup()
        set("pixelCount", pixelCount)
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: pixelCount > 0 && winnerBuffer != nil && depthsBuffer != nil && indicesBuffer != nil)
    }

#if os(macOS) || os(iOS) || os(visionOS)
    open override func dispatchThreads(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        computeEncoder.dispatchThreads(
            MTLSize(width: pixelCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        )
    }
#endif

    open override func dispatchThreadgroups(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        let threads = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: (pixelCount + threads - 1) / threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }
}

/// Nearest reject + resolve (one thread per pixel). See `NearestResolve/Shaders.metal`.
open class NearestResolveProcessor: BasePointRasteriserProcessor {
    public var width: Int = 0 {
        didSet { set("screenSize", simd_int2(Int32(width), Int32(height))) }
    }
    public var height: Int = 0 {
        didSet { set("screenSize", simd_int2(Int32(width), Int32(height))) }
    }
    public var backgroundColor: SIMD4<Float> = .zero {
        didSet { set("backgroundColor", backgroundColor) }
    }
    public var invProjectionMatrix: simd_float4x4 = matrix_identity_float4x4 {
        didSet { set("invProjectionMatrix", invProjectionMatrix) }
    }
    public var enablePointRejection: Bool = true {
        didSet { set("enablePointRejection", enablePointRejection ? 1 : 0) }
    }
    public var rejectionConeThreshold: Float = 0.5 {
        didSet { set("rejectionConeThreshold", rejectionConeThreshold) }
    }
    public var isOrthographic: Bool = false {
        didSet { set("isOrthographic", isOrthographic ? 1 : 0) }
    }
    /// Relative reverse-Z band within which a nearer neighbor is treated as the
    /// same splatted surface (not an occluder) in the rejection cone test, so
    /// multi-pixel splats don't self-reject into black rims.
    public var depthTolerance: Float = 0.01 {
        didSet { set("depthTolerance", depthTolerance) }
    }

    public var depthsBuffer: MTLBuffer? { didSet { set(depthsBuffer, index: .Custom0) } }
    public var indicesBuffer: MTLBuffer? { didSet { set(indicesBuffer, index: .Custom1) } }
    public var lodColorsBuffer: MTLBuffer? { didSet { set(lodColorsBuffer, index: .Custom2) } }
    public var outputTexture: MTLTexture? { didSet { set(outputTexture, index: .Custom0) } }
    public var depthTexture: MTLTexture? { didSet { set(depthTexture, index: .Custom1) } }

    open override func setup() {
        super.setup()
        set("screenSize", simd_int2(Int32(width), Int32(height)))
        set("backgroundColor", backgroundColor)
        set("invProjectionMatrix", invProjectionMatrix)
        set("enablePointRejection", enablePointRejection ? 1 : 0)
        set("rejectionConeThreshold", rejectionConeThreshold)
        set("isOrthographic", isOrthographic ? 1 : 0)
        set("depthTolerance", depthTolerance)
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(
            commandBuffer,
            isReady: width > 0 && height > 0
                && depthsBuffer != nil && indicesBuffer != nil && lodColorsBuffer != nil
                && outputTexture != nil && depthTexture != nil
        )
    }

#if os(macOS) || os(iOS) || os(visionOS)
    open override func dispatchThreads(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        let tw = max(pipeline.threadExecutionWidth, 1)
        let th = max(1, pipeline.maxTotalThreadsPerThreadgroup / tw)
        computeEncoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(tw, width), height: min(th, height), depth: 1)
        )
    }
#endif

    open override func dispatchThreadgroups(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        let tw = 16, th = 16
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: (width + tw - 1) / tw, height: (height + th - 1) / th, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1)
        )
    }
}
