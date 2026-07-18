import Metal
import Satin
import simd

/// Resolves the `RasterPixel` accumulation buffer into the output color +
/// reverse-Z depth textures (one thread per pixel). See `Resolve/Shaders.metal`.
open class ResolveProcessor: BasePointRasteriserProcessor {
    public var width: Int = 0 {
        didSet { set("screenSize", simd_int2(Int32(width), Int32(height))) }
    }
    public var height: Int = 0 {
        didSet { set("screenSize", simd_int2(Int32(width), Int32(height))) }
    }
    public var backgroundColor: SIMD4<Float> = .zero {
        didSet { set("backgroundColor", backgroundColor) }
    }
    /// Inverse projection matrix, for reconstructing view-space positions from
    /// the pixel buffer's reverse-Z depths in the rejection operator.
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
    /// Resolve the weighted-blended-OIT accumulation (translucent defocus /
    /// motion blur) instead of the count-average.
    public var coverageEnabled: Bool = false {
        didSet { set("coverageEnabled", coverageEnabled ? 1 : 0) }
    }

    public var pixelBuffer: MTLBuffer? { didSet { set(pixelBuffer, index: .Custom0) } }
    public var outputTexture: MTLTexture? { didSet { set(outputTexture, index: .Custom0) } }
    /// Per-pixel reverse-Z NDC depth (R32Float; 0 = no cloud).
    public var depthTexture: MTLTexture? { didSet { set(depthTexture, index: .Custom1) } }

    open override func setup() {
        super.setup()
        set("screenSize", simd_int2(Int32(width), Int32(height)))
        set("backgroundColor", backgroundColor)
        set("invProjectionMatrix", invProjectionMatrix)
        set("enablePointRejection", enablePointRejection ? 1 : 0)
        set("rejectionConeThreshold", rejectionConeThreshold)
        set("isOrthographic", isOrthographic ? 1 : 0)
        set("coverageEnabled", coverageEnabled ? 1 : 0)
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: width > 0 && height > 0 && pixelBuffer != nil && outputTexture != nil && depthTexture != nil)
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
        let tw = 16
        let th = 16
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: (width + tw - 1) / tw, height: (height + th - 1) / th, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1)
        )
    }
}
