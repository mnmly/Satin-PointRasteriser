import Metal
import Satin
import simd

/// Neighbor-average hole fill over color + depth, one dispatch per iteration.
/// The caller ping-pongs the input/output textures between iterations. See
/// `HoleFill/Shaders.metal`.
open class HoleFillProcessor: BasePointRasteriserProcessor {
    public var width: Int = 0 {
        didSet { set("screenSize", simd_int2(Int32(width), Int32(height))) }
    }
    public var height: Int = 0 {
        didSet { set("screenSize", simd_int2(Int32(width), Int32(height))) }
    }

    public var inputColorTexture: MTLTexture? { didSet { set(inputColorTexture, index: .Custom0) } }
    public var inputDepthTexture: MTLTexture? { didSet { set(inputDepthTexture, index: .Custom1) } }
    public var outputColorTexture: MTLTexture? { didSet { set(outputColorTexture, index: .Custom2) } }
    public var outputDepthTexture: MTLTexture? { didSet { set(outputDepthTexture, index: .Custom3) } }

    open override func setup() {
        super.setup()
        set("screenSize", simd_int2(Int32(width), Int32(height)))
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(
            commandBuffer,
            isReady: width > 0 && height > 0
                && inputColorTexture != nil && inputDepthTexture != nil
                && outputColorTexture != nil && outputDepthTexture != nil
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
        let tw = 16
        let th = 16
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: (width + tw - 1) / tw, height: (height + th - 1) / th, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1)
        )
    }
}
