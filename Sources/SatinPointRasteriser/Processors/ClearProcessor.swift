import Metal
import Satin

/// Zeroes the screen-sized `RasterPixel` accumulation buffer. See `Clear/Shaders.metal`.
open class ClearProcessor: BasePointRasteriserProcessor {
    public var pixelCount: Int = 0 {
        didSet { set("pixelCount", pixelCount) }
    }

    public var pixelBuffer: MTLBuffer? {
        didSet { set(pixelBuffer, index: .Custom0) }
    }

    open override func setup() {
        super.setup()
        set("pixelCount", pixelCount)
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: pixelCount > 0 && pixelBuffer != nil)
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
