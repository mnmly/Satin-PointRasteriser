import Metal
import Satin

/// Single-thread pass: reads the LOD survivor count, clamps it to the cloud's
/// capacity, and writes the indirect dispatch args for the depth/color passes.
/// See `LODDispatchFinalize/Shaders.metal`.
open class LODDispatchFinalizeProcessor: BasePointRasteriserProcessor {
    public var lodStatsBuffer: MTLBuffer? { didSet { set(lodStatsBuffer, index: .Custom0) } }
    public var dispatchArgsBuffer: MTLBuffer? { didSet { set(dispatchArgsBuffer, index: .Custom1) } }

    /// Per-cloud LOD capacity, bound via `setBytes` at Custom2.
    public var lodCapacity: UInt32 = 0

    open override func applyAdditionalBindings(_ computeEncoder: MTLComputeCommandEncoder) {
        var capacity = lodCapacity
        computeEncoder.setBytes(&capacity, length: MemoryLayout<UInt32>.stride, index: ComputeBufferIndex.Custom2.rawValue)
    }

    var isEncodeReady: Bool {
        lodStatsBuffer != nil && dispatchArgsBuffer != nil
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: isEncodeReady)
    }

#if os(macOS) || os(iOS) || os(visionOS)
    open override func dispatchThreads(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        computeEncoder.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
    }
#endif

    open override func dispatchThreadgroups(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
    }
}
