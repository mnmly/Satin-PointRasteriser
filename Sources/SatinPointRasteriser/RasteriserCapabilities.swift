import Foundation
import Metal

/// GPU capabilities probed once from an `MTLDevice`, gating fast paths that
/// need Apple9+ hardware (e.g. 64-bit buffer atomics, MSL 3.1).
public struct RasteriserCapabilities: Sendable {
    /// URL of the `Pipelines` resource bundle copied into `SatinPointRasteriser`,
    /// used to compile kernels (including the probe) at runtime, mirroring the
    /// sibling `Satin-ComputeRasteriser` package's `MetalFileCompiler`/`Bundle.module` pattern.
    public nonisolated(unsafe) static var pipelinesURL: URL = {
        Bundle.module.resourceURL!.appendingPathComponent("Pipelines")
    }()

    /// `true` when the device reports `MTLGPUFamily.apple9` support (M5-class
    /// Apple GPUs and later).
    public let supportsApple9: Bool
    /// Whether the rasteriser should use the 64-bit packed depth+index atomic
    /// fast path. Defaults to ``supportsApple9`` but can be overridden (e.g.
    /// forced off after ``verify64BitAtomics(device:)`` fails empirically).
    public let use64BitAtomics: Bool
    /// Human-readable summary of the highest matched Apple GPU family, for logs/UI.
    public let gpuFamilyDescription: String

    /// Probes `device` directly: ``supportsApple9`` from `device.supportsFamily(.apple9)`,
    /// ``use64BitAtomics`` defaulted to ``supportsApple9`` unless `use64BitAtomics` is passed explicitly.
    public init(device: MTLDevice, use64BitAtomics: Bool? = nil) {
        let supportsApple9 = device.supportsFamily(.apple9)
        self.supportsApple9 = supportsApple9
        self.use64BitAtomics = use64BitAtomics ?? supportsApple9
        gpuFamilyDescription = Self.familyDescription(device: device)
    }

    private static func familyDescription(device: MTLDevice) -> String {
        let candidatesHighestFirst: [(MTLGPUFamily, String)] = [
            (.apple9, "Apple9"),
            (.apple8, "Apple8"),
            (.apple7, "Apple7"),
            (.apple6, "Apple6"),
            (.apple5, "Apple5"),
            (.apple4, "Apple4"),
            (.apple3, "Apple3"),
            (.mac2, "Mac2"),
        ]
        // Families are cumulative (an Apple9 device also reports Apple8/.../Mac2),
        // so the first match scanning highest-to-lowest is the true tier.
        return candidatesHighestFirst.first { device.supportsFamily($0.0) }?.1 ?? "Unknown"
    }

    /// Compiles and dispatches the probe kernel (`probeAtomicULongMax` in
    /// `Pipelines/Probe/Shaders.metal`) against a handful of known values and
    /// checks the resulting atomic max, rather than trusting `supportsFamily`
    /// alone — a runtime probe catches drivers/hardware that reject 64-bit
    /// buffer atomics despite reporting the feature. Never throws: any
    /// compile, allocation, or dispatch failure is reported as `false`.
    public static func verify64BitAtomics(device: MTLDevice) -> Bool {
        let inputs: [UInt64] = [3, 42, 17, 4_294_967_296, 9, 4_294_967_295]
        let expected = inputs.max() ?? 0

        do {
            let url = pipelinesURL
                .appendingPathComponent("Probe")
                .appendingPathComponent("Shaders.metal")
            let source = try String(contentsOf: url, encoding: .utf8)

            let options = MTLCompileOptions()
            options.languageVersion = .version3_1
            let library = try device.makeLibrary(source: source, options: options)
            guard let function = library.makeFunction(name: "probeAtomicULongMax") else {
                return false
            }
            let pipeline = try device.makeComputePipelineState(function: function)

            guard
                let resultBuffer = device.makeBuffer(length: MemoryLayout<UInt64>.stride, options: .storageModeShared),
                let inputsBuffer = device.makeBuffer(bytes: inputs, length: MemoryLayout<UInt64>.stride * inputs.count, options: .storageModeShared),
                let commandQueue = device.makeCommandQueue(),
                let commandBuffer = commandQueue.makeCommandBuffer(),
                let encoder = commandBuffer.makeComputeCommandEncoder()
            else {
                return false
            }

            resultBuffer.contents().storeBytes(of: UInt64(0), as: UInt64.self)
            var count = UInt32(inputs.count)

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(resultBuffer, offset: 0, index: 0)
            encoder.setBuffer(inputsBuffer, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)

            let width = min(inputs.count, pipeline.maxTotalThreadsPerThreadgroup)
            let threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
            let threadgroups = MTLSize(width: (inputs.count + width - 1) / width, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if commandBuffer.error != nil {
                return false
            }

            let observed = resultBuffer.contents().load(as: UInt64.self)
            return observed == expected
        } catch {
            return false
        }
    }
}
