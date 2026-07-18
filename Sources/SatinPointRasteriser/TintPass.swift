import Combine
import Foundation
import Metal
import Satin

/// Drives an on-GPU color-tint compute pass for the
/// ``PointRasteriserPointCloud``s of a ``PointRasteriser``. A single `encode`
/// with `cloud: nil` dispatches for **all** of the rasteriser's clouds; pass an
/// explicit `cloud:` to target one. Mirror of ``DisplacementPass``: owns the
/// pipeline build (with live reload), auto-allocates each cloud's `tintBuffer`,
/// flips
/// ``PointRasteriserConfiguration/applyTint`` `= true` on first encode, and binds
/// the cloud's batches + xyz + levels + tint + colors buffers each frame.
///
/// **Sketch contract** (identical to Satin-ComputeRasteriser's `TintPass`). The
/// preamble exposes `RasterBatch`, `scr_decodePointAt`, `scr_decodeColorAt`,
/// `scr_resolveTintThread`, the `SCR_TINT_BUF_*` slot constants, and the
/// `SCR_TINT_KERNEL_BUFFERS` macro (auto-bound slots 0–7; user uniforms at
/// ``bufferUser0`` (8) on). The output buffer is `device float4 *tints`
/// (pack-order index). The color pass composes `final = mix(stored.rgb,
/// tint.rgb, tint.a)`, so `tint.a == 0` is a pass-through, `tint.a == 1` a full
/// replacement, and a **negative** `tint.a` a discard sentinel (the point
/// contributes nothing → a pixel covered only by such points is transparent).
///
/// - Note: Nearest-point render mode ignores tint (mirrors the sibling); tint is
///   applied only in `.highQualityAverage` mode. See ``DisplacementPass`` for the
///   contiguous-packing note about the internal thread-resolve.
public final class TintPass {
    // MARK: - Buffer slot constants (mirror `SCR_TINT_BUF_*` in the preamble).
    public static let bufferBatches: Int = 0
    public static let bufferXYZLow: Int = 1
    public static let bufferXYZMed: Int = 2
    public static let bufferXYZHigh: Int = 3
    public static let bufferLevels: Int = 4
    public static let bufferTints: Int = 5
    public static let bufferInfo: Int = 6
    public static let bufferColors: Int = 7
    public static let bufferUser0: Int = 8
    public static let bufferUser1: Int = 9
    public static let bufferUser2: Int = 10
    public static let bufferUser3: Int = 11
    public static let bufferUser4: Int = 12
    public static let bufferUser5: Int = 13
    public static let bufferUser6: Int = 14
    public static let bufferUser7: Int = 15

    /// Bind extra uniforms / buffers each frame, after the pass's own bindings.
    public var bindUserBuffers: ((MTLComputeCommandEncoder) -> Void)?
    /// Fired on main after a live-reload rebuilds the pipeline.
    public var onReloaded: (() -> Void)?

    /// Opt this tint into **translucent defocus** (weighted-blended OIT). The
    /// kernel writes a circle-of-confusion into `tints[i].a`; the rasteriser then
    /// treats defocused points as translucent (skip depth write, accumulate
    /// coverage-weighted) instead of a colour mix. Flips
    /// ``PointRasteriserConfiguration/tintAlphaIsCoverage`` on encode.
    public var alphaIsCoverage: Bool = false

    private weak var rasteriser: PointRasteriser?
    private let kernelURL: URL
    private let kernelName: String
    private let compiler: MetalFileCompiler
    private var pipeline: MTLComputePipelineState?
    private var cancellable: AnyCancellable?

    private struct InfoData {
        var pointsPerBatch: UInt32
        var batchCount: UInt32 = 0
        var totalPoints: UInt32 = 0
        var reserved: UInt32 = 0
    }

    public init(
        rasteriser: PointRasteriser,
        kernelURL: URL,
        kernelName: String = "computeTint",
        live: Bool = true
    ) {
        self.rasteriser = rasteriser
        self.kernelURL = kernelURL
        self.kernelName = kernelName
        self.compiler = MetalFileCompiler(watch: live)
        rebuildPipeline()
        if live {
            cancellable = compiler.onUpdatePublisher
                .receive(on: DispatchQueue.main)
                .sink { [weak self] in self?.rebuildPipeline(); self?.onReloaded?() }
        }
    }

    /// Per-frame entry point. When `cloud` is `nil` this encodes a dispatch for
    /// **every** cloud on the rasteriser (each gets its own auto-allocated
    /// `tintBuffer`); pass an explicit `cloud:` to target a single one. Call
    /// before `rasteriser.encode(...)`.
    public func encode(commandBuffer: MTLCommandBuffer, cloud: PointRasteriserPointCloud? = nil) {
        guard let rasteriser, let pipeline else { return }
        if rasteriser.configuration.tintAlphaIsCoverage != alphaIsCoverage {
            rasteriser.configuration.tintAlphaIsCoverage = alphaIsCoverage
        }
        let targets = cloud.map { [$0] } ?? rasteriser.pointClouds
        for target in targets {
            encode(for: target, rasteriser: rasteriser, pipeline: pipeline, commandBuffer: commandBuffer)
        }
    }

    /// Encode a single cloud's tint dispatch. `InfoData` is bound per-dispatch via
    /// `setBytes` so two clouds sharing one command buffer can't race on a shared
    /// info buffer (each dispatch sees its own cloud's counts).
    private func encode(for target: PointRasteriserPointCloud, rasteriser: PointRasteriser, pipeline: MTLComputePipelineState, commandBuffer: MTLCommandBuffer) {
        guard let batchesBuffer = target.batchesBuffer,
              let xyzLow = target.xyzLowBuffer,
              let xyzMed = target.xyzMedBuffer,
              let xyzHigh = target.xyzHighBuffer,
              let levels = target.levelsBuffer,
              let colors = target.colorsBuffer,
              target.totalPoints > 0
        else { return }

        if target.tintBuffer == nil {
            target.tintBuffer = target.makeTintBuffer(storage: .private, label: "TintPass.Tints")
        }
        guard let out = target.tintBuffer else { return }

        if !rasteriser.configuration.applyTint {
            rasteriser.configuration.applyTint = true
        }

        var infoData = InfoData(pointsPerBatch: UInt32(target.pointsPerBatch), batchCount: UInt32(target.batchCount), totalPoints: UInt32(target.totalPoints))

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "TintPass"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(batchesBuffer, offset: 0, index: Self.bufferBatches)
        encoder.setBuffer(xyzLow, offset: 0, index: Self.bufferXYZLow)
        encoder.setBuffer(xyzMed, offset: 0, index: Self.bufferXYZMed)
        encoder.setBuffer(xyzHigh, offset: 0, index: Self.bufferXYZHigh)
        encoder.setBuffer(levels, offset: 0, index: Self.bufferLevels)
        encoder.setBuffer(out, offset: 0, index: Self.bufferTints)
        encoder.setBytes(&infoData, length: MemoryLayout<InfoData>.size, index: Self.bufferInfo)
        encoder.setBuffer(colors, offset: 0, index: Self.bufferColors)
        bindUserBuffers?(encoder)

        let tew = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: target.totalPoints, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tew, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Disable the color pass's tint reads without destroying the cloud.
    ///
    /// - Note: The `applyTint` flag is **global** to the rasteriser, not per-pass
    ///   or per-cloud. Calling `disable()` on any `TintPass` turns off tint
    ///   application for every cloud the rasteriser draws.
    public func disable() {
        rasteriser?.configuration.applyTint = false
    }

    private func rebuildPipeline() {
        guard let rasteriser else { return }
        do {
            let userSource = try compiler.parse(kernelURL)
            let source = ScrSketchPreamble.tint + "\n" + userSource
            let library = try rasteriser.context.device.makeLibrary(source: source, options: nil)
            guard let function = library.makeFunction(name: kernelName) else {
                print("[TintPass] kernel '\(kernelName)' not found in \(kernelURL.lastPathComponent)")
                return
            }
            pipeline = try rasteriser.context.device.makeComputePipelineState(function: function)
        } catch {
            print("[TintPass] pipeline build failed: \(error)")
        }
    }
}
