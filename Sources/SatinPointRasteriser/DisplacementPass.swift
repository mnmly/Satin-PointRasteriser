import Combine
import Foundation
import Metal
import Satin

/// Drives an on-GPU displacement compute pass for the
/// ``PointRasteriserPointCloud``s of a ``PointRasteriser``. A single `encode`
/// with `cloud: nil` dispatches for **all** of the rasteriser's clouds; pass an
/// explicit `cloud:` to target one. Owns the pipeline build (with live reload),
/// auto-allocates each cloud's `displacementBuffer`, flips
/// ``PointRasteriserConfiguration/applyDisplacement`` `= true` on first encode,
/// and binds the cloud's batches + xyz + levels + displacement + colors buffers
/// each frame. The sketch only writes the kernel + any extra uniforms.
///
/// **Sketch contract** (identical to Satin-ComputeRasteriser's `DisplacementPass`).
/// The user `.metal` file is concatenated with the package's displacement
/// preamble before compile, which exposes:
///
///   * `RasterBatch` — mirrors the rasteriser's batch metadata.
///   * `scr_decodePointAt(pointIndex, batch, xyzLow, xyzMed, xyzHigh, levels)` —
///     dequantises a position from the 30/20/10-bit packed buffers.
///   * `scr_decodeColorAt(pointIndex, colors)` — the point's colour as 0..1 rgba.
///   * `scr_resolveDisplacementThread(id, info, batches, batch, pointIndex,
///     localOffset)` — turns a `thread_position_in_grid` into a `(batch,
///     pointIndex)` pair, returning `false` for non-resident slots.
///   * `SCR_DISPLACEMENT_KERNEL_BUFFERS` macro (auto-bound slots 0–7) + the
///     `SCR_DISP_BUF_*` slot constants; user uniforms bind at ``bufferUser0`` (8) on.
///
/// Write a **NaN** displacement to cull a point (both the depth and color passes
/// skip a point whose displaced position is NaN — a true removal). The output
/// buffer is `device float3 *displacements` (16 B stride), indexed by
/// **pack-order** point index — the same index the rasteriser's
/// `lodSourceIndices` map to.
///
/// - Note: Architecture difference from the sibling (contract preserved): this
///   package packs a cloud's points **contiguously** (no fixed-stride slot
///   pool), so the preamble's `scr_resolveDisplacementThread` binary-searches the
///   batch table by `firstPoint` instead of dividing by a slot stride. User
///   kernels are unaffected — the helper signature and all slot constants are
///   unchanged. Displacement is applied in the raster passes (looked up through
///   the LOD buffer's `lodSourceIndices`); LOD *selection* ignores it.
public final class DisplacementPass {
    // MARK: - Buffer slot constants (mirror `SCR_DISP_BUF_*` in the preamble).
    public static let bufferBatches: Int = 0
    public static let bufferXYZLow: Int = 1
    public static let bufferXYZMed: Int = 2
    public static let bufferXYZHigh: Int = 3
    public static let bufferLevels: Int = 4
    public static let bufferDisplacements: Int = 5
    public static let bufferInfo: Int = 6
    /// Native per-point colours (packed RGBA8), so a kernel can read each point's colour.
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
        kernelName: String = "computeDisplacement",
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
    /// `displacementBuffer`); pass an explicit `cloud:` to target a single one.
    /// Call before `rasteriser.encode(...)`/`draw(...)`.
    public func encode(commandBuffer: MTLCommandBuffer, cloud: PointRasteriserPointCloud? = nil) {
        guard let rasteriser, let pipeline else { return }
        let targets = cloud.map { [$0] } ?? rasteriser.pointClouds
        for target in targets {
            encode(for: target, rasteriser: rasteriser, pipeline: pipeline, commandBuffer: commandBuffer)
        }
    }

    /// Encode a single cloud's displacement dispatch. `InfoData` is bound
    /// per-dispatch via `setBytes` so two clouds sharing one command buffer can't
    /// race on a shared info buffer (each dispatch sees its own cloud's counts).
    private func encode(for target: PointRasteriserPointCloud, rasteriser: PointRasteriser, pipeline: MTLComputePipelineState, commandBuffer: MTLCommandBuffer) {
        guard let batchesBuffer = target.batchesBuffer,
              let xyzLow = target.xyzLowBuffer,
              let xyzMed = target.xyzMedBuffer,
              let xyzHigh = target.xyzHighBuffer,
              let levels = target.levelsBuffer,
              let colors = target.colorsBuffer,
              target.totalPoints > 0
        else { return }

        if target.displacementBuffer == nil {
            target.displacementBuffer = target.makeDisplacementBuffer(storage: .private, label: "DisplacementPass.Displacements")
        }
        guard let out = target.displacementBuffer else { return }

        if !rasteriser.configuration.applyDisplacement {
            rasteriser.configuration.applyDisplacement = true
        }

        var infoData = InfoData(pointsPerBatch: UInt32(target.pointsPerBatch), batchCount: UInt32(target.batchCount), totalPoints: UInt32(target.totalPoints))

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "DisplacementPass"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(batchesBuffer, offset: 0, index: Self.bufferBatches)
        encoder.setBuffer(xyzLow, offset: 0, index: Self.bufferXYZLow)
        encoder.setBuffer(xyzMed, offset: 0, index: Self.bufferXYZMed)
        encoder.setBuffer(xyzHigh, offset: 0, index: Self.bufferXYZHigh)
        encoder.setBuffer(levels, offset: 0, index: Self.bufferLevels)
        encoder.setBuffer(out, offset: 0, index: Self.bufferDisplacements)
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

    /// Disable the rasteriser's displacement reads without destroying the cloud.
    ///
    /// - Note: The `applyDisplacement` flag is **global** to the rasteriser, not
    ///   per-pass or per-cloud. Calling `disable()` on any `DisplacementPass` turns
    ///   off displacement application for every cloud the rasteriser draws.
    public func disable() {
        rasteriser?.configuration.applyDisplacement = false
    }

    private func rebuildPipeline() {
        guard let rasteriser else { return }
        do {
            let userSource = try compiler.parse(kernelURL)
            let source = ScrSketchPreamble.displacement + "\n" + userSource
            let library = try rasteriser.context.device.makeLibrary(source: source, options: nil)
            guard let function = library.makeFunction(name: kernelName) else {
                print("[DisplacementPass] kernel '\(kernelName)' not found in \(kernelURL.lastPathComponent)")
                return
            }
            pipeline = try rasteriser.context.device.makeComputePipelineState(function: function)
        } catch {
            print("[DisplacementPass] pipeline build failed: \(error)")
        }
    }
}
