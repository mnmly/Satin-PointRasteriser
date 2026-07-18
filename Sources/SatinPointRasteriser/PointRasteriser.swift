import Foundation
import Metal
import Satin
import simd

/// Compute point-cloud rasteriser implementing the Magnopus "LOD point cloud"
/// pipeline: LODSelect (full-sweep compaction) → finalize → clear → depth →
/// color → resolve → composite.
///
/// An `Object` subclass, so adding it to a Satin scene drives its per-frame
/// ``update(renderContext:camera:viewport:index:)`` + ``encode(_:)`` during
/// `RenderEncoder.draw`. Composite the resolved cloud over the scene with
/// ``draw(renderPassDescriptor:commandBuffer:)`` in a following pass.
///
/// Point clouds are children (``addPointCloud(_:)`` / ``removePointCloud(_:)``);
/// ``visiblePointClouds`` honors Object visibility. Each phase (LODSelect,
/// depth, color) runs across all visible clouds before the next, so occlusion
/// is independent of cloud order.
public final class PointRasteriser: Object, @unchecked Sendable {
    /// Runtime configuration; setting it re-pushes values to the pass processors.
    public var configuration: PointRasteriserConfiguration = .init() {
        didSet {
            // Selection-affecting changes invalidate any in-flight amortized
            // sweep — restart it so the next completed sweep reflects the change.
            if configuration.enableCLOD != oldValue.enableCLOD
                || configuration.lodBias != oldValue.lodBias
                || configuration.enableLODDither != oldValue.enableLODDither {
                restartLODSweep()
            }
            applyConfiguration()
        }
    }

    /// Abandon every cloud's in-flight LOD sweep and restart it from the first
    /// batch next frame. The front (last completed) sweep keeps rendering until
    /// the new one finishes. Call this on a camera teleport (fast cuts) so the
    /// stale front doesn't linger; selection-affecting config changes call it
    /// automatically.
    public func restartLODSweep() {
        for cloud in pointClouds { cloud.restartLODSweep() }
    }

    public nonisolated(unsafe) static var pipelinesURL: URL = {
        Bundle.module.resourceURL!.appendingPathComponent("Pipelines")
    }()

    /// Resolved RGBA color (`.rgba8Unorm`, private). Alpha 0 where no cloud lands.
    /// Final composited RGBA color (post reject/resolve + hole fill). This is
    /// the texture ``draw(renderPassDescriptor:commandBuffer:)`` samples.
    public private(set) var outputTexture: MTLTexture?
    /// Final per-pixel reverse-Z NDC depth (`.r32Float`, private; 0 = no cloud),
    /// consistent with ``outputTexture`` after hole filling.
    public private(set) var depthTexture: MTLTexture?
    private var pixelBuffer: MTLBuffer?

    // Resolve targets ("A") + hole-fill ping-pong scratch ("B"), for color and
    // depth. Resolve always writes the A textures; hole fill alternates A/B and
    // the result is published to `outputTexture`/`depthTexture`.
    private var resolveColorTexture: MTLTexture?
    private var resolveColorTextureB: MTLTexture?
    private var resolveDepthTexture: MTLTexture?
    private var resolveDepthTextureB: MTLTexture?

    /// The drawable (composite target) viewport, in pixels. Internal
    /// rasterisation buffers are sized to this × ``renderScale`` (see
    /// ``scaledPixelSize``); the composite resolves back down to this.
    private var viewport: SIMD4<Float> = .zero
    private var scaleFactor: Float = 1.0

    /// Supersampling factor for the internal point-rasterisation passes. The
    /// depth/color/resolve/hole-fill passes render at `viewport × renderScale`
    /// and the composite linearly resolves back down to the drawable — trading
    /// fill cost (≈`renderScale²` pixels + memory) for anti-aliased point
    /// edges. `1` (the default) renders at drawable resolution with no
    /// supersampling and is a no-op for every existing consumer. The
    /// composite's single bilinear tap resolves a 2× footprint exactly, so `2`
    /// is the quality sweet spot; higher factors keep costing `renderScale²`
    /// with diminishing resolve quality. Clamped to `≥ 1`.
    public var renderScale: Float = 1.0 {
        didSet {
            let clamped = max(1.0, renderScale)
            if clamped != renderScale { renderScale = clamped; return } // re-enters didSet, then falls through
            guard renderScale != oldValue else { return }
            resizeResources()
        }
    }

    /// Internal (supersampled) pixel dimensions = drawable viewport ×
    /// ``renderScale``, rounded. Both the resource buffers and the per-frame
    /// `screenSize` derive from this single source so they always agree.
    private var scaledPixelSize: (width: Int, height: Int) {
        let s = max(1.0, renderScale)
        return (max(0, Int((viewport.z * s).rounded())),
                max(0, Int((viewport.w * s).rounded())))
    }

    // Per-viewport resource cache (LRU cap 4), keyed by integer pixel size, so
    // alternating render sizes (e.g. live drawable vs. offline target) don't
    // reallocate the pixel buffer + textures every frame.
    private struct CachedResources {
        var pixelBuffer: MTLBuffer
        var colorTexture: MTLTexture
        var colorTextureB: MTLTexture
        var depthTexture: MTLTexture
        var depthTextureB: MTLTexture
        // Nearest-mode scratch: 64-bit winner (Apple9 path) + uint depth/index
        // (both paths converge here for the shared nearest reject+resolve).
        var nearestWinnerBuffer: MTLBuffer
        var nearestDepthBuffer: MTLBuffer
        var nearestIndexBuffer: MTLBuffer
    }
    private var resourceCache: [SIMD2<Int32>: CachedResources] = [:]
    private var resourceCacheLRU: [SIMD2<Int32>] = []
    private static let resourceCacheCap = 4

    private var nearestWinnerBuffer: MTLBuffer?
    private var nearestDepthBuffer: MTLBuffer?
    private var nearestIndexBuffer: MTLBuffer?

    /// GPU capability probe, gating the Apple9+ 64-bit nearest fast path.
    public let capabilities: RasteriserCapabilities

    private lazy var lodSelectProcessor = LODSelectProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var lodFinalizeProcessor = LODDispatchFinalizeProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var clearProcessor = ClearProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var depthProcessor = DepthPassProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var colorProcessor = ColorPassProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var resolveProcessor = ResolveProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var holeFillProcessor = HoleFillProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)

    // Nearest-point mode.
    private lazy var nearestWinner64Processor = NearestWinner64Processor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var nearestSplitProcessor = NearestSplitProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var nearestDepthProcessor = NearestDepthProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var nearestIndexProcessor = NearestIndexProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var nearestResolveProcessor = NearestResolveProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)

    // Tracks the SIMD-aggregation define currently compiled into depth/color, so
    // toggling `configuration.enableSimdAggregation` only recompiles on change.
    private var simdAggregationCompiled: Bool?

    private lazy var postMaterial: SourceMaterial = {
        let material = SourceMaterial(
            context: context,
            pipelineURL: Self.pipelinesURL.appendingPathComponent("PointRasteriserPost.metal"),
            live: true
        )
        material.label = "PointRasteriserPost"
        material.lighting = false
        material.depthWriteEnabled = false
        material.depthCompareFunction = .always
        material.blending = .alpha
        return material
    }()

    private lazy var postProcessor = PostProcessEncoder(
        label: "PointRasteriserPostProcessor",
        context: context,
        material: postMaterial,
        colorLoadAction: .load,
        depthLoadAction: .load
    )

    private lazy var postDepthMaterial: SourceMaterial = {
        let material = SourceMaterial(
            context: context,
            pipelineURL: Self.pipelinesURL.appendingPathComponent("PointRasteriserPost.metal"),
            live: true
        )
        material.label = "PointRasteriserPostDepth"
        material.lighting = false
        material.depthWriteEnabled = true
        material.depthCompareFunction = .greaterEqual
        material.blending = .alpha
        return material
    }()

    private lazy var postDepthProcessor = PostProcessEncoder(
        label: "PointRasteriserPostDepthProcessor",
        context: context,
        material: postDepthMaterial,
        colorLoadAction: .load,
        depthLoadAction: .load
    )

    /// - Parameter use64BitAtomics: Overrides the Apple9 capability probe for the
    ///   nearest-point fast path (pass `false` to force the portable two-pass
    ///   fallback, e.g. for testing or on drivers that misreport the feature).
    public init(context: Context, label: String = "PointRasteriser", use64BitAtomics: Bool? = nil) {
        capabilities = RasteriserCapabilities(device: context.device, use64BitAtomics: use64BitAtomics)
        super.init(context: context, label: label)
    }

    public required init(from decoder: any Decoder) throws {
        fatalError("init(from:) has not been implemented")
    }

    @discardableResult
    public func addPointCloud(_ cloud: PointRasteriserPointCloud) -> PointRasteriserPointCloud {
        add(cloud)
        return cloud
    }

    public func removePointCloud(_ cloud: PointRasteriserPointCloud) {
        remove(cloud)
    }

    /// All point clouds in the subtree (not just direct children).
    public var pointClouds: [PointRasteriserPointCloud] {
        var out: [PointRasteriserPointCloud] = []
        func collect(_ object: Object) {
            for child in object.children {
                if let cloud = child as? PointRasteriserPointCloud { out.append(cloud) }
                collect(child)
            }
        }
        collect(self)
        return out
    }

    /// Subtree clouds whose own `visible` and every ancestor up to here is visible.
    public var visiblePointClouds: [PointRasteriserPointCloud] {
        var out: [PointRasteriserPointCloud] = []
        func collect(_ object: Object, ancestorsVisible: Bool) {
            for child in object.children {
                let chainVisible = ancestorsVisible && child.visible
                if let cloud = child as? PointRasteriserPointCloud, chainVisible {
                    out.append(cloud)
                }
                collect(child, ancestorsVisible: chainVisible)
            }
        }
        collect(self, ancestorsVisible: true)
        return out
    }

    public func resize(size: (width: Float, height: Float), scaleFactor: Float = 1.0) {
        let nextViewport = SIMD4<Float>(0, 0, size.width, size.height)
        guard nextViewport != viewport || scaleFactor != self.scaleFactor else { return }
        viewport = nextViewport
        self.scaleFactor = scaleFactor
        resizeResources()
    }

    public override func setup() {
        super.setup()
        applyConfiguration()
        if viewport.z > 0, viewport.w > 0 {
            resizeResources()
        }
    }

    public override func update(renderContext: Context, camera: Camera, viewport: simd_float4, index: Int) {
        guard visible else { return }
        if self.viewport.z != viewport.z || self.viewport.w != viewport.w {
            self.viewport = viewport
            resizeResources()
        }

        let viewProjection = camera.projectionMatrix * camera.viewMatrix
        // Rasterisation runs at the supersampled internal resolution.
        let (scaledWidth, scaledHeight) = scaledPixelSize
        let screenSize = SIMD2<UInt32>(UInt32(scaledWidth), UInt32(scaledHeight))
        // Point footprints are specified in pixels: the min/max clamps are pixel
        // sizes, so they scale with `renderScale` to hold the on-screen
        // footprint constant. `pointSizeScale` is pixel-like in screen-space
        // mode (scale it) but a world-space radius otherwise (leave it — the
        // already-scaled `screenSize` carries the resolution through).
        let s = max(1.0, renderScale)
        let scaledMinPointSize = configuration.minimumPointSize * s
        let scaledMaxPointSize = configuration.maximumPointSize * s
        let scaledPointSizeScale = configuration.pointSizeMode == .screenSpace
            ? configuration.pointSizeScale * s
            : configuration.pointSizeScale

        // Stash the selection-affecting camera state so `encode` can build the
        // full-sweep LODSelect skip key (see ``LODSelectKey``).
        selectionViewMatrix = camera.viewMatrix
        selectionProjectionMatrix = camera.projectionMatrix
        selectionScreenSize = screenSize

        lodSelectProcessor.screenSize = screenSize
        lodSelectProcessor.viewMatrix = camera.viewMatrix
        lodSelectProcessor.projectionMatrix = camera.projectionMatrix
        lodSelectProcessor.enableFrustumCulling = configuration.enableFrustumCulling
        lodSelectProcessor.lodBias = configuration.lodBias
        lodSelectProcessor.enableCLOD = configuration.enableCLOD
        lodSelectProcessor.lodDither = configuration.enableLODDither

        depthProcessor.screenSize = screenSize
        depthProcessor.viewMatrix = camera.viewMatrix
        depthProcessor.projectionMatrix = camera.projectionMatrix
        depthProcessor.pointSizeMode = configuration.pointSizeMode
        depthProcessor.minimumPointSize = scaledMinPointSize
        depthProcessor.maximumPointSize = scaledMaxPointSize
        depthProcessor.pointSizeScale = scaledPointSizeScale

        colorProcessor.screenSize = screenSize
        colorProcessor.viewMatrix = camera.viewMatrix
        colorProcessor.projectionMatrix = camera.projectionMatrix
        colorProcessor.pointSizeMode = configuration.pointSizeMode
        colorProcessor.minimumPointSize = scaledMinPointSize
        colorProcessor.maximumPointSize = scaledMaxPointSize
        colorProcessor.pointSizeScale = scaledPointSizeScale
        colorProcessor.depthTolerance = configuration.depthTolerance
        colorProcessor.colorizeChunks = configuration.colorizeChunks
        colorProcessor.colorizeOverdraw = configuration.colorizeOverdraw
        colorProcessor.antialiasEdges = configuration.pointEdgeAntialiasing

        // Per-point feature flags (identical on depth + color so occlusion agrees).
        let motionBlurOn = configuration.motionBlur > 0
        let coverage = (configuration.applyTint && configuration.tintAlphaIsCoverage) || motionBlurOn
        depthProcessor.applyDisplacement = configuration.applyDisplacement
        depthProcessor.applyTint = configuration.applyTint
        depthProcessor.tintAlphaIsCoverage = configuration.tintAlphaIsCoverage
        colorProcessor.applyDisplacement = configuration.applyDisplacement
        colorProcessor.applyTint = configuration.applyTint
        colorProcessor.tintAlphaIsCoverage = configuration.tintAlphaIsCoverage
        colorProcessor.motionBlur = configuration.motionBlur
        colorProcessor.motionBlurSamples = configuration.motionBlurSamples
        colorProcessor.motionBlurMaxSpread = configuration.motionBlurMaxSpread

        let invProjection = camera.projectionMatrix.inverse
        // projectionMatrix[3][3] == 1 for orthographic, 0 for perspective.
        let isOrthographic = camera.projectionMatrix[3][3] == 1.0
        resolveProcessor.invProjectionMatrix = invProjection
        resolveProcessor.enablePointRejection = configuration.enablePointRejection
        resolveProcessor.rejectionConeThreshold = configuration.rejectionConeThreshold
        resolveProcessor.depthTolerance = configuration.depthTolerance
        resolveProcessor.isOrthographic = isOrthographic
        resolveProcessor.coverageEnabled = coverage
        resolveProcessor.edgeAntialias = configuration.pointEdgeAntialiasing

        // Nearest-mode processors share the projection / point-size uniforms.
        for processor in [nearestWinner64Processor, nearestDepthProcessor, nearestIndexProcessor] as [NearestRasterProcessor] {
            processor.screenSize = screenSize
            processor.viewMatrix = camera.viewMatrix
            processor.projectionMatrix = camera.projectionMatrix
            processor.pointSizeMode = configuration.pointSizeMode
            processor.minimumPointSize = scaledMinPointSize
            processor.maximumPointSize = scaledMaxPointSize
            processor.pointSizeScale = scaledPointSizeScale
        }
        nearestResolveProcessor.invProjectionMatrix = invProjection
        nearestResolveProcessor.enablePointRejection = configuration.enablePointRejection
        nearestResolveProcessor.rejectionConeThreshold = configuration.rejectionConeThreshold
        nearestResolveProcessor.depthTolerance = configuration.depthTolerance
        nearestResolveProcessor.isOrthographic = isOrthographic

        applySimdAggregationDefine()

        // Previous-frame view-projection for motion-blur velocity, keyed per
        // camera so stereo stays correct (each eye its own prev state).
        let camKey = ObjectIdentifier(camera)
        currentCameraKey = camKey
        let prevVP = previousViewProjection[camKey] ?? viewProjection
        for cloud in visiblePointClouds {
            cloud.updateFiles(viewProjection: viewProjection, modelMatrix: cloud.worldMatrix, prevViewProjection: prevVP)
        }
        previousViewProjection[camKey] = viewProjection
    }

    /// Last frame's `projection · view` per camera (stereo → one per eye), for
    /// motion-blur camera velocity.
    private var previousViewProjection: [ObjectIdentifier: simd_float4x4] = [:]
    /// Camera key from the most recent `update`, to pick the matching per-eye
    /// previous-displacement buffer in `encode`.
    private var currentCameraKey: ObjectIdentifier?

    // Selection-affecting camera state stashed by `update`, consumed by `encode`
    // to build the full-sweep LODSelect skip key.
    private var selectionViewMatrix = matrix_identity_float4x4
    private var selectionProjectionMatrix = matrix_identity_float4x4
    private var selectionScreenSize = SIMD2<UInt32>.zero

    /// Everything the LODSelect kernel reads that decides which points survive
    /// (frustum + precision level + CLOD threshold + dither). In full-sweep mode
    /// (`lodPointsPerFrame == 0`), when a cloud's key is unchanged from its last
    /// completed select — and it already has front data — the select + finalize
    /// dispatches are skipped and the existing front LOD set is reused. The test
    /// is exact (literal `==`, float-equal matrices): any difference re-runs.
    private struct LODSelectKey: Equatable {
        var viewMatrix: simd_float4x4
        var projectionMatrix: simd_float4x4
        var worldMatrix: simd_float4x4        // cloud transform → file.transform·world
        var screenSize: SIMD2<UInt32>
        var enableFrustumCulling: Bool
        var lodBias: Int
        var enableCLOD: Bool
        var enableLODDither: Bool
        var contentGeneration: UInt64
    }

    /// Per (cloud, camera) last-completed-select key. Keyed by both so a stereo
    /// or multi-camera setup keeps a key per eye rather than thrashing one.
    private struct SelectMapKey: Hashable { let cloud: ObjectIdentifier; let camera: ObjectIdentifier }
    private var lastLODSelectKeys: [SelectMapKey: LODSelectKey] = [:]

    /// Number of clouds whose LODSelect was **skipped** last `encode` (front set
    /// reused because nothing selection-affecting changed). Full-sweep mode only.
    public private(set) var lodSelectSkippedLastFrame = 0
    /// Number of clouds whose LODSelect actually **ran** last `encode`.
    public private(set) var lodSelectRanLastFrame = 0

    private func lodSelectKey(for cloud: PointRasteriserPointCloud) -> LODSelectKey {
        LODSelectKey(
            viewMatrix: selectionViewMatrix,
            projectionMatrix: selectionProjectionMatrix,
            worldMatrix: cloud.worldMatrix,
            screenSize: selectionScreenSize,
            enableFrustumCulling: configuration.enableFrustumCulling,
            lodBias: configuration.lodBias,
            enableCLOD: configuration.enableCLOD,
            enableLODDither: configuration.enableLODDither,
            contentGeneration: cloud.contentGeneration
        )
    }

    /// (Re)compile the depth/color kernels with or without the
    /// `PR_SIMD_AGGREGATION` define to match ``PointRasteriserConfiguration/enableSimdAggregation``.
    /// Only recompiles when the flag actually changes.
    private func applySimdAggregationDefine() {
        let want = configuration.enableSimdAggregation
        guard simdAggregationCompiled != want else { return }
        simdAggregationCompiled = want
        let defines: [ShaderDefine] = want
            ? [ShaderDefine(key: "PR_SIMD_AGGREGATION", value: NSString(string: "1"))]
            : []
        depthProcessor.defines = defines
        colorProcessor.defines = defines
    }

    public override func encode(_ commandBuffer: MTLCommandBuffer) {
        guard visible,
              let pixelBuffer,
              outputTexture != nil,
              Int(viewport.z) > 0,
              Int(viewport.w) > 0
        else { return }

        let allClouds = visiblePointClouds.filter { $0.batchCount > 0 }
        // Nearest mode is single-cloud (winner index alone can't identify a
        // cloud); it operates on the first visible cloud.
        let clouds = configuration.renderMode == .nearestPoint ? Array(allClouds.prefix(1)) : allClouds

        // Warn once per cloud if a completed sweep overflowed its LOD capacity
        // (reads last frame's drained stats — cheap, no GPU stall).
        for cloud in allClouds { cloud.logOverflowWarningIfNeeded() }

        // Front end (both modes): reset stats → LODSelect → finalize dispatch args.
        encodeLODCompaction(commandBuffer, clouds: clouds)

        // Mode-specific rasterization + reject/resolve into the A textures.
        switch configuration.renderMode {
        case .highQualityAverage:
            encodeHighQualityAverage(commandBuffer, pixelBuffer: pixelBuffer, clouds: clouds)
        case .nearestPoint:
            encodeNearestPoint(commandBuffer, cloud: clouds.first)
        }

        // Hole fill (both modes); publish the result as the output/depth textures.
        let (color, depth) = encodeHoleFill(commandBuffer)
        outputTexture = color
        depthTexture = depth

        // Snapshot this frame's displacement into each cloud's per-camera prev
        // buffer so next frame's color pass can build displacement velocity.
        if configuration.motionBlur > 0 {
            snapshotDisplacementForMotionBlur(commandBuffer, clouds: clouds)
        }
    }

    /// Blit each cloud's `displacementBuffer` into its per-camera prev buffer
    /// (lazily allocated), for next frame's motion-blur velocity.
    private func snapshotDisplacementForMotionBlur(_ commandBuffer: MTLCommandBuffer, clouds: [PointRasteriserPointCloud]) {
        guard let key = currentCameraKey, let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.label = "\(label).MotionBlurDisplacementSnapshot"
        for cloud in clouds {
            guard let src = cloud.displacementBuffer else { continue }
            if cloud.prevDisplacementBuffers[key]?.length != src.length {
                cloud.prevDisplacementBuffers[key] = cloud.makeDisplacementBuffer(storage: .private, label: "\(cloud.label).PrevDisplacement")
            }
            guard let dst = cloud.prevDisplacementBuffers[key] else { continue }
            blit.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: src.length)
        }
        blit.endEncoding()
    }

    /// Advance each cloud's LOD sweep by one frame: reset the back stats at a
    /// sweep start, compact this frame's batch chunk into the back set, finalize
    /// + swap on completion. The raster passes downstream read the **front**
    /// set, so a partial sweep is never rasterized: on non-completing frames the
    /// front (last completed sweep) is untouched; on the completing frame the
    /// swap happens here (CPU-side, between the finalize and raster encoders) so
    /// the raster reads the freshly finalized back set, ordered by the encoder
    /// boundary. See ``PointRasteriserPointCloud`` for the sweep lifecycle.
    private func encodeLODCompaction(_ commandBuffer: MTLCommandBuffer, clouds: [PointRasteriserPointCloud]) {
        let budget = configuration.lodPointsPerFrame
        let fullSweep = budget == 0
        if budget > 0 {
            for cloud in clouds { cloud.ensureDoubleBuffered() }
        }

        // Full-sweep skip: reuse a cloud's front LOD set when nothing
        // selection-affecting changed since its last completed select (exact
        // key match + it already has front data). Amortized mode always runs
        // (it has its own per-frame cadence). Skipped clouds are excluded from
        // every LOD encoder and not advanced, so their front stays intact for
        // the raster passes.
        let cameraKey = currentCameraKey ?? ObjectIdentifier(self)
        var selectClouds: [PointRasteriserPointCloud] = []
        var skipped = 0
        if fullSweep {
            selectClouds.reserveCapacity(clouds.count)
            for cloud in clouds {
                let mapKey = SelectMapKey(cloud: ObjectIdentifier(cloud), camera: cameraKey)
                if cloud.hasFrontData, lastLODSelectKeys[mapKey] == lodSelectKey(for: cloud) {
                    skipped += 1
                } else {
                    selectClouds.append(cloud)
                }
            }
        } else {
            selectClouds = clouds
        }
        lodSelectSkippedLastFrame = skipped
        lodSelectRanLastFrame = selectClouds.count
        guard !selectClouds.isEmpty else { return }

        // Plan each running cloud's chunk once; drive every encoder + the swap.
        let chunks = selectClouds.map { $0.planLODChunk(pointBudget: budget) }

        // Reset back stats only at a sweep start (append accumulates otherwise).
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "\(label).LODReset"
            for (cloud, chunk) in zip(selectClouds, chunks) where chunk.startsSweep {
                guard let stats = cloud.backLodStatsBuffer else { continue }
                blit.fill(buffer: stats, range: 0 ..< stats.length, value: 0)
            }
            blit.endEncoding()
        }

        // LODSelect this frame's batch chunk into the back set.
        if let encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent) {
            encoder.label = "\(label).LODSelect"
            lodSelectProcessor.update()
            for (cloud, chunk) in zip(selectClouds, chunks) where chunk.batchCount > 0 {
                bindLODSelect(cloud, chunk: chunk)
                lodSelectProcessor.encode(into: encoder, isReady: lodSelectProcessor.isEncodeReady)
            }
            encoder.endEncoding()
        }

        // Finalize (clamp count + dispatch args) only for sweeps completing now.
        if let encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent) {
            encoder.label = "\(label).LODFinalize"
            lodFinalizeProcessor.update()
            for (cloud, chunk) in zip(selectClouds, chunks) where chunk.completesSweep {
                lodFinalizeProcessor.lodStatsBuffer = cloud.backLodStatsBuffer
                lodFinalizeProcessor.dispatchArgsBuffer = cloud.backLodDispatchArgsBuffer
                lodFinalizeProcessor.lodCapacity = UInt32(cloud.lodCapacity)
                lodFinalizeProcessor.encode(into: encoder, isReady: lodFinalizeProcessor.isEncodeReady)
            }
            encoder.endEncoding()
        }

        // Advance cursors + swap completed sweeps to front. Must happen after the
        // finalize encoder is recorded and before the raster passes bind `front`.
        for (cloud, chunk) in zip(selectClouds, chunks) {
            cloud.advanceSweep(after: chunk)
        }

        // Remember each full-sweep cloud's completed-select key so a subsequent
        // unchanged frame skips. (Amortized mode leaves the map alone.)
        if fullSweep {
            for cloud in selectClouds {
                lastLODSelectKeys[SelectMapKey(cloud: ObjectIdentifier(cloud), camera: cameraKey)] = lodSelectKey(for: cloud)
            }
        }
    }

    private func encodeHighQualityAverage(_ commandBuffer: MTLCommandBuffer, pixelBuffer: MTLBuffer, clouds: [PointRasteriserPointCloud]) {
        clearProcessor.pixelBuffer = pixelBuffer
        clearProcessor.pixelCount = scaledPixelSize.width * scaledPixelSize.height
        clearProcessor.update(commandBuffer)

        if let encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent) {
            encoder.label = "\(label).Depth"
            depthProcessor.update()
            for cloud in clouds {
                bindDepth(cloud, pixelBuffer: pixelBuffer)
                depthProcessor.encode(into: encoder, isReady: depthProcessor.isEncodeReady)
            }
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent) {
            encoder.label = "\(label).Color"
            colorProcessor.update()
            for cloud in clouds {
                bindColor(cloud, pixelBuffer: pixelBuffer)
                colorProcessor.encode(into: encoder, isReady: colorProcessor.isEncodeReady)
            }
            encoder.endEncoding()
        }

        resolveProcessor.pixelBuffer = pixelBuffer
        resolveProcessor.outputTexture = resolveColorTexture
        resolveProcessor.depthTexture = resolveDepthTexture
        resolveProcessor.update(commandBuffer)
    }

    /// Nearest-point rasterization (single cloud): the Apple9 64-bit fast path
    /// or the portable two-pass fallback, both converging on the shared uint
    /// depth/index buffers, then the nearest reject+resolve.
    private func encodeNearestPoint(_ commandBuffer: MTLCommandBuffer, cloud: PointRasteriserPointCloud?) {
        guard let cloud, encodeNearestWinner(commandBuffer, cloud: cloud),
              let depths = nearestDepthBuffer,
              let indices = nearestIndexBuffer
        else { return }

        nearestResolveProcessor.depthsBuffer = depths
        nearestResolveProcessor.indicesBuffer = indices
        nearestResolveProcessor.lodColorsBuffer = cloud.frontLodColorsBuffer
        nearestResolveProcessor.outputTexture = resolveColorTexture
        nearestResolveProcessor.depthTexture = resolveDepthTexture
        nearestResolveProcessor.update(commandBuffer)
    }

    /// Fill `nearestDepthBuffer` + `nearestIndexBuffer` for `cloud` from its front
    /// LOD set: per pixel the nearest LOD point's depth + winning **LOD index**
    /// (0xffffffff where no point lands). The Apple9 64-bit winner + split, or the
    /// portable two-pass fallback. No resolve — shared by nearest render + pick.
    @discardableResult
    private func encodeNearestWinner(_ commandBuffer: MTLCommandBuffer, cloud: PointRasteriserPointCloud) -> Bool {
        guard let depths = nearestDepthBuffer,
              let indices = nearestIndexBuffer,
              let lodPositions = cloud.frontLodPositionsBuffer,
              let files = cloud.filesBuffer,
              let stats = cloud.frontLodStatsBuffer,
              let args = cloud.frontLodDispatchArgsBuffer
        else { return false }

        let pixelCount = scaledPixelSize.width * scaledPixelSize.height

        if capabilities.use64BitAtomics, let winner = nearestWinnerBuffer {
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.label = "\(label).NearestReset"
                blit.fill(buffer: winner, range: 0 ..< winner.length, value: 0)
                blit.endEncoding()
            }
            nearestWinner64Processor.lodPositionsBuffer = lodPositions
            nearestWinner64Processor.filesBuffer = files
            nearestWinner64Processor.filesBufferOffset = cloud.filesBufferOffset
            nearestWinner64Processor.winnerBuffer = winner
            nearestWinner64Processor.lodStatsBuffer = stats
            nearestWinner64Processor.indirectArgsBuffer = args
            nearestWinner64Processor.update(commandBuffer)

            nearestSplitProcessor.pixelCount = pixelCount
            nearestSplitProcessor.winnerBuffer = winner
            nearestSplitProcessor.depthsBuffer = depths
            nearestSplitProcessor.indicesBuffer = indices
            nearestSplitProcessor.update(commandBuffer)
        } else {
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.label = "\(label).NearestReset"
                blit.fill(buffer: depths, range: 0 ..< depths.length, value: 0)
                blit.fill(buffer: indices, range: 0 ..< indices.length, value: 0xff)
                blit.endEncoding()
            }
            nearestDepthProcessor.lodPositionsBuffer = lodPositions
            nearestDepthProcessor.filesBuffer = files
            nearestDepthProcessor.filesBufferOffset = cloud.filesBufferOffset
            nearestDepthProcessor.depthsBuffer = depths
            nearestDepthProcessor.lodStatsBuffer = stats
            nearestDepthProcessor.indirectArgsBuffer = args
            nearestDepthProcessor.update(commandBuffer)

            nearestIndexProcessor.lodPositionsBuffer = lodPositions
            nearestIndexProcessor.filesBuffer = files
            nearestIndexProcessor.filesBufferOffset = cloud.filesBufferOffset
            nearestIndexProcessor.depthsBuffer = depths
            nearestIndexProcessor.indicesBuffer = indices
            nearestIndexProcessor.lodStatsBuffer = stats
            nearestIndexProcessor.indirectArgsBuffer = args
            nearestIndexProcessor.update(commandBuffer)
        }
        return true
    }

    // MARK: - Picking

    /// Pick the front-most point of `cloud` under a viewport location, returning
    /// the **pack-order** source index — the same index the ``DisplacementPass`` /
    /// ``TintPass`` buffers use, and into a ``PackedPointCloud``'s
    /// `orderedPositions` / `sourceIndices`.
    ///
    /// Runs the nearest-winner pass for `cloud` off-screen on its own command
    /// buffer, **waits**, reads back the winning LOD index in a small band around
    /// the cursor, and maps it through the cloud's `lodSourceIndices`. Nearest
    /// mode uses undisplaced LOD positions (mirrors the sibling), so picking
    /// ignores displacement. Call between frames (e.g. from a click handler), not
    /// inside the render loop — it shares the rasteriser's nearest buffers.
    ///
    /// - Parameters:
    ///   - ndc: pick location in normalized device coords, x/y in `[-1, 1]` with
    ///     **y up**.
    ///   - cloud: the point cloud to test (one of ``pointClouds``).
    ///   - camera: the camera the cloud is rendered through.
    ///   - searchRadius: pixel radius of the band searched around the cursor
    ///     (point clouds are sparse; the exact pixel is often empty).
    /// - Returns: the pack-order source index under `ndc`, or `nil` if off-cloud.
    public func pickPointIndex(atNDC ndc: SIMD2<Float>, in cloud: PointRasteriserPointCloud, camera: Camera, searchRadius: Int = 10) -> UInt32? {
        // The nearest-index buffer is at the supersampled resolution, so map and
        // index in that space (NDC is resolution-independent).
        let (width, height) = scaledPixelSize
        guard width > 0, height > 0, nearestIndexBuffer != nil, cloud.frontLodPositionsBuffer != nil else { return nil }
        let s = max(1.0, renderScale)

        // NDC (y-up) → buffer pixel (the winner pass flips Y).
        let px = min(max(Int((ndc.x * 0.5 + 0.5) * Float(width)), 0), width - 1)
        let pyFromBottom = Int((ndc.y * 0.5 + 0.5) * Float(height))
        let py = min(max(height - 1 - pyFromBottom, 0), height - 1)

        // Configure the nearest processors for this camera (subset of update()).
        let screenSize = SIMD2<UInt32>(UInt32(width), UInt32(height))
        for processor in [nearestWinner64Processor, nearestDepthProcessor, nearestIndexProcessor] as [NearestRasterProcessor] {
            processor.screenSize = screenSize
            processor.viewMatrix = camera.viewMatrix
            processor.projectionMatrix = camera.projectionMatrix
            processor.pointSizeMode = configuration.pointSizeMode
            processor.minimumPointSize = configuration.minimumPointSize * s
            processor.maximumPointSize = configuration.maximumPointSize * s
            processor.pointSizeScale = configuration.pointSizeMode == .screenSpace
                ? configuration.pointSizeScale * s
                : configuration.pointSizeScale
        }
        cloud.updateFiles(viewProjection: camera.projectionMatrix * camera.viewMatrix, modelMatrix: cloud.worldMatrix)

        let stride = MemoryLayout<UInt32>.stride
        // Search radius is a screen-pixel budget; widen it into the supersampled
        // buffer so the picked region matches on-screen regardless of renderScale.
        let r = Int((Float(max(0, searchRadius)) * s).rounded())
        let rowStart = max(0, py - r), rowEnd = min(height - 1, py + r)
        let rowCount = rowEnd - rowStart + 1
        let bandLength = rowCount * width * stride

        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { return nil }
        commandBuffer.label = "\(label).Pick"
        guard encodeNearestWinner(commandBuffer, cloud: cloud),
              let indices = nearestIndexBuffer,
              let staging = context.device.makeBuffer(length: bandLength, options: .storageModeShared),
              let blit = commandBuffer.makeBlitCommandEncoder()
        else { return nil }
        blit.label = "\(label).PickReadback"
        blit.copy(from: indices, sourceOffset: rowStart * width * stride, to: staging, destinationOffset: 0, size: bandLength)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Nearest non-sentinel LOD index to the cursor within the circular window.
        let buf = staging.contents().bindMemory(to: UInt32.self, capacity: rowCount * width)
        var bestLodIndex: UInt32?
        var bestDist = Int.max
        let r2 = r * r
        for y in rowStart ... rowEnd {
            let dy = y - py
            let rowBase = (y - rowStart) * width
            for x in max(0, px - r) ... min(width - 1, px + r) {
                let dx = x - px
                let dist = dx * dx + dy * dy
                if dist > r2 || dist >= bestDist { continue }
                let v = buf[rowBase + x]
                if v != UInt32.max { bestDist = dist; bestLodIndex = v }
            }
        }
        guard let lodIndex = bestLodIndex else { return nil }
        return cloud.frontLodSourceIndex(at: Int(lodIndex))
    }

    /// Run `holeFillIterations` neighbor-average passes, ping-ponging color and
    /// depth between the A/B textures. Returns the textures holding the final
    /// result (the A textures unchanged when hole fill is disabled).
    private func encodeHoleFill(_ commandBuffer: MTLCommandBuffer) -> (MTLTexture?, MTLTexture?) {
        let iterations = max(0, configuration.holeFillIterations)
        guard iterations > 0,
              let colorA = resolveColorTexture, let colorB = resolveColorTextureB,
              let depthA = resolveDepthTexture, let depthB = resolveDepthTextureB
        else {
            return (resolveColorTexture, resolveDepthTexture)
        }

        holeFillProcessor.width = scaledPixelSize.width
        holeFillProcessor.height = scaledPixelSize.height

        var srcColor = colorA, dstColor = colorB
        var srcDepth = depthA, dstDepth = depthB
        for _ in 0 ..< iterations {
            holeFillProcessor.inputColorTexture = srcColor
            holeFillProcessor.inputDepthTexture = srcDepth
            holeFillProcessor.outputColorTexture = dstColor
            holeFillProcessor.outputDepthTexture = dstDepth
            holeFillProcessor.update(commandBuffer)
            swap(&srcColor, &dstColor)
            swap(&srcDepth, &dstDepth)
        }
        // After the final swap, src holds the most recent output.
        return (srcColor, srcDepth)
    }

    /// Composite the resolved cloud over the render pass. Depth-aware when
    /// ``PointRasteriserConfiguration/writesSceneDepth`` is set and a depth
    /// texture exists; else an always-on-top overlay.
    public func draw(renderPassDescriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer) {
        guard let outputTexture else { return }
        if configuration.writesSceneDepth, let depthTexture {
            postDepthMaterial.set(outputTexture, index: FragmentTextureIndex.Custom1)
            postDepthMaterial.set(depthTexture, index: FragmentTextureIndex.Custom2)
            postDepthProcessor.draw(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer)
        } else {
            postMaterial.set(outputTexture, index: FragmentTextureIndex.Custom1)
            postProcessor.draw(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer)
        }
    }

    /// Composite onto a specific sub-region of the render target — e.g. one eye's
    /// half viewport for stereo offline rendering, or a sub-rect for a
    /// reference-plane / offline export. Forwards to Satin's
    /// `PostProcessEncoder.draw(viewports:)`, which honors the supplied viewport
    /// instead of the post-processor's internal renderer viewport. Same
    /// depth-aware vs. always-on-top branching as ``draw(renderPassDescriptor:commandBuffer:)``.
    public func draw(
        renderPassDescriptor: MTLRenderPassDescriptor,
        commandBuffer: MTLCommandBuffer,
        viewport: MTLViewport
    ) {
        guard let outputTexture else { return }
        if configuration.writesSceneDepth, let depthTexture {
            postDepthMaterial.set(outputTexture, index: FragmentTextureIndex.Custom1)
            postDepthMaterial.set(depthTexture, index: FragmentTextureIndex.Custom2)
            postDepthProcessor.draw(
                renderPassDescriptor: renderPassDescriptor,
                commandBuffer: commandBuffer,
                viewports: [viewport]
            )
        } else {
            postMaterial.set(outputTexture, index: FragmentTextureIndex.Custom1)
            postProcessor.draw(
                renderPassDescriptor: renderPassDescriptor,
                commandBuffer: commandBuffer,
                viewports: [viewport]
            )
        }
    }

    // MARK: - Bindings

    // LODSelect appends into the BACK set (the in-flight sweep).
    private func bindLODSelect(_ cloud: PointRasteriserPointCloud, chunk: PointRasteriserPointCloud.LODChunk) {
        lodSelectProcessor.batchCount = chunk.batchCount
        lodSelectProcessor.firstBatch = UInt32(chunk.firstBatch)
        lodSelectProcessor.lodCapacity = UInt32(cloud.lodCapacity)
        lodSelectProcessor.batchesBuffer = cloud.batchesBuffer
        lodSelectProcessor.xyzLowBuffer = cloud.xyzLowBuffer
        lodSelectProcessor.xyzMedBuffer = cloud.xyzMedBuffer
        lodSelectProcessor.xyzHighBuffer = cloud.xyzHighBuffer
        lodSelectProcessor.filesBuffer = cloud.filesBuffer
        lodSelectProcessor.filesBufferOffset = cloud.filesBufferOffset
        lodSelectProcessor.colorsBuffer = cloud.colorsBuffer
        lodSelectProcessor.levelsBuffer = cloud.levelsBuffer
        lodSelectProcessor.lodPositionsBuffer = cloud.backLodPositionsBuffer
        lodSelectProcessor.lodColorsBuffer = cloud.backLodColorsBuffer
        lodSelectProcessor.lodSourceIndicesBuffer = cloud.backLodSourceIndicesBuffer
        lodSelectProcessor.lodStatsBuffer = cloud.backLodStatsBuffer
    }

    // Raster passes read the FRONT set (the last completed sweep). Displacement /
    // tint / prev-displacement bind to the cloud's own buffers or a shared zeroed
    // stand-in (the shaders gate reads on the feature flags, so a stand-in that is
    // never read still satisfies Metal's "argument bound" requirement, and when a
    // flag is on without a real buffer it reads zeros = no effect).
    private func bindDepth(_ cloud: PointRasteriserPointCloud, pixelBuffer: MTLBuffer) {
        let standIn = zeroStandIn(forPoints: cloud.totalPoints)
        depthProcessor.lodPositionsBuffer = cloud.frontLodPositionsBuffer
        depthProcessor.filesBuffer = cloud.filesBuffer
        depthProcessor.filesBufferOffset = cloud.filesBufferOffset
        depthProcessor.pixelBuffer = pixelBuffer
        depthProcessor.lodStatsBuffer = cloud.frontLodStatsBuffer
        depthProcessor.lodSourceIndicesBuffer = cloud.frontLodSourceIndicesBuffer
        depthProcessor.displacementBuffer = cloud.displacementBuffer ?? standIn
        depthProcessor.tintBuffer = cloud.tintBuffer ?? standIn
        depthProcessor.indirectArgsBuffer = cloud.frontLodDispatchArgsBuffer
    }

    private func bindColor(_ cloud: PointRasteriserPointCloud, pixelBuffer: MTLBuffer) {
        let standIn = zeroStandIn(forPoints: cloud.totalPoints)
        colorProcessor.lodPositionsBuffer = cloud.frontLodPositionsBuffer
        colorProcessor.lodColorsBuffer = cloud.frontLodColorsBuffer
        colorProcessor.filesBuffer = cloud.filesBuffer
        colorProcessor.filesBufferOffset = cloud.filesBufferOffset
        colorProcessor.pixelBuffer = pixelBuffer
        colorProcessor.lodStatsBuffer = cloud.frontLodStatsBuffer
        colorProcessor.lodSourceIndicesBuffer = cloud.frontLodSourceIndicesBuffer
        colorProcessor.displacementBuffer = cloud.displacementBuffer ?? standIn
        colorProcessor.tintBuffer = cloud.tintBuffer ?? standIn
        // Previous-frame displacement, keyed per eye (nil → stand in the current
        // one, i.e. zero displacement velocity this frame).
        let prev = currentCameraKey.flatMap { cloud.prevDisplacementBuffers[$0] } ?? cloud.displacementBuffer ?? standIn
        colorProcessor.prevDisplacementBuffer = prev
        colorProcessor.indirectArgsBuffer = cloud.frontLodDispatchArgsBuffer
    }

    /// A shared zeroed buffer covering `points` (16 B/point — fits float3 and
    /// float4), allocated once and grown as larger clouds appear. Shared storage
    /// so it is zeroed on the CPU (no extra command buffer); read as zeros by the
    /// GPU on Apple's unified memory with no penalty.
    private var zeroStandInBuffer: MTLBuffer?
    private var zeroStandInPoints = 0
    private func zeroStandIn(forPoints points: Int) -> MTLBuffer? {
        if let buf = zeroStandInBuffer, zeroStandInPoints >= points { return buf }
        let stride = MemoryLayout<SIMD4<Float>>.stride
        let length = max(points, 1) * stride
        guard let buf = context.device.makeBuffer(length: length, options: .storageModeShared) else { return zeroStandInBuffer }
        buf.label = "\(label).ZeroStandIn"
        memset(buf.contents(), 0, length)
        zeroStandInBuffer = buf
        zeroStandInPoints = max(points, 1)
        return buf
    }

    // MARK: - Configuration / resources

    private func applyConfiguration() {
        resolveProcessor.backgroundColor = configuration.backgroundColor
        lodSelectProcessor.enableFrustumCulling = configuration.enableFrustumCulling
        lodSelectProcessor.lodBias = configuration.lodBias
        lodSelectProcessor.enableCLOD = configuration.enableCLOD
        lodSelectProcessor.lodDither = configuration.enableLODDither
        depthProcessor.pointSizeMode = configuration.pointSizeMode
        depthProcessor.minimumPointSize = configuration.minimumPointSize
        depthProcessor.maximumPointSize = configuration.maximumPointSize
        depthProcessor.pointSizeScale = configuration.pointSizeScale
        colorProcessor.pointSizeMode = configuration.pointSizeMode
        colorProcessor.minimumPointSize = configuration.minimumPointSize
        colorProcessor.maximumPointSize = configuration.maximumPointSize
        colorProcessor.pointSizeScale = configuration.pointSizeScale
        colorProcessor.depthTolerance = configuration.depthTolerance
        colorProcessor.colorizeChunks = configuration.colorizeChunks
        colorProcessor.colorizeOverdraw = configuration.colorizeOverdraw
        colorProcessor.antialiasEdges = configuration.pointEdgeAntialiasing
        nearestResolveProcessor.backgroundColor = configuration.backgroundColor
        applySimdAggregationDefine()
    }

    private func resizeResources() {
        // Internal buffers render at the supersampled resolution…
        let (width, height) = scaledPixelSize
        let pixelCount = width * height
        guard width > 0, height > 0, pixelCount > 0 else { return }

        let key = SIMD2<Int32>(Int32(width), Int32(height))
        let resources = resourceCache[key] ?? allocateAndCache(width: width, height: height, key: key)

        if let idx = resourceCacheLRU.firstIndex(of: key) { resourceCacheLRU.remove(at: idx) }
        resourceCacheLRU.append(key)

        pixelBuffer = resources.pixelBuffer
        resolveColorTexture = resources.colorTexture
        resolveColorTextureB = resources.colorTextureB
        resolveDepthTexture = resources.depthTexture
        resolveDepthTextureB = resources.depthTextureB
        nearestWinnerBuffer = resources.nearestWinnerBuffer
        nearestDepthBuffer = resources.nearestDepthBuffer
        nearestIndexBuffer = resources.nearestIndexBuffer
        // Default published result is the resolve target; hole fill overrides it
        // per frame in `encode`.
        outputTexture = resources.colorTexture
        depthTexture = resources.depthTexture

        clearProcessor.pixelCount = pixelCount
        resolveProcessor.width = width
        resolveProcessor.height = height
        resolveProcessor.backgroundColor = configuration.backgroundColor
        nearestResolveProcessor.width = width
        nearestResolveProcessor.height = height
        nearestResolveProcessor.backgroundColor = configuration.backgroundColor
        holeFillProcessor.width = width
        holeFillProcessor.height = height
        // …but the composite resolves back down to the drawable viewport (it
        // samples the supersampled `outputTexture` with normalized coords, so
        // the larger source is bilinearly downsampled = supersampling).
        postProcessor.resize(size: (viewport.z, viewport.w), scaleFactor: scaleFactor)
        postDepthProcessor.resize(size: (viewport.z, viewport.w), scaleFactor: scaleFactor)
    }

    private func allocateAndCache(width: Int, height: Int, key: SIMD2<Int32>) -> CachedResources {
        let pixelCount = width * height
        let pixel = context.device.makeBuffer(
            length: pixelCount * MemoryLayout<RasterPixel>.stride,
            options: .storageModePrivate
        )!
        pixel.label = "\(label).Pixels[\(width)x\(height)]"
        let colorA = makeOutputTexture(width: width, height: height, label: "\(label).ColorA[\(width)x\(height)]")!
        let colorB = makeOutputTexture(width: width, height: height, label: "\(label).ColorB[\(width)x\(height)]")!
        let depthA = makeDepthTexture(width: width, height: height, label: "\(label).DepthA[\(width)x\(height)]")!
        let depthB = makeDepthTexture(width: width, height: height, label: "\(label).DepthB[\(width)x\(height)]")!
        let nearestWinner = context.device.makeBuffer(length: pixelCount * MemoryLayout<UInt64>.stride, options: .storageModePrivate)!
        nearestWinner.label = "\(label).NearestWinner[\(width)x\(height)]"
        let nearestDepth = context.device.makeBuffer(length: pixelCount * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        nearestDepth.label = "\(label).NearestDepth[\(width)x\(height)]"
        let nearestIndex = context.device.makeBuffer(length: pixelCount * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        nearestIndex.label = "\(label).NearestIndex[\(width)x\(height)]"

        let resources = CachedResources(
            pixelBuffer: pixel,
            colorTexture: colorA, colorTextureB: colorB,
            depthTexture: depthA, depthTextureB: depthB,
            nearestWinnerBuffer: nearestWinner,
            nearestDepthBuffer: nearestDepth,
            nearestIndexBuffer: nearestIndex
        )
        resourceCache[key] = resources
        while resourceCacheLRU.count >= Self.resourceCacheCap {
            let evict = resourceCacheLRU.removeFirst()
            resourceCache.removeValue(forKey: evict)
        }
        return resources
    }

    private func makeOutputTexture(width: Int, height: Int, label: String) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        descriptor.storageMode = .private
        let texture = context.device.makeTexture(descriptor: descriptor)
        texture?.label = label
        return texture
    }

    private func makeDepthTexture(width: Int, height: Int, label: String) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: width, height: height, mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        let texture = context.device.makeTexture(descriptor: descriptor)
        texture?.label = label
        return texture
    }
}
