#if os(macOS)
import AppKit
import Metal
import Satin
import SatinPointRasteriser
import simd
#if canImport(SwiftPDAL)
import SatinPointRasteriserStreaming
import SwiftPDAL
#endif

/// `ViewRenderer` driving a ``PointRasteriser`` over a fixture cube grid (or a
/// loaded PLY), modeled on Satin-ComputeRasteriser's app renderer. Clears to
/// dark gray, composites the resolved cloud on top, and orbits with a
/// `PerspectiveCameraController`.
///
/// Configuration is **pulled**, not pushed: every frame's `update()` rebuilds
/// a `PointRasteriserConfiguration` from ``appState`` and assigns it to
/// `rasteriser.configuration`, so the settings UI (bound directly to
/// ``appState``) needs no per-field setter plumbing — see
/// ``PointRasteriserExampleState``.
final class PointRasteriserExampleRenderer: ViewRenderer, @unchecked Sendable {
    private lazy var renderer = RenderEncoder(context: defaultContext)
    let rasteriser: PointRasteriser
    private(set) var pointCloud: PointRasteriserPointCloud
    private lazy var scene = Object(context: defaultContext, label: "PointRasteriser Example", [rasteriser])

    private lazy var camera = PerspectiveCamera(
        context: defaultContext,
        position: [0, 0, 2.4],
        near: 0.01,
        far: 100,
        fov: 45
    )
    private var cameraController: PerspectiveCameraController?
    private var currentViewport: SIMD2<Float> = SIMD2(800, 600)

    /// Settings-UI state; mirrored into `rasteriser.configuration` every frame.
    let appState = PointRasteriserExampleState()

    /// Every cloud from the most recent PLY import (one per selected file).
    /// `pointCloud` is the first of these (the telemetry / DoF "front" cloud);
    /// the rest are extra clouds the rasteriser merges each frame.
    private var plyClouds: [PointRasteriserPointCloud] = []

    #if canImport(SwiftPDAL)
    // One StreamingAdapter / cloud per open COPC file. PointRasteriser
    // traverses and merges all added clouds per frame, so concurrent streams
    // "just work" — this exercises the merged multi-cloud dispatch path.
    private var streamingAdapters: [StreamingAdapter] = []
    private var streamingClouds: [PointRasteriserPointCloud] = []
    private var lastCOPCURLs: [URL] = []
    #endif
    /// COPC files to auto-load on `setup()` (from repeatable `--copc <path>`
    /// launch arguments). Loaded through the same path as the file importer.
    private let initialCOPCURLs: [URL]

    // Cycled by the 'A' key: 0 = full sweep (default), then amortized budgets.
    // Mirrors into `appState.lodPointsPerFrame` (the settings UI slider drives
    // the same field), so either input source stays in sync.
    private let amortizeBudgets = [0, 40_000, 8_000]
    private var amortizeIndex = 0

    // 'D' toggles a live-compiled sine-wave displacement sketch (proves the
    // DisplacementPass end-to-end, independent of the DoF recipe below). Both
    // write each cloud's displacementBuffer — don't enable both at once.
    private var displacementPass: DisplacementPass?
    private var displaceTime: Float = 0
    private let startDate = Date()

    /// Demo sketch: warps points vertically by a travelling sine of world-X.
    /// `amplitude` and `frequency` are bound per frame (USER1/USER2) so the
    /// settings sliders can amplify the wave — essential for large COPC scenes
    /// where the old hard-coded 0.08 unit / ×8 values were invisible.
    private static let displacementKernel = """
    kernel void computeDisplacement(
        uint id [[thread_position_in_grid]],
        SCR_DISPLACEMENT_KERNEL_BUFFERS,
        constant float &time [[buffer(SCR_DISP_BUF_USER0)]],
        constant float &amplitude [[buffer(SCR_DISP_BUF_USER1)]],
        constant float &frequency [[buffer(SCR_DISP_BUF_USER2)]]
    ) {
        RasterBatch batch; uint pointIndex; uint localOffset;
        if (!scr_resolveDisplacementThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
        const float3 p = scr_decodePointAt(pointIndex, batch, xyzLow, xyzMed, xyzHigh, levels);
        const float wave = sin(p.x * frequency + time * 3.0) * amplitude;
        displacements[pointIndex] = float3(0.0, wave, 0.0);
    }
    """

    // MARK: - Depth of field (ported from Satin-ComputeRasteriser's DoF recipe:
    // DisplacementPass jitter scatters out-of-focus points; TintPass in
    // coverage/OIT mode makes them translucent instead of hard-occluding).

    private var dofDisplacementPass: DisplacementPass?
    private var dofTintPass: TintPass?
    // Previously-applied DoF toggle states, so `encodeDof` disables a pass only
    // on the enabled→disabled transition. Calling `disable()` every frame stomps
    // the rasteriser-global `configuration.applyDisplacement` flag that other
    // passes (e.g. the 'D'-key sine demo) enable during the same draw.
    private var dofWasEnabled = false
    private var dofJitterWasEnabled = false
    private var dofTranslucentWasEnabled = false
    /// World-space centre of the loaded cloud, for `dofAutoFocus`.
    private var cloudCenter: SIMD3<Float> = .zero
    /// World-space radius (half bounding diagonal) of the loaded cloud, used to
    /// scale the sine-displacement amplitude/frequency so the demo is visible on
    /// clouds of any extent. Defaults to the fixture's ~unit-cube radius.
    private var cloudRadius: Float = 0.87

    // Byte-compatible with the embedded .metal `CameraUniforms` / `DofParams`.
    private struct DofCameraUniforms {
        var modelView: simd_float4x4 = matrix_identity_float4x4
        var near: Float = 0.01
        var far: Float = 100
        var focalDistance: Float = 1
    }

    private struct DofParams {
        var band: Float = 0.04
        var falloff: Float = 0.25
        var scatter: Float = 0.05
        var maxDefocus: Float = 0.85
    }

    init(initialCOPCURLs: [URL] = []) {
        self.initialCOPCURLs = initialCOPCURLs
        let device = MTLCreateSystemDefaultDevice()!
        let context = Context(
            device: device,
            sampleCount: 1,
            colorPixelFormat: .bgra8Unorm,
            depthPixelFormat: .depth32Float
        )
        rasteriser = PointRasteriser(context: context)
        pointCloud = PointRasteriserPointCloud(
            context: context,
            packed: PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 64) // ≈262k points
        )
        super.init(context: context)
    }

    override func setup() {
        renderer.setClearColor([0.025, 0.028, 0.034, 1.0])
        rasteriser.addPointCloud(pointCloud)
        camera.lookAt(target: .zero)
        let controller = PerspectiveCameraController(camera: camera, view: metalView)
        controller.defaultDistance = 2.4
        controller.enable()
        cameraController = controller

        // Write the demo kernel to a temp file and build a (live) DisplacementPass.
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("PointRasteriserExampleDisplacement.metal")
        try? Self.displacementKernel.write(to: url, atomically: true, encoding: .utf8)
        displacementPass = DisplacementPass(rasteriser: rasteriser, kernelURL: url, live: true)
        displacementPass?.bindUserBuffers = { [weak self] encoder in
            guard let self else { return }
            var t = self.displaceTime
            // Amplitude/frequency are authored in cloud-radius-relative units and
            // scaled to world here, so one slider setting reads the same on the
            // unit-cube fixture and a 100 m COPC scene.
            let radius = max(self.cloudRadius, 1e-4)
            var amplitude = self.appState.sineDisplacementAmplitude * radius
            var frequency = self.appState.sineDisplacementFrequency / radius
            encoder.setBytes(&t, length: MemoryLayout<Float>.stride, index: DisplacementPass.bufferUser0)
            encoder.setBytes(&amplitude, length: MemoryLayout<Float>.stride, index: DisplacementPass.bufferUser1)
            encoder.setBytes(&frequency, length: MemoryLayout<Float>.stride, index: DisplacementPass.bufferUser2)
        }

        makeDofPasses()

        #if canImport(SwiftPDAL)
        if !initialCOPCURLs.isEmpty {
            loadCOPC(urls: initialCOPCURLs)
        }
        #endif
    }

    override func update() {
        cameraController?.update()
        rasteriser.configuration = appState.makeConfiguration()
        refreshLODTelemetry()
        #if canImport(SwiftPDAL)
        refreshStreamingTelemetry()
        #endif
    }

    /// Pull the front sweep's CPU-readable stats into `appState` for the settings UI.
    private func refreshLODTelemetry() {
        appState.lodSweepProgress = pointCloud.lodSweepProgress
        appState.lodCount = pointCloud.lodCount
        appState.lodOverflow = pointCloud.lodOverflow
        appState.lodOverflowed = pointCloud.lodOverflowed
        appState.lodSelectSkipped = rasteriser.lodSelectSkippedLastFrame > 0 && rasteriser.lodSelectRanLastFrame == 0
    }

    override func cleanup() {
        cameraController?.disable()
        cameraController = nil
        super.cleanup()
    }

    override func draw(renderPassDescriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer) {
        // Sine-wave demo and DoF passes run first, into the same command
        // buffer, so the rasteriser's depth/color passes see this frame's
        // displacement/tint.
        if appState.sineDisplacementEnabled {
            displaceTime = Float(Date().timeIntervalSince(startDate))
            // cloud: nil → displace every loaded cloud (all imported PLYs), each
            // into its own displacementBuffer.
            displacementPass?.encode(commandBuffer: commandBuffer, cloud: nil)
        }
        encodeDof(commandBuffer: commandBuffer)

        renderer.draw(
            renderPassDescriptor: renderPassDescriptor,
            commandBuffer: commandBuffer,
            scene: scene,
            camera: camera
        )
        rasteriser.draw(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer)
    }

    override func resize(size: (width: Float, height: Float), scaleFactor: Float) {
        camera.aspect = size.width / size.height
        cameraController?.resize(size)
        renderer.resize(size)
        rasteriser.resize(size: size, scaleFactor: scaleFactor)
        currentViewport = SIMD2(size.width, size.height)
    }

    // 'A' cycles the amortized LOD budget (0 = full sweep). 'R' restarts the sweep.
    override func keyDown(with event: NSEvent) -> Bool {
        switch event.charactersIgnoringModifiers?.lowercased() {
        case "a":
            amortizeIndex = (amortizeIndex + 1) % amortizeBudgets.count
            let budget = amortizeBudgets[amortizeIndex]
            appState.lodPointsPerFrame = budget
            print("lodPointsPerFrame = \(budget) (0 = full sweep)")
            return true
        case "r":
            restartLODSweep()
            return true
        case "d":
            appState.sineDisplacementEnabled.toggle()
            if !appState.sineDisplacementEnabled { displacementPass?.disable() }
            print("sine displacement \(appState.sineDisplacementEnabled ? "on" : "off")")
            return true
        default:
            return false
        }
    }

    /// Abandon the in-flight LOD sweep and restart it from batch 0. Exposed
    /// for the settings UI's "Restart sweep" button.
    func restartLODSweep() {
        rasteriser.restartLODSweep()
        print("restarted LOD sweep")
    }

    // MARK: - PLY loading

    /// Convenience single-file entry point — loads `url` as a one-element set.
    func loadPLY(url: URL) {
        loadPLYs(urls: [url])
    }

    /// Parses one or more PLY files off the main thread, then installs them as
    /// **separate** clouds (all added to the rasteriser, which merges them each
    /// frame), replacing whatever set is currently loaded, and reframes the
    /// camera to the combined bounds. Per-file parse errors are surfaced through
    /// ``PointRasteriserExampleState/errorMessage`` and the file is skipped, so a
    /// selection with one bad file still loads the rest.
    func loadPLYs(urls: [URL]) {
        guard !urls.isEmpty else { return }
        appState.isLoading = true
        appState.errorMessage = nil
        Task.detached { [weak self] in
            guard let self else { return }
            // Parse every file to contiguous arrays off the main thread; the GPU
            // pack (the fast path) then runs on the main thread so even 40M+
            // point clouds load in a fraction of the old CPU pack.
            var parsed: [ParsedPLY] = []
            var parseSeconds = 0.0
            for url in urls {
                let shouldStop = url.startAccessingSecurityScopedResource()
                defer { if shouldStop { url.stopAccessingSecurityScopedResource() } }
                do {
                    let parseStart = Date()
                    let (positions, colors) = try PLYPointCloudLoader.loadArrays(url: url)
                    parseSeconds += Date().timeIntervalSince(parseStart)
                    parsed.append(ParsedPLY(name: url.lastPathComponent, positions: positions, colors: colors))
                } catch {
                    await MainActor.run {
                        self.appState.errorMessage = "PLY load failed (\(url.lastPathComponent)): \(error.localizedDescription)"
                    }
                }
            }
            let totalParseSeconds = parseSeconds
            await MainActor.run {
                self.installGPUPackedClouds(parsed, parseSeconds: totalParseSeconds)
            }
        }
    }

    /// One parsed PLY file, ready to GPU-pack on the main thread.
    private struct ParsedPLY {
        let name: String
        let positions: [SIMD3<Float>]
        let colors: [SIMD4<Float>]
    }

    /// Cached GPU packer (compiles the pack kernels once). Nil until first use.
    private var gpuPacker: GPUPacker?

    @MainActor
    private func installGPUPackedClouds(_ parsed: [ParsedPLY], parseSeconds: TimeInterval) {
        guard !parsed.isEmpty else {
            appState.isLoading = false
            return
        }
        do {
            let packer: GPUPacker
            if let existing = gpuPacker {
                packer = existing
            } else {
                packer = try GPUPacker(device: defaultContext.device)
                gpuPacker = packer
            }

            // Drop the current set (fixture / prior PLYs / any streams) before
            // appending the new clouds, so `pointCloud` is never a stale child.
            teardownLoadedClouds()
            #if canImport(SwiftPDAL)
            appState.isStreaming = false
            #endif

            let packStart = Date()
            var newClouds: [PointRasteriserPointCloud] = []
            var totalPoints = 0
            var combinedMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
            var combinedMax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
            for item in parsed {
                let cloud = PointRasteriserPointCloud.gpuPacked(
                    context: defaultContext, packer: packer, queue: defaultContext.commandQueue,
                    positions: item.positions, colors: item.colors, label: item.name
                )
                rasteriser.addPointCloud(cloud)
                newClouds.append(cloud)
                totalPoints += cloud.pointCount
                combinedMin = simd_min(combinedMin, cloud.sourceBoundsMin)
                combinedMax = simd_max(combinedMax, cloud.sourceBoundsMax)
            }
            let packSeconds = Date().timeIntervalSince(packStart)

            plyClouds = newClouds
            pointCloud = newClouds[0]
            frameCamera(toBoundsMin: combinedMin, boundsMax: combinedMax)
            rasteriser.restartLODSweep()
            appState.isLoading = false
            let label = parsed.count == 1 ? parsed[0].name : "\(parsed[0].name) +\(parsed.count - 1)"
            let fileWord = parsed.count == 1 ? "file" : "files"
            appState.status = "\(label) (\(totalPoints) pts across \(parsed.count) \(fileWord)) — parse \(String(format: "%.2f", parseSeconds))s, GPU pack \(String(format: "%.3f", packSeconds))s"
            appState.errorMessage = nil
            print("[loadPLYs] \(label): \(totalPoints) pts, \(parsed.count) \(fileWord), parse \(String(format: "%.2f", parseSeconds))s, GPU pack \(String(format: "%.3f", packSeconds))s")
        } catch {
            appState.isLoading = false
            appState.errorMessage = error.localizedDescription
        }
    }

    /// Remove every currently-loaded cloud from the rasteriser — the fixture /
    /// PLY set and (when the streaming module is present) all COPC streams — so a
    /// fresh import can append cleanly. `removePointCloud` is a no-op for a cloud
    /// that isn't a child, so the belt-and-suspenders `pointCloud` removal is
    /// safe even when it's already one of `plyClouds`.
    @MainActor
    private func teardownLoadedClouds() {
        #if canImport(SwiftPDAL)
        for adapter in streamingAdapters { adapter.close() }
        streamingAdapters.removeAll()
        for cloud in streamingClouds { rasteriser.removePointCloud(cloud) }
        streamingClouds.removeAll()
        #endif
        for cloud in plyClouds { rasteriser.removePointCloud(cloud) }
        plyClouds.removeAll()
        rasteriser.removePointCloud(pointCloud)
    }

    private func frameCamera(toBoundsMin boundsMin: SIMD3<Float>, boundsMax: SIMD3<Float>) {
        let center = (boundsMin + boundsMax) * 0.5
        let extent = boundsMax - boundsMin
        let radius = max(simd_length(extent) * 0.5, 0.01)
        let distance = radius / max(tan(camera.fov * 0.5 * .pi / 180.0), 0.001)
        let position = center + SIMD3<Float>(0, 0, distance * 1.35)

        // DoF: focus on the cloud centre by default; size the manual slider to fit.
        cloudCenter = center
        cloudRadius = radius
        let camDistance = simd_length(position - center)
        appState.dofFocus = camDistance
        appState.dofFocusMax = max(camDistance * 3, radius * 6)

        cameraController?.disable()
        camera.position = position
        camera.near = max(distance - radius * 3.0, 0.001)
        camera.far = distance + radius * 4.0
        camera.lookAt(target: center)
        cameraController?.defaultDistance = simd_length(position - center)
        cameraController?.defaultPosition = camera.position
        cameraController?.defaultOrientation = camera.orientation
        cameraController?.enable()
    }
}

// MARK: - Streaming (COPC)

#if canImport(SwiftPDAL)
extension PointRasteriserExampleRenderer {
    /// Pull each active adapter's residency telemetry into `appState` for the
    /// settings sheet + status overlay. No-op (and zeroes the fields) when
    /// nothing is streaming.
    private func refreshStreamingTelemetry() {
        guard !streamingAdapters.isEmpty else { return }
        var chunks = 0
        var points = 0
        var freeSlots = 0
        var pinned = 0
        var pending = 0
        var starved = 0
        var decodeMPS = 0.0
        var decodePending = 0
        var decodeInFlight = 0
        var firstError: String?
        for adapter in streamingAdapters {
            adapter.update(camera: camera, viewport: currentViewport)
            chunks += adapter.residentChunks
            points += adapter.residentPoints
            pinned += adapter.pinnedResidentChunks
            pending += adapter.pendingUploadCount
            starved += adapter.starvedTickCount
            decodeMPS += adapter.decodedPointsPerSecond / 1_000_000
            decodePending += adapter.decodePendingRequests
            decodeInFlight += adapter.decodeInFlight
            if firstError == nil { firstError = adapter.lastError }
        }
        for cloud in streamingClouds { freeSlots += cloud.freeSlotCount }
        appState.streamingChunks = chunks
        appState.streamingPoints = points
        appState.streamingFreeSlots = freeSlots
        appState.streamingPinnedChunks = pinned
        appState.streamingPendingUploads = pending
        appState.streamingStarvedTicks = starved
        appState.streamingDecodeMPS = decodeMPS
        appState.streamingDecodePending = decodePending
        appState.streamingDecodeInFlight = decodeInFlight
        if let firstError { appState.errorMessage = firstError }
    }

    /// Live-retune the residency detail target (screen px per node; smaller =
    /// more detail). Applies to every open source without a re-open.
    func setStreamingTargetChunkPx(_ px: Float) {
        appState.streamingTargetChunkPx = px
        for adapter in streamingAdapters { adapter.setTargetChunkScreenSize(px) }
    }

    /// Convenience single-file entry point — loads `url` as a one-element set.
    func loadCOPC(url: URL) {
        loadCOPC(urls: [url])
    }

    /// Open one or more COPC LAZ files, each as its own streaming source /
    /// cloud / adapter, all added to the scene. Replaces whatever set is
    /// currently loaded (the previous streams and the PLY/fixture cloud are
    /// torn down first). The `streamingBudgetMB` total is split equally
    /// across the N sources so aggregate memory stays comparable to the
    /// single-file case, and the camera frames to the first source's bounds.
    func loadCOPC(urls: [URL]) {
        guard !urls.isEmpty else { return }
        Task.detached { [weak self] in
            guard let self else { return }
            // Tear down the existing set and split the budget on the main
            // actor before any source opens, so installs only ever append.
            let (perSourceBytes, residency) = await MainActor.run { () -> (Int, StreamingResidencyChoice) in
                let totalBytes = max(64, self.appState.streamingBudgetMB) * 1024 * 1024
                let perSourceBytes = max(1, totalBytes / urls.count)
                self.lastCOPCURLs = urls
                self.teardownLoadedClouds()
                self.appState.status = urls.count == 1
                    ? urls[0].lastPathComponent
                    : "\(urls[0].lastPathComponent) +\(urls.count - 1)"
                self.appState.errorMessage = nil
                self.appState.isStreaming = true
                self.appState.streamingChunks = 0
                self.appState.streamingPoints = 0
                self.appState.streamingFreeSlots = 0
                return (perSourceBytes, self.appState.streamingResidency)
            }
            await withTaskGroup(of: Void.self) { group in
                for url in urls {
                    group.addTask {
                        await self.openAndInstall(
                            url: url, budgetBytes: perSourceBytes,
                            residency: residency, cloudCount: urls.count
                        )
                    }
                }
            }
        }
    }

    private func openAndInstall(url: URL, budgetBytes: Int, residency: StreamingResidencyChoice, cloudCount: Int) async {
        let shouldStop = url.startAccessingSecurityScopedResource()
        defer {
            if shouldStop { url.stopAccessingSecurityScopedResource() }
        }
        do {
            // Parallel decode via reader pool + fast ticks. decodeConcurrency
            // = active core count; LAZ decompress is single-threaded per
            // chunk so going past cores doesn't help.
            let cores = max(2, ProcessInfo.processInfo.activeProcessorCount)
            let policy: SwiftPDAL.ResidencyPolicy =
                residency == .distance ? .distanceOnly : .frustumFirstThenHalo
            // Source-native coarse pinning: depth ≤ 2 nodes stay resident so the
            // scene always has low-density coverage ("briefly coarse", never
            // black). A negligible point fraction; the budget clears it easily.
            let opts = StreamingOptions(
                maxInFlightLoads: cores * 2,
                decodeConcurrency: cores,
                driverTickInterval: .milliseconds(16),
                residencyPolicy: policy,
                alwaysResidentDepth: 2
            )
            let source = try await SwiftPDAL.CopcStreamingPointCloudSource.open(url, options: opts)
            source.setBudget(budgetBytes)
            await MainActor.run {
                self.installStreamingSource(source, url: url, budgetBytes: budgetBytes, cloudCount: cloudCount)
            }
        } catch {
            // Also log to stdout — the HUD isn't visible when the app is
            // driven by launch arguments for capture sessions.
            print("[PointRasteriserExample] COPC open failed for \(url.path): \(error)")
            await MainActor.run {
                self.appState.errorMessage = "COPC open failed (\(url.lastPathComponent)): \(error.localizedDescription)"
            }
        }
    }

    func setStreamingBudget(MB: Int) {
        appState.streamingBudgetMB = MB
        // The slider means "total across all clouds" — split equally.
        let totalBytes = max(64, MB) * 1024 * 1024
        let perSourceBytes = max(1, totalBytes / max(1, streamingAdapters.count))
        for adapter in streamingAdapters { adapter.setBudget(bytes: perSourceBytes) }
    }

    /// Switching residency policy requires re-opening the sources — the
    /// driver reads the policy at construction. Re-opens the current set
    /// with the new choice; ~100 ms hiccup while the hierarchy is rescanned.
    func setResidency(_ choice: StreamingResidencyChoice) {
        appState.streamingResidency = choice
        if !lastCOPCURLs.isEmpty {
            loadCOPC(urls: lastCOPCURLs)
        }
    }

    @MainActor
    private func installStreamingSource(_ source: SwiftPDAL.CopcStreamingPointCloudSource, url: URL, budgetBytes: Int, cloudCount: Int) {
        // The set was already torn down in `loadCOPC(urls:)`; this only appends.
        let isFirst = streamingClouds.isEmpty

        // Pool capacity: cap by budget so we don't oversubscribe VRAM. Every
        // chunk occupies whole slots (its last batch is partial), so a pool
        // sized 1:1 to the budget exhausts before the source stops admitting
        // and the adapter drops chunks ("slot pool full"). 2x slot headroom
        // absorbs that granularity waste. The 65536-slot ceiling keeps the
        // per-frame threadgroup count bounded (every slot gets a threadgroup
        // whether resident or not) and is divided across the set so N clouds
        // cost what one did.
        let pointsPerBatch = source.info.pointsPerBatch
        let bytesPerSlot = pointsPerBatch * source.info.bytesPerPoint
        let slotsByBudget = max(1, budgetBytes / max(1, bytesPerSlot))
        let slotCapacity = min(slotsByBudget * 2, 65536 / max(1, cloudCount))

        // The source pre-shifts every chunk's positions by `info.originShift`
        // so they're small FP32-safe values centered near origin. We
        // deliberately do NOT bake originShift back into RasterFile.world —
        // that would re-translate to absolute world coords and the
        // subsequent `viewMatrix * world` would combine two huge
        // translations and lose precision. Render in shifted space; frame
        // the camera there too.
        let originShiftF = SIMD3<Float>(
            Float(source.info.originShift.x),
            Float(source.info.originShift.y),
            Float(source.info.originShift.z)
        )
        let cloud = PointRasteriserPointCloud(
            context: defaultContext,
            slotCapacity: slotCapacity,
            pointsPerBatch: pointsPerBatch,
            label: "PointRasteriserPointCloud.Streaming"
        )
        rasteriser.addPointCloud(cloud)
        streamingClouds.append(cloud)

        let adapter = StreamingAdapter(source: source, cloud: cloud)
        streamingAdapters.append(adapter)

        // Frame to (and point DoF at) the first cloud of the set. Each
        // dataset is pre-shifted to near-origin, so the others overlap it —
        // fine here.
        if isFirst {
            pointCloud = cloud
            let shiftedMin = source.info.bounds.min - originShiftF
            let shiftedMax = source.info.bounds.max - originShiftF
            frameCamera(toBoundsMin: shiftedMin, boundsMax: shiftedMax)
        }
    }
}
#endif

// MARK: - Depth of field

extension PointRasteriserExampleRenderer {
    /// Build the DoF passes from the embedded kernels (identical contract to
    /// Satin-ComputeRasteriser's DoF recipe — `SCR_DISPLACEMENT_KERNEL_BUFFERS`
    /// / `SCR_TINT_KERNEL_BUFFERS` and the `SCR_*_BUF_USER0/1` slots are
    /// unchanged). The displacement pass scatters out-of-focus points; the
    /// tint pass (in coverage / weighted-blended OIT mode) makes them
    /// see-through so they blend instead of hard-occluding.
    func makeDofPasses() {
        guard let jURL = writeTempKernel(Self.dofJitterKernel, name: "PointRasteriserExampleDofJitter"),
              let tURL = writeTempKernel(Self.dofTranslucencyKernel, name: "PointRasteriserExampleDofTranslucency")
        else { return }
        let dp = DisplacementPass(rasteriser: rasteriser, kernelURL: jURL, live: false)
        dp.bindUserBuffers = { [weak self] enc in self?.bindDof(enc, user0: DisplacementPass.bufferUser0) }
        dofDisplacementPass = dp
        let tp = TintPass(rasteriser: rasteriser, kernelURL: tURL, live: false)
        tp.alphaIsCoverage = true // translucent defocus (OIT), not a colour mix
        tp.bindUserBuffers = { [weak self] enc in self?.bindDof(enc, user0: TintPass.bufferUser0) }
        dofTintPass = tp
    }

    /// Encode the DoF passes before the rasteriser draws (so the colour pass
    /// sees this frame's displacement + tint). Disables a pass when its
    /// toggle is off.
    func encodeDof(commandBuffer: MTLCommandBuffer) {
        guard appState.dofEnabled else {
            // Only disable on the enabled→disabled transition; disabling every
            // frame would stomp the rasteriser-global `applyDisplacement` flag
            // that other passes (the sine demo) enable during the same draw.
            if dofWasEnabled {
                dofDisplacementPass?.disable(); dofTintPass?.disable()
            }
            dofWasEnabled = false
            dofJitterWasEnabled = false
            dofTranslucentWasEnabled = false
            return
        }
        dofWasEnabled = true
        // cloud: nil → apply DoF to every loaded cloud. `bindDof` binds the front
        // cloud's world/focus for all of them; imported PLYs share an identity
        // world matrix, so that's exact here.
        if appState.dofJitter {
            dofDisplacementPass?.encode(commandBuffer: commandBuffer, cloud: nil)
        } else if dofJitterWasEnabled {
            dofDisplacementPass?.disable()
        }
        dofJitterWasEnabled = appState.dofJitter
        if appState.dofTranslucent {
            dofTintPass?.encode(commandBuffer: commandBuffer, cloud: nil)
        } else if dofTranslucentWasEnabled {
            dofTintPass?.disable()
        }
        dofTranslucentWasEnabled = appState.dofTranslucent
    }

    /// Bind USER0 = per-cloud camera (modelView + focal distance), USER1 = the
    /// DoF params, for whichever pass is encoding. The focus band is a
    /// fraction of the focal distance, so it auto-scales to the loaded cloud.
    private func bindDof(_ enc: MTLComputeCommandEncoder, user0: Int) {
        let fileWorld = pointCloud.files.first?.world ?? matrix_identity_float4x4
        let world = pointCloud.worldMatrix * fileWorld
        let focus = appState.dofAutoFocus ? simd_length(camera.worldPosition - cloudCenter) : appState.dofFocus
        var cam = DofCameraUniforms(
            modelView: camera.viewMatrix * world,
            near: camera.near, far: camera.far, focalDistance: max(focus, 1e-3)
        )
        withUnsafeBytes(of: &cam) { enc.setBytes($0.baseAddress!, length: $0.count, index: user0) }
        var dof = DofParams(
            band: appState.dofBand, falloff: appState.dofFalloff,
            scatter: appState.dofJitter ? appState.dofScatter : 0,
            maxDefocus: appState.dofTranslucent ? appState.dofMaxDefocus : 0
        )
        withUnsafeBytes(of: &dof) { enc.setBytes($0.baseAddress!, length: $0.count, index: user0 + 1) }
    }

    private func writeTempKernel(_ source: String, name: String) -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(name).metal")
        do {
            try source.write(to: url, atomically: true, encoding: .utf8)
            return url
        } catch {
            appState.errorMessage = "DoF kernel write failed: \(error.localizedDescription)"
            return nil
        }
    }

    // USER0 = camera, USER1 = DoF params. The package's ScrSketchPreamble
    // prepends its preamble before compile (same macros as the sibling).
    private static let dofStructs = """
    typedef struct {
        float4x4 modelView;     // camera.view · cloud.world  (decoded-local → view)
        float    near;
        float    far;
        float    focalDistance; // sharp distance from the camera (view-space units)
    } CameraUniforms;
    typedef struct {
        float band;       // sharp half-band, fraction of focal distance
        float falloff;    // ramp to full effect, fraction of focal distance
        float scatter;    // jitter spread, fraction of focal distance
        float maxDefocus; // transparency cap (1 = can fully vanish)
    } DofParams;
    inline float dofHash1(uint x) {
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        return float(x) * (1.0 / 4294967296.0);
    }
    inline float3 dofHash3(uint i, uint salt) {
        return float3(dofHash1(i * 747796405u  + salt),
                      dofHash1(i * 2891336453u + salt + 17u),
                      dofHash1(i * 3266489917u + salt + 101u)) * 2.0 - 1.0;
    }
    """

    static let dofJitterKernel = dofStructs + """

    kernel void computeDisplacement(
        SCR_DISPLACEMENT_KERNEL_BUFFERS,
        constant CameraUniforms &cam [[buffer(SCR_DISP_BUF_USER0)]],
        constant DofParams      &dof [[buffer(SCR_DISP_BUF_USER1)]],
        uint id [[thread_position_in_grid]])
    {
        RasterBatch batch; uint pointIndex, localOffset;
        if (!scr_resolveDisplacementThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
        const float3 p = scr_decodePointAt(pointIndex, batch, xyzLow, xyzMed, xyzHigh, levels);
        const float viewDepth = -(cam.modelView * float4(p, 1.0)).z;
        const float band    = cam.focalDistance * dof.band;
        const float falloff = max(cam.focalDistance * dof.falloff, 1e-3);
        const float coc = saturate((abs(viewDepth - cam.focalDistance) - band) / falloff);
        const float3 dir = dofHash3(pointIndex, 0u);
        displacements[pointIndex] = dir * (coc * coc * cam.focalDistance * dof.scatter);
    }
    """

    static let dofTranslucencyKernel = dofStructs + """

    kernel void computeTint(
        SCR_TINT_KERNEL_BUFFERS,
        constant CameraUniforms &cam [[buffer(SCR_TINT_BUF_USER0)]],
        constant DofParams      &dof [[buffer(SCR_TINT_BUF_USER1)]],
        uint id [[thread_position_in_grid]])
    {
        RasterBatch batch; uint pointIndex, localOffset;
        if (!scr_resolveTintThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
        const float3 p = scr_decodePointAt(pointIndex, batch, xyzLow, xyzMed, xyzHigh, levels);
        const float viewDepth = -(cam.modelView * float4(p, 1.0)).z;
        const float band    = cam.focalDistance * dof.band;
        const float falloff = max(cam.focalDistance * dof.falloff, 1e-3);
        const float coc = saturate((abs(viewDepth - cam.focalDistance) - band) / falloff);
        // alpha = coc → rasteriser composites with coverage = 1 - coc (translucent).
        tints[pointIndex] = float4(0.0, 0.0, 0.0, coc * saturate(dof.maxDefocus));
    }
    """
}
#endif
