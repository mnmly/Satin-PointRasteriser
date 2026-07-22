import Foundation
import Satin
import SatinPointRasteriser
import SwiftPDAL
import simd

/// Projection-math helper used to convert renderer camera state into the
/// screen-space pixel scale SwiftPDAL's residency scorer expects
/// (``SwiftPDAL/StreamingCameraView/pixelScale``).
///
/// Kept local to the streaming target (rather than the core library) so the
/// core stays free of any streaming-specific vocabulary — see
/// ``StreamingAdapter``'s doc comment for the split rationale.
public enum PointRasteriserStreamingProjection {
    /// Scale-invariant screen-space-pixel factor:
    ///
    /// ```
    /// pixelScale = viewportHeight / (2 · tan(fovY / 2))
    ///            = viewportHeight × 0.5 × projectionMatrix[1][1]
    /// ```
    ///
    /// `projectionMatrix[1][1]` equals `1 / tan(fovY / 2)` for any perspective
    /// matrix; for an orthographic matrix it is `2 / orthoHeight`, so the same
    /// expression yields the depth-independent pixels-per-world-unit scale.
    /// SwiftPDAL's scorer consumes both forms. Using the matrix form avoids
    /// depending on a particular camera type or tracking the FOV separately.
    @inlinable
    public static func screenSpacePixelScale(
        viewportHeight: Float,
        projectionMatrix: simd_float4x4
    ) -> Float {
        viewportHeight * 0.5 * projectionMatrix.columns.1.y
    }
}

/// Why a settled residency target stopped short of full coverage.
///
/// A settled frame is either fully covered or clamped; when clamped, the two
/// causes are independent and can co-occur:
/// * `byBudget` — the driver's byte budget excluded frustum-visible candidates
///   (mirrors ``StreamingAdapter/residencyBudgetLimited``). Raise
///   ``StreamingAdapter/setBudget(bytes:)`` for fuller coverage.
/// * `byPool` — the GPU upload backlog is permanently blocked on a full slot
///   pool (mirrors ``StreamingAdapter/uploadBlockedOnFullPool``). Raise the
///   ``PointRasteriserPointCloud`` slot capacity.
public enum StreamingResidencyCoverage: Equatable, Sendable {
    /// Every wanted chunk for the submitted views is resident and uploaded.
    case full
    /// Settled short of full: the wanted set was clamped. `byBudget` = the
    /// driver's budget excluded frustum-visible candidates; `byPool` = the
    /// upload backlog is blocked on a full slot pool.
    case clamped(byBudget: Bool, byPool: Bool)
}

/// Typed settle state for a streaming residency target — the structured form of
/// ``StreamingAdapter/isCaughtUp`` plus its two clamp flags. Derived purely from
/// existing adapter members; see ``StreamingAdapter/residencyState``.
public enum StreamingResidencyState: Equatable, Sendable {
    /// Still streaming toward the submitted target.
    case streaming
    /// Nothing more will change for the submitted target at the current budget.
    case settled(StreamingResidencyCoverage)
}

/// An immutable, `Sendable` snapshot of the adapter's residency + decode
/// telemetry, replacing the pile of individually-read properties.
///
/// Every field is a copy taken at snapshot time; the struct holds no reference
/// to the adapter. Like the individual telemetry properties it is sourced from,
/// it **must be read on the same thread that calls ``StreamingAdapter/pump()``**
/// (see the "Telemetry (read on the main thread)" MARK in ``StreamingAdapter``)
/// — the underlying counters are mutated by `pump()` without synchronization.
public struct StreamingResidencyTelemetry: Sendable {
    /// Typed settle state (``StreamingAdapter/residencyState``).
    public var state: StreamingResidencyState
    /// Slot-pool capacity in whole batches (`cloud.batchCount`).
    public var poolCapacity: Int
    /// Free (unallocated) slots in the pool (`cloud.freeSlotCount`).
    public var freeSlots: Int
    /// Chunks currently resident in the slot pool.
    public var residentChunks: Int
    /// Sum of points across resident chunks.
    public var residentPoints: Int
    /// Resident chunks at depth ≤ `coarsePinnedDepth` (guaranteed-coverage set).
    public var pinnedResidentChunks: Int
    /// Decoded-but-not-yet-uploaded chunks in the GPU upload backlog.
    public var pendingUploadCount: Int
    /// Source decode-pipeline nodes mid-decompress.
    public var decodeInFlight: Int
    /// Source decode-pipeline nodes scheduled but not yet decoding.
    public var decodePendingRequests: Int
    /// Estimated decode throughput in points/second (EMA).
    public var decodedPointsPerSecond: Double
    /// Chunks uploaded to the GPU pool over the adapter's lifetime.
    public var totalChunksUploaded: Int
    /// Chunks evicted (slots freed) over the adapter's lifetime.
    public var totalChunksEvicted: Int
    /// Decoded chunks dropped from the pending queue by pre-upload eviction.
    public var totalPendingDroppedByEviction: Int
    /// Ticks on which a pending chunk could not upload (slot pool full).
    public var starvedTickCount: Int
    /// Last non-fatal notice (e.g. "slot pool full"), if any.
    public var lastError: String?

    public init(
        state: StreamingResidencyState,
        poolCapacity: Int,
        freeSlots: Int,
        residentChunks: Int,
        residentPoints: Int,
        pinnedResidentChunks: Int,
        pendingUploadCount: Int,
        decodeInFlight: Int,
        decodePendingRequests: Int,
        decodedPointsPerSecond: Double,
        totalChunksUploaded: Int,
        totalChunksEvicted: Int,
        totalPendingDroppedByEviction: Int,
        starvedTickCount: Int,
        lastError: String?
    ) {
        self.state = state
        self.poolCapacity = poolCapacity
        self.freeSlots = freeSlots
        self.residentChunks = residentChunks
        self.residentPoints = residentPoints
        self.pinnedResidentChunks = pinnedResidentChunks
        self.pendingUploadCount = pendingUploadCount
        self.decodeInFlight = decodeInFlight
        self.decodePendingRequests = decodePendingRequests
        self.decodedPointsPerSecond = decodedPointsPerSecond
        self.totalChunksUploaded = totalChunksUploaded
        self.totalChunksEvicted = totalChunksEvicted
        self.totalPendingDroppedByEviction = totalPendingDroppedByEviction
        self.starvedTickCount = starvedTickCount
        self.lastError = lastError
    }
}

/// Bridges a SwiftPDAL `StreamingPointCloudSource` to a
/// `PointRasteriserPointCloud` slot pool.
///
/// The renderer package stays free of PDAL/lazperf; this adapter is the glue
/// layer where the streaming driver and the GPU residency pool meet.
///
/// ### Fill strategy (fixes the "missing blocks / never fill" class)
/// * **Coarse pin (source-native)** — pinning is now owned by SwiftPDAL via
///   ``SwiftPDAL/StreamingOptions/alwaysResidentDepth``: nodes at depth ≤ D are
///   loaded coarse-first, never appear in an eviction delta, are re-admission
///   de-duplicated, and are charged against the byte budget. COPC's full density
///   at any point is the union of a node and its ancestors, so a handful of
///   shallow nodes blanket the whole scene at low density → "black blocks"
///   become "briefly coarse". The adapter just applies the source's deltas; set
///   ``coarsePinnedDepth`` to mirror the option so the pinned-coverage telemetry
///   is accurate. **Budget note:** because pins now count against the budget,
///   size ``setBudget(bytes:)`` to comfortably exceed the pinned footprint (the
///   pinned coarse set is a negligible fraction of the points, so the default
///   budgets clear it easily).
/// * **Throughput** — up to ``maxChunkUploadsPerTick`` decoded chunks are
///   uploaded per tick (memcpy into the slot pool). The default is generous
///   because the per-chunk memcpy is cheap (≈170 KB/batch); a low cap was the
///   integration bottleneck. Uploads are de-duplicated and slot-full requests
///   are retried on later ticks.
/// * **Slot-granularity starvation guard** — small leaf nodes each occupy a
///   whole slot, so the GPU slot pool can fill long before the source's *byte*
///   budget does. ``starvedTickCount`` surfaces this so a caller can size
///   ``PointRasteriserPointCloud`` slot capacity by node count.
public final class StreamingAdapter {
    private let source: any StreamingPointCloudSource
    private let cloud: PointRasteriserPointCloud
    private var slotsByChunk: [ChunkID: [Int]] = [:]
    private var pendingAdds: [ResidentChunk] = []
    private var pendingCursor: Int = 0

    /// Octree depth (inclusive) treated as "pinned" **for telemetry only** —
    /// used to count ``pinnedResidentChunks`` (the guaranteed-coverage set).
    /// Set this to mirror the source's
    /// ``SwiftPDAL/StreamingOptions/alwaysResidentDepth``; the actual pinning is
    /// enforced by the source (which never evicts pinned nodes), so this value
    /// no longer drives any residency decision. `-1` = no pinning.
    public var coarsePinnedDepth: Int = 2

    /// Max decoded chunks uploaded to the slot pool per ``update(camera:viewport:)``.
    /// The per-chunk memcpy is cheap; a low cap was the integration bottleneck.
    /// `Int.max` uploads the whole backlog each tick.
    public var maxChunkUploadsPerTick: Int = 32

    // MARK: - Telemetry (read on the main thread)

    /// Chunks currently resident in the slot pool.
    public private(set) var residentChunks: Int = 0
    /// Sum of points across resident chunks (mirrors `cloud.pointCount`).
    public private(set) var residentPoints: Int = 0
    /// Resident chunks at depth ≤ ``coarsePinnedDepth`` — the source-pinned
    /// guaranteed-coverage set (only meaningful when the mirror value is set).
    public private(set) var pinnedResidentChunks: Int = 0
    /// Decoded-but-not-yet-uploaded chunks in the adapter's **GPU upload**
    /// backlog (distinct from the source's decode queue — see ``decodePendingRequests``).
    public var pendingUploadCount: Int { pendingAdds.count - pendingCursor }
    /// Source decode-pipeline back-pressure: nodes scheduled but not yet
    /// decoding (from ``SwiftPDAL/StreamingDecodeStats/pendingRequests``).
    public private(set) var decodePendingRequests: Int = 0
    /// Source decode-pipeline nodes mid-decompress (from
    /// ``SwiftPDAL/StreamingDecodeStats/inFlightDecodes``).
    public private(set) var decodeInFlight: Int = 0
    /// Chunks uploaded to the GPU pool over the adapter's lifetime.
    public private(set) var totalChunksUploaded: Int = 0
    /// Chunks evicted (slots freed) over the adapter's lifetime.
    public private(set) var totalChunksEvicted: Int = 0
    /// Decoded chunks dropped from the pending queue because the source evicted
    /// them before they were uploaded (camera moved on) — lost refinement.
    public private(set) var totalPendingDroppedByEviction: Int = 0
    /// Ticks on which a pending chunk could not upload because the slot pool was
    /// full (no eviction to free a slot).
    public private(set) var starvedTickCount: Int = 0
    /// `true` while the upload backlog is blocked on a full slot pool (the last
    /// ``pump()`` broke on "no free slot" rather than draining or hitting the
    /// per-tick cap). With the source settled this state is permanent — nothing
    /// will be evicted to free a slot — so ``isCaughtUp`` treats the placeable
    /// set as complete instead of waiting forever. See ``isCaughtUp``.
    public private(set) var uploadBlockedOnFullPool: Bool = false
    /// Estimated decode throughput in points/second (EMA of the source's
    /// monotonic ``SwiftPDAL/StreamingDecodeStats/decodedPoints``).
    public private(set) var decodedPointsPerSecond: Double = 0

    /// Last non-fatal notice (e.g. "slot pool full").
    public private(set) var lastError: String?

    private var lastDecodedPoints: UInt64 = 0
    private var lastDecodeSampleTime: Date?

    public init(source: any StreamingPointCloudSource, cloud: PointRasteriserPointCloud) {
        self.source = source
        self.cloud = cloud
    }

    /// Forwards a byte budget hint to the underlying source.
    public func setBudget(bytes: Int) { source.setBudget(bytes) }

    /// Live-retune the residency scorer's target on-screen node size (smaller =
    /// more detail). Re-scores on the next pass; resident chunks are untouched.
    public func setTargetChunkScreenSize(_ pixels: Float) { source.setTargetChunkScreenSize(pixels) }

    /// Per-frame tick. Submits the latest camera view and applies the driver's
    /// delta, uploading decoded chunks into the slot pool.
    public func update(camera: Camera, viewport: SIMD2<Float>) {
        submit(camera: camera, viewport: viewport)
        pump()
    }

    /// Build a ``SwiftPDAL/StreamingCameraView`` from a renderer camera + drawable
    /// size and submit it as the residency target (single-view). Split out of
    /// ``update(camera:viewport:)`` so a caller can drive the target and the
    /// upload pump independently (e.g. an offline "pump until resident" loop).
    public func submit(camera: Camera, viewport: SIMD2<Float>) {
        source.submit(view: cameraView(camera, viewport: viewport))
    }

    /// Submit the **union** of several camera views as the residency target —
    /// for a multi-projection frame (CAVE walls + floor, stereo eyes) or camera-
    /// path look-ahead. A node is wanted if any view sees it, at the detail the
    /// best-placed view needs. Size the budget to hold the union;
    /// ``residencyBudgetLimited`` reports when it doesn't. See
    /// ``SwiftPDAL/StreamingPointCloudSource/submit(views:)``.
    public func submit(cameras: [(camera: Camera, viewport: SIMD2<Float>)]) {
        source.submit(views: cameras.map { cameraView($0.camera, viewport: $0.viewport) })
    }

    private func cameraView(_ camera: Camera, viewport: SIMD2<Float>) -> StreamingCameraView {
        StreamingCameraView(
            position: camera.worldPosition,
            viewProjection: camera.projectionMatrix * camera.viewMatrix,
            pixelScale: PointRasteriserStreamingProjection.screenSpacePixelScale(
                viewportHeight: viewport.y,
                projectionMatrix: camera.projectionMatrix
            )
        )
    }

    /// `true` once the frame is fully streamed in: every wanted chunk (for the
    /// submitted view/views at the current budget) is resident on the driver
    /// **and** uploaded into this adapter's slot pool. Read after ``pump()`` to
    /// drive a pre-roll loop. Independent of ``residencyBudgetLimited`` — a
    /// budget-clamped frame still becomes "caught up" once the clamped set is in.
    ///
    /// **Slot-granularity overflow:** the driver clamps its wanted set by *bytes*
    /// (charging each node its actual point count) while the slot pool is sized
    /// in whole batches, so a settled in-budget wanted set can need more GPU
    /// slots than the pool has — every small/partial node occupies a full slot.
    /// Once the source is settled no eviction will ever free a slot, so that
    /// overflow is permanently un-placeable. Treat it like the budget clamp:
    /// caught up once everything *placeable* is uploaded
    /// (``uploadBlockedOnFullPool``), rather than stalling forever.
    ///
    /// Reimplemented in terms of ``residencyState`` — provably identical to the
    /// original `source.isResidencySettled && (pendingUploadCount == 0 ||
    /// uploadBlockedOnFullPool)`, since ``residencyState`` returns a `.settled`
    /// case under **exactly** that condition and `.streaming` otherwise.
    public var isCaughtUp: Bool {
        if case .settled = residencyState { return true }
        return false
    }

    /// Typed settle state for the submitted residency target — the structured
    /// counterpart to ``isCaughtUp`` and its two clamp flags. Derived **only**
    /// from existing members, so `isCaughtUp == (residencyState is .settled)`
    /// holds for every input combination. Read after ``pump()``.
    ///
    /// The mapping:
    /// * source not settled, **or** an upload backlog remains that is *not*
    ///   pool-blocked → ``StreamingResidencyState/streaming``;
    /// * settled with no clamping →
    ///   ``StreamingResidencyState/settled(_:)`` of ``StreamingResidencyCoverage/full``;
    /// * settled but clamped by budget and/or a full pool →
    ///   `.settled(.clamped(byBudget:byPool:))` carrying
    ///   ``residencyBudgetLimited`` and ``uploadBlockedOnFullPool`` respectively.
    ///
    /// **Pool-clamp scenario (`.settled(.clamped(byPool: true))`):** the driver
    /// clamps its wanted set by *bytes* (charging each node its actual point
    /// count) while the slot pool allocates *whole batches*, so a settled,
    /// in-budget wanted set can need more slots than the pool has — every small
    /// or partial node still consumes a full slot. Once the source is settled no
    /// eviction will free a slot, so the overflow is permanently un-placeable and
    /// ``pump()`` leaves ``uploadBlockedOnFullPool`` set. This case is the typed
    /// signal for that condition, which was previously visible only as a
    /// ``pendingUploadCount`` stuck above zero with no further progress.
    public var residencyState: StreamingResidencyState {
        guard source.isResidencySettled,
              pendingUploadCount == 0 || uploadBlockedOnFullPool
        else { return .streaming }

        if residencyBudgetLimited || uploadBlockedOnFullPool {
            return .settled(.clamped(
                byBudget: residencyBudgetLimited,
                byPool: uploadBlockedOnFullPool
            ))
        }
        return .settled(.full)
    }

    /// A single `Sendable` snapshot of the current residency + decode telemetry,
    /// replacing ad-hoc string-formatting of the individual properties. Same
    /// threading rule as those properties: read on the thread that calls
    /// ``pump()`` (see the "Telemetry (read on the main thread)" MARK).
    public var telemetry: StreamingResidencyTelemetry {
        StreamingResidencyTelemetry(
            state: residencyState,
            poolCapacity: cloud.batchCount,
            freeSlots: cloud.freeSlotCount,
            residentChunks: residentChunks,
            residentPoints: residentPoints,
            pinnedResidentChunks: pinnedResidentChunks,
            pendingUploadCount: pendingUploadCount,
            decodeInFlight: decodeInFlight,
            decodePendingRequests: decodePendingRequests,
            decodedPointsPerSecond: decodedPointsPerSecond,
            totalChunksUploaded: totalChunksUploaded,
            totalChunksEvicted: totalChunksEvicted,
            totalPendingDroppedByEviction: totalPendingDroppedByEviction,
            starvedTickCount: starvedTickCount,
            lastError: lastError
        )
    }

    /// Whether the driver's wanted set was clamped by the byte budget with
    /// frustum-visible candidates left out. Read alongside ``isCaughtUp`` to
    /// decide whether to raise ``setBudget(bytes:)`` for full coverage. See
    /// ``SwiftPDAL/StreamingPointCloudSource/residencyBudgetLimited``.
    public var residencyBudgetLimited: Bool { source.residencyBudgetLimited }

    /// GPU payload bytes for the whole file (`totalPoints × bytesPerPoint`) — the
    /// natural ceiling for a budget auto-raise: at this budget every node is
    /// wanted (whole-file mode), so ``residencyBudgetLimited`` can no longer be
    /// true. A pre-roll raising the budget to cover a multi-view union never
    /// needs to exceed it.
    public var totalPayloadBytes: Int {
        Int(source.info.totalPoints) * source.info.bytesPerPoint
    }

    /// Poll the driver's residency delta and upload decoded chunks into the slot
    /// pool. This is ``update(camera:viewport:)`` minus the camera submit, so a
    /// pre-roll can `submit(...)` once and then `pump()` repeatedly (yielding to
    /// let decode workers run) until ``isCaughtUp``.
    public func pump() {
        // Decode telemetry from the source's authoritative, thread-safe stats.
        // decodedPoints is monotonic → rate = Δpoints / Δt (EMA-smoothed).
        let stats = source.decodeStats()
        decodePendingRequests = stats.pendingRequests
        decodeInFlight = stats.inFlightDecodes
        let now = Date()
        if let last = lastDecodeSampleTime {
            let dt = now.timeIntervalSince(last)
            if dt > 0.05, stats.decodedPoints >= lastDecodedPoints {
                let inst = Double(stats.decodedPoints - lastDecodedPoints) / dt
                decodedPointsPerSecond = decodedPointsPerSecond == 0 ? inst : decodedPointsPerSecond * 0.8 + inst * 0.2
                lastDecodedPoints = stats.decodedPoints
                lastDecodeSampleTime = now
            }
        } else {
            lastDecodedPoints = stats.decodedPoints
            lastDecodeSampleTime = now
        }

        let delta = source.pollLatest()
        guard delta != nil || pendingCursor < pendingAdds.count else { return }

        var dirty = false
        if let delta {
            // Apply evictions. Pinned (coarse) nodes never appear in `removed`
            // — the source enforces that — so no special-casing here.
            var slotsToFree: [Int] = []
            for id in delta.removed {
                if let slots = slotsByChunk.removeValue(forKey: id) {
                    slotsToFree.append(contentsOf: slots)
                    totalChunksEvicted += 1
                }
            }
            if !delta.removed.isEmpty {
                if pendingCursor > 0 {
                    pendingAdds.removeFirst(pendingCursor)
                    pendingCursor = 0
                }
                // Drop still-pending chunks the source evicted before upload.
                let removedIDs = Set(delta.removed)
                let before = pendingAdds.count
                pendingAdds.removeAll { removedIDs.contains($0.id) }
                totalPendingDroppedByEviction += before - pendingAdds.count
            }
            if !slotsToFree.isEmpty {
                cloud.removeBatches(slots: slotsToFree, commit: false)
                dirty = true
            }
            pendingAdds.append(contentsOf: delta.added)
        }

        // Upload decoded chunks, de-duplicating against the resident set.
        var uploaded = 0
        var starvedThisTick = false
        while uploaded < maxChunkUploadsPerTick, pendingCursor < pendingAdds.count {
            let chunk = pendingAdds[pendingCursor]

            // Already resident (a pinned chunk the source re-admitted, or a dup).
            if slotsByChunk[chunk.id] != nil {
                pendingCursor += 1
                continue
            }

            let batches = chunk.batches.map(Self.toRasterBatch)
            if cloud.freeSlotCount < batches.count {
                if batches.count > cloud.batchCount {
                    lastError = "chunk needs \(batches.count) slots but pool capacity is \(cloud.batchCount); raise slotCapacity"
                    pendingCursor += 1
                    continue
                }
                // Pool full, no slot free this tick. Leave the chunk at the
                // cursor for a later tick (retried when an eviction frees a
                // slot or the camera moves). Record starvation for telemetry.
                lastError = "slot pool full (capacity \(cloud.batchCount)); raise slotCapacity or lower budget"
                starvedThisTick = true
                break
            }
            pendingCursor += 1
            let slots = cloud.addBatches(
                positionsXYZLow: chunk.xyzLow,
                positionsXYZMed: chunk.xyzMed,
                positionsXYZHigh: chunk.xyzHigh,
                colors: chunk.colors,
                levels: chunk.levels,
                batches: batches,
                commit: false
            )
            slotsByChunk[chunk.id] = slots
            totalChunksUploaded += 1
            dirty = true
            uploaded += 1
        }
        if starvedThisTick { starvedTickCount += 1 }
        uploadBlockedOnFullPool = starvedThisTick

        if pendingCursor == pendingAdds.count {
            pendingAdds.removeAll(keepingCapacity: true)
            pendingCursor = 0
        } else if pendingCursor > 64, pendingCursor * 2 > pendingAdds.count {
            pendingAdds.removeFirst(pendingCursor)
            pendingCursor = 0
        }

        if dirty { cloud.commitBatchUpdates() }

        residentChunks = slotsByChunk.count
        residentPoints = cloud.pointCount
        pinnedResidentChunks = coarsePinnedDepth < 0
            ? 0
            : slotsByChunk.keys.reduce(0) { $1.depth <= coarsePinnedDepth ? $0 + 1 : $0 }
    }

    /// Closes the underlying source. Idempotent on the source side.
    public func close() { pendingAdds.removeAll(); pendingCursor = 0; source.close() }

    private static func toRasterBatch(_ s: StreamingRasterBatch) -> RasterBatch {
        var b = RasterBatch(
            min: SIMD3<Float>(s.minX, s.minY, s.minZ),
            max: SIMD3<Float>(s.maxX, s.maxY, s.maxZ),
            numPoints: s.numPoints,
            firstPoint: s.firstPoint,
            fileIndex: s.fileIndex
        )
        b.state = s.state
        b.padding3 = s.padding3
        b.padding4 = s.padding4
        b.padding5 = s.padding5
        b.padding6 = s.padding6
        b.padding7 = s.padding7
        b.padding8 = s.padding8
        return b
    }
}
