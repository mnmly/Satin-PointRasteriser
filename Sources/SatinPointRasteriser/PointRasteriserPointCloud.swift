import Foundation
import Metal
import os
import Satin
import simd

/// A point cloud rendered by ``PointRasteriser``.
///
/// Owns two buffer families:
/// * **Source** — the quantized 30-bit layout (batches, `xyzLow`/`xyzMed`/
///   `xyzHigh`, packed colors, per-point LOD level) plus a ring-buffered per-file
///   transform table.
/// * **LOD** — the compacted SoA "LOD point cloud" the ``PointRasteriser``'s
///   LODSelect pass fills (positions / colors / source indices) plus a
///   stats+count buffer and indirect dispatch args for the raster passes.
///
/// ### Residency modes
/// * **Wholesale** (``init(context:packed:lodCapacity:label:)``) uploads a whole
///   ``PackedPointCloud`` once, contiguously — every point is resident and the
///   pack-order point index equals the source index (so picking maps directly
///   through `orderedPositions`).
/// * **Slot pool** (``init(context:slotCapacity:pointsPerBatch:files:lodCapacity:label:)``)
///   pre-allocates a fixed grid of `slotCapacity` batch slots, each
///   `pointsPerBatch` points wide, that a streaming source pages in/out via
///   ``addBatches(positionsXYZLow:positionsXYZMed:positionsXYZHigh:colors:levels:batches:commit:)``
///   / ``removeBatches(slots:commit:)``. Non-resident slots (``RasterBatch/state``
///   `== 0`) are skipped by LODSelect and cost one early-out.
///
/// ### Amortized LOD (double buffering)
/// When LOD selection is amortized (``PointRasteriserConfiguration/lodPointsPerFrame``
/// > 0) the cloud keeps **two** LOD buffer sets: LODSelect appends into the
/// **back** set across several frames (a "sweep") while every raster pass reads
/// the **front** set — the last *completed* sweep — so a partial sweep is never
/// rasterized. The sets swap only when a sweep finishes (2× LOD memory; the
/// second set is allocated lazily by ``ensureDoubleBuffered()``).
///
/// **Slot mutation vs. an in-flight sweep.** Adds/removes take effect immediately
/// on the source buffers and each batch joins the *next sweep boundary it is
/// eligible for*: a slot mutated ahead of the sweep cursor is picked up this
/// sweep, one behind waits for the next. This never corrupts the no-partial-sweep
/// guarantee (mutations only touch the source + **back** buffers; the **front**
/// changes only at a completing swap) and never decodes garbage (LODSelect gates
/// on `state`). The one accepted staleness: a batch removed behind the cursor
/// leaves its already-appended points in the back set until the next full sweep
/// re-selects without it. See the report for the rationale (chosen over
/// restart-on-mutation, which would starve amortized sweeps under a streaming
/// add rate, and over staging, which adds latency + a CPU blob buffer).
public final class PointRasteriserPointCloud: Object, @unchecked Sendable {
    /// Source point **capacity** (buffer sizing). Wholesale: the packed cloud's
    /// point count. Slot pool: `slotCapacity × pointsPerBatch`.
    ///
    /// - Note: For a wholesale cloud this equals the current resident point count
    ///   and can change across an in-place ``replacePackedPointCloud(_:)`` /
    ///   ``replacePackedPointCloud(packer:queue:positions:colors:count:)`` — the
    ///   backing buffers grow but never shrink, so `totalPoints` may be smaller
    ///   than the allocated capacity after a shrinking replace.
    public private(set) var totalPoints: Int
    /// Number of batch slots LODSelect dispatches over (wholesale: the current
    /// batch count; slot pool: `slotCapacity`). Changes across a wholesale
    /// in-place replace.
    public private(set) var batchCount: Int
    /// Capacity of each compacted LOD buffer set. Grows (never shrinks) to cover
    /// the current source point count so a full selection can never overflow.
    public private(set) var lodCapacity: Int
    /// Points per full batch slot (source-batch stride).
    public private(set) var pointsPerBatch: Int
    /// Whether this cloud is a mutable slot pool (vs. a wholesale upload).
    public let isSlotPool: Bool

    /// Allocated source-buffer capacity (points / batch slots). May exceed the
    /// logical ``totalPoints`` / ``batchCount`` after a shrinking wholesale
    /// replace — buffers grow but never shrink, so identity survives churn.
    private var sourcePointCapacity: Int
    private var sourceBatchCapacity: Int

    /// Object-space AABB of the source points. Set by the wholesale array init
    /// (from the packed cloud's bounds) and by the GPU-pack path (from the
    /// GPU-computed global bounds); handy for framing a camera. `.zero` until known.
    public private(set) var sourceBoundsMin: SIMD3<Float> = .zero
    public private(set) var sourceBoundsMax: SIMD3<Float> = .zero

    /// CPU mirror of every slot's ``RasterBatch``; the sweep planner reads its
    /// `numPoints`, and add/remove mutate it before flushing to `batchesBuffer`.
    private var batchMirror: [RasterBatch]
    /// LIFO of free slot indices (slot-pool mode). Newest-freed wins (warm cache).
    private var freeSlots: [Int]
    /// Inclusive slot range dirtied since the last flush (`lo > hi` = clean).
    private var dirtySlotLo = Int.max
    private var dirtySlotHi = -1

    /// Number of slots currently holding a resident batch.
    public private(set) var residentBatchCount: Int
    /// Sum of `numPoints` across resident batches.
    public private(set) var residentPointCount: Int
    /// Alias for ``residentPointCount`` (streaming-adapter parity).
    public var pointCount: Int { residentPointCount }
    /// Free slots available for ``addBatches``.
    public var freeSlotCount: Int { freeSlots.count }

    /// Monotonically increments whenever the GPU-visible batch content changes
    /// (a mutation flush). A consumer that caches per-content work — e.g.
    /// ``PointRasteriser``'s full-sweep LODSelect skip — compares this to detect
    /// "content literally unchanged". Any new code path that alters the resident
    /// point/batch content must ensure it flushes (or otherwise bumps this).
    public private(set) var contentGeneration: UInt64 = 0

    // MARK: - Per-point feature buffers

    /// Optional per-point displacement (`float3`, 16 B stride), added to positions
    /// in the depth + color passes when ``PointRasteriserConfiguration/applyDisplacement``
    /// is set. Auto-allocated by a ``DisplacementPass``.
    public var displacementBuffer: MTLBuffer?
    /// Optional per-point tint (`float4`, 16 B stride), mixed in the color pass
    /// when ``PointRasteriserConfiguration/applyTint`` is set. Auto-allocated by a ``TintPass``.
    public var tintBuffer: MTLBuffer?
    /// Previous frame's displacement per rendering camera, for motion-blur velocity.
    public var prevDisplacementBuffers: [ObjectIdentifier: MTLBuffer] = [:]

    /// Allocate a `totalPoints × stride(float3)` buffer for ``displacementBuffer``.
    public func makeDisplacementBuffer(storage: MTLStorageMode = .private, label: String? = nil) -> MTLBuffer? {
        let length = max(1, totalPoints) * MemoryLayout<SIMD3<Float>>.stride
        let buffer = context.device.makeBuffer(length: length, options: storage == .private ? .storageModePrivate : .storageModeShared)
        buffer?.label = label ?? "\(self.label).Displacement"
        return buffer
    }

    /// Allocate a `totalPoints × stride(float4)` buffer for ``tintBuffer``.
    public func makeTintBuffer(storage: MTLStorageMode = .private, label: String? = nil) -> MTLBuffer? {
        let length = max(1, totalPoints) * MemoryLayout<SIMD4<Float>>.stride
        let buffer = context.device.makeBuffer(length: length, options: storage == .private ? .storageModePrivate : .storageModeShared)
        buffer?.label = label ?? "\(self.label).Tint"
        return buffer
    }

    /// Per-file transforms (one for the fixture / single-file case).
    public private(set) var files: [RasterFile]

    // Source buffers.
    public private(set) var batchesBuffer: MTLBuffer?
    public private(set) var xyzLowBuffer: MTLBuffer?
    public private(set) var xyzMedBuffer: MTLBuffer?
    public private(set) var xyzHighBuffer: MTLBuffer?
    public private(set) var colorsBuffer: MTLBuffer?
    public private(set) var levelsBuffer: MTLBuffer?

    // Ring-buffered per-file transforms.
    public private(set) var filesBuffer: MTLBuffer?
    public private(set) var filesBufferOffset: Int = 0
    private var filesSlotIndex: Int = -1
    private var filesSlotStride: Int = 0
    /// Files ring depth: Satin triple-buffering × 2 (stereo headroom).
    public static let filesBufferSlotCount = Satin.maxBuffersInFlight * 2

    // MARK: - LOD buffer sets

    private struct LODSet {
        let positions: MTLBuffer
        let colors: MTLBuffer
        let sourceIndices: MTLBuffer
        let stats: MTLBuffer
        let dispatchArgs: MTLBuffer
    }

    private var lodSetA: LODSet!
    private var lodSetB: LODSet?
    private var frontIsA = true

    /// Next source batch to process in the in-flight sweep (0 = sweep start).
    public private(set) var sweepCursor = 0
    /// Whether a completed sweep has ever populated the front set.
    public private(set) var hasFrontData = false

    private var front: LODSet { frontIsA ? lodSetA : (lodSetB ?? lodSetA) }
    private var back: LODSet {
        guard let b = lodSetB else { return lodSetA }
        return frontIsA ? b : lodSetA
    }

    /// `true` once the second LOD set has been allocated (amortization enabled).
    public var isDoubleBuffered: Bool { lodSetB != nil }

    // Front-set accessors (raster passes read these).
    public var frontLodPositionsBuffer: MTLBuffer? { front.positions }
    public var frontLodColorsBuffer: MTLBuffer? { front.colors }
    public var frontLodSourceIndicesBuffer: MTLBuffer? { front.sourceIndices }
    public var frontLodStatsBuffer: MTLBuffer? { front.stats }
    public var frontLodDispatchArgsBuffer: MTLBuffer? { front.dispatchArgs }

    // Back-set accessors (LODSelect + finalize write these).
    public var backLodPositionsBuffer: MTLBuffer? { back.positions }
    public var backLodColorsBuffer: MTLBuffer? { back.colors }
    public var backLodSourceIndicesBuffer: MTLBuffer? { back.sourceIndices }
    public var backLodStatsBuffer: MTLBuffer? { back.stats }
    public var backLodDispatchArgsBuffer: MTLBuffer? { back.dispatchArgs }

    // MARK: - Wholesale init

    /// Creates a cloud from a wholesale packed point cloud (contiguous, fully
    /// resident). Not mutable — use the slot-pool init for streaming.
    /// - Parameter lodCapacity: cap on each compacted LOD buffer set. Defaults to
    ///   the cloud's full source point count so **overflow is impossible by
    ///   default** (every survivable point fits). Pass a smaller value only as a
    ///   deliberate memory saver — survivors beyond it are dropped, scattering
    ///   batch-shaped holes across the image. Memory math: the LOD set is 20 B
    ///   per point (SoA: 12 B position + 4 B color + 4 B index), doubled while
    ///   amortization double-buffers — e.g. 44M points ≈ 880 MB, or ≈ 1.76 GB
    ///   amortized.
    public init(
        context: Context,
        packed: PackedPointCloud,
        lodCapacity: Int? = nil,
        label: String = "PointRasteriserPointCloud"
    ) {
        self.isSlotPool = false
        self.totalPoints = packed.pointCount
        self.batchCount = packed.batchCount
        self.sourcePointCapacity = packed.pointCount
        self.sourceBatchCapacity = packed.batchCount
        self.lodCapacity = max(1, min(packed.pointCount, lodCapacity ?? packed.pointCount))
        self.pointsPerBatch = Int(packed.batches.first?.numPoints ?? UInt32(max(1, packed.pointCount)))
        self.batchMirror = packed.batches
        self.freeSlots = []
        self.residentBatchCount = packed.batchCount
        self.residentPointCount = packed.pointCount
        self.files = packed.files.isEmpty ? [RasterFile()] : packed.files
        super.init(context: context, label: label)
        allocateWholesaleSourceBuffers(packed: packed)
        sourceBoundsMin = packed.boundsMin
        sourceBoundsMax = packed.boundsMax
        lodSetA = makeLODSet(tag: "A")
        rebuildFilesBuffer()
    }

    // MARK: - GPU-pack wholesale init

    /// Creates a wholesale cloud with **empty** source buffers sized for `count`
    /// points at `pointsPerBatch` per batch, to be filled in-place by a
    /// ``GPUPacker`` (no CPU pack, no Swift arrays). After the pack command
    /// buffer completes, call ``adoptGPUBatchBounds(boundsMin:boundsMax:)`` to
    /// sync the CPU batch mirror. Prefer the ``gpuPacked(context:packer:queue:positions:colors:count:pointsPerBatch:lodCapacity:shuffle:label:)``
    /// factory, which wires the whole flow up.
    ///
    /// - Parameter lodCapacity: cap on each compacted LOD buffer set; defaults to
    ///   `count` so overflow is impossible (see the array init for memory math).
    public init(
        context: Context,
        gpuPackPointCount count: Int,
        pointsPerBatch: Int = pointRasteriserThreadsPerGroup * 80,
        lodCapacity: Int? = nil,
        label: String = "PointRasteriserPointCloud"
    ) {
        precondition(count > 0 && pointsPerBatch > 0, "count and pointsPerBatch must be positive")
        precondition(pointsPerBatch <= 65535, "pointsPerBatch must fit the uint16 LOD prefix counts")
        self.isSlotPool = false
        self.totalPoints = count
        self.pointsPerBatch = pointsPerBatch
        let batches = max(1, (count + pointsPerBatch - 1) / pointsPerBatch)
        self.batchCount = batches
        self.sourcePointCapacity = count
        self.sourceBatchCapacity = batches
        self.lodCapacity = max(1, min(count, lodCapacity ?? count))
        // Placeholder mirror (right numPoints/firstPoint so planLODChunk works
        // even before adopt); adoptGPUBatchBounds() refreshes AABBs post-pack.
        var mirror: [RasterBatch] = []
        mirror.reserveCapacity(batches)
        for b in 0 ..< batches {
            let n = min(pointsPerBatch, count - b * pointsPerBatch)
            mirror.append(RasterBatch(min: .zero, max: .zero, numPoints: UInt32(n), firstPoint: UInt32(b * pointsPerBatch), fileIndex: 0))
        }
        self.batchMirror = mirror
        self.freeSlots = []
        self.residentBatchCount = batches
        self.residentPointCount = count
        self.files = [RasterFile()]
        super.init(context: context, label: label)
        allocatePoolSourceBuffers() // same contiguous sizing; GPUPacker writes into these
        lodSetA = makeLODSet(tag: "A")
        rebuildFilesBuffer()
    }

    /// Sync the CPU batch mirror + source bounds from the GPU after a
    /// ``GPUPacker`` pack completes: reads every ``RasterBatch`` the pack wrote
    /// into ``batchesBuffer`` (AABBs, numPoints, cumulative LOD counts) and
    /// records the GPU-computed global bounds.
    public func adoptGPUBatchBounds(boundsMin: SIMD3<Float>, boundsMax: SIMD3<Float>) {
        sourceBoundsMin = boundsMin
        sourceBoundsMax = boundsMax
        guard let batchesBuffer else { return }
        let stride = MemoryLayout<RasterBatch>.stride
        let n = min(batchCount, batchesBuffer.length / stride)
        guard n > 0 else { return }
        let ptr = batchesBuffer.contents().bindMemory(to: RasterBatch.self, capacity: n)
        var points = 0
        for i in 0 ..< n {
            batchMirror[i] = ptr[i]
            points += Int(ptr[i].numPoints)
        }
        residentBatchCount = n
        residentPointCount = points
        // The GPUPacker wrote batch/point content straight into the source
        // buffers (bypassing flushBatchMirror), so invalidate any content-caching
        // consumer (e.g. PointRasteriser's full-sweep LODSelect skip) explicitly.
        contentGeneration &+= 1
    }

    // MARK: - In-place wholesale replace

    /// Replace this wholesale cloud's entire contents **in place** — same object
    /// identity, so anything holding this instance (a scene node, a selection /
    /// picking binding) survives the swap. This is the mutation story for
    /// wholesale clouds; slot pools mutate via ``addBatches`` / ``removeBatches``.
    ///
    /// Source and LOD buffers **grow to fit but never shrink**, so a smaller
    /// replace reuses the existing allocation and only the first ``batchCount``
    /// slots render (no stale points bleed through from the previous, larger
    /// content). On every replace:
    /// - ``contentGeneration`` bumps, invalidating ``PointRasteriser``'s
    ///   full-sweep LODSelect-skip cache so the new content is re-selected.
    /// - ``lodCapacity`` grows to the new source point count, so a full selection
    ///   can never overflow (never reintroduces silent truncation).
    /// - The amortized LOD sweep resets to a fresh **full** sweep: the next frame
    ///   publishes only the new content and completes in one frame, so the front
    ///   set is never a partial or mixed (old+new) sweep.
    /// - If the point count changed, ``displacementBuffer`` / ``tintBuffer`` /
    ///   per-camera prev-displacement buffers are dropped; the Displacement/Tint
    ///   passes re-allocate them zeroed on the next encode, so user displacement/
    ///   tint kernels see a fresh zeroed buffer after a resizing replace.
    ///
    /// - Precondition: not a slot pool.
    public func replacePackedPointCloud(_ packed: PackedPointCloud) {
        precondition(!isSlotPool, "replacePackedPointCloud requires a wholesale cloud; slot pools use addBatches/removeBatches")
        let newPPB = Int(packed.batches.first?.numPoints ?? UInt32(max(1, packed.pointCount)))
        beginReplace(pointCount: packed.pointCount, batchCount: max(1, packed.batchCount), pointsPerBatch: max(1, newPPB))
        files = packed.files.isEmpty ? [RasterFile()] : packed.files
        rebuildFilesBuffer()
        uploadPacked(packed)
        batchMirror = packed.batches.isEmpty
            ? [RasterBatch(min: .zero, max: .zero, numPoints: 0, firstPoint: 0, fileIndex: 0, state: 0)]
            : packed.batches
        residentBatchCount = packed.batchCount
        residentPointCount = packed.pointCount
        sourceBoundsMin = packed.boundsMin
        sourceBoundsMax = packed.boundsMax
        contentGeneration &+= 1
    }

    /// GPU in-place replace: pack `positions`/`colors` (caller-owned, shared
    /// buffers — e.g. a GeometrySet's own Position/Color buffers) straight into
    /// this cloud's source buffers via `packer`, on a fresh command buffer
    /// committed and waited on before returning. Same identity and invariant
    /// guarantees as ``replacePackedPointCloud(_:)``, and decode-identical to it
    /// for the same input.
    ///
    /// - Precondition: not a slot pool.
    public func replacePackedPointCloud(
        packer: GPUPacker,
        queue: MTLCommandQueue,
        positions: MTLBuffer,
        colors: MTLBuffer,
        count: Int,
        shuffle: Bool = true
    ) {
        precondition(!isSlotPool, "replacePackedPointCloud requires a wholesale cloud; slot pools use addBatches/removeBatches")
        let ppb = pointsPerBatch
        let clampedCount = max(0, count)
        let newBatches = max(1, (clampedCount + ppb - 1) / ppb)
        beginReplace(pointCount: clampedCount, batchCount: newBatches, pointsPerBatch: ppb)
        files = [RasterFile()]
        rebuildFilesBuffer()
        // Fresh placeholder mirror (right numPoints/firstPoint so planLODChunk
        // works even before adopt); adoptGPUBatchBounds refreshes AABBs post-pack.
        batchMirror = (0 ..< newBatches).map { b in
            let n = max(0, min(ppb, clampedCount - b * ppb))
            return RasterBatch(min: .zero, max: .zero, numPoints: UInt32(n), firstPoint: UInt32(b * ppb), fileIndex: 0)
        }
        residentBatchCount = newBatches
        residentPointCount = clampedCount
        guard clampedCount > 0, let commandBuffer = queue.makeCommandBuffer() else {
            contentGeneration &+= 1
            return
        }
        commandBuffer.label = "\(label).replacePackedPointCloud"
        packer.pack(positions: positions, colors: colors, count: clampedCount, shuffle: shuffle, into: self, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let bounds = packer.lastBounds() ?? (min: SIMD3<Float>.zero, max: SIMD3<Float>.zero)
        adoptGPUBatchBounds(boundsMin: bounds.min, boundsMax: bounds.max) // bumps contentGeneration
    }

    /// Grow source + LOD buffers to fit the new content (never shrinks), retarget
    /// the logical sizes, drop resize-invalidated feature buffers, and reset the
    /// amortized sweep to a fresh full sweep. Content upload and the
    /// ``contentGeneration`` bump are the caller's responsibility.
    private func beginReplace(pointCount: Int, batchCount newBatchCount: Int, pointsPerBatch newPPB: Int) {
        let sizeChanged = pointCount != totalPoints
        if pointCount > sourcePointCapacity || newBatchCount > sourceBatchCapacity {
            reallocateSourceBuffers(
                pointCapacity: max(sourcePointCapacity, pointCount),
                batchCapacity: max(sourceBatchCapacity, newBatchCount)
            )
        }
        if pointCount > lodCapacity {
            lodCapacity = pointCount
            lodSetA = makeLODSet(tag: "A")
            if lodSetB != nil { lodSetB = makeLODSet(tag: "B") }
        }
        totalPoints = pointCount
        batchCount = newBatchCount
        pointsPerBatch = newPPB
        if sizeChanged {
            displacementBuffer = nil
            tintBuffer = nil
            prevDisplacementBuffers.removeAll()
        }
        resetSweepStateForReplace()
    }

    /// Reset amortized-sweep + telemetry so a replaced cloud never renders a
    /// stale or partial front: cursor to 0, front data invalidated (forcing a
    /// full sweep next frame that completes before it is read), both LOD sets'
    /// stats zeroed.
    private func resetSweepStateForReplace() {
        sweepCursor = 0
        hasFrontData = false
        frontIsA = true
        overflowWarned = false
        lodSetA.stats.contents().initializeMemory(as: UInt32.self, repeating: 0, count: 3)
        if let b = lodSetB { b.stats.contents().initializeMemory(as: UInt32.self, repeating: 0, count: 3) }
    }

    /// Reallocate the source buffers to the given capacity (grow-only in
    /// practice). Buffer identity changes, but the cloud's does not — the
    /// rasteriser rebinds `cloud.*Buffer` every encode.
    private func reallocateSourceBuffers(pointCapacity: Int, batchCapacity: Int) {
        xyzLowBuffer = makeEmptyBuffer(length: pointCapacity * 4, label: "\(label).XYZLow")
        xyzMedBuffer = makeEmptyBuffer(length: pointCapacity * 4, label: "\(label).XYZMed")
        xyzHighBuffer = makeEmptyBuffer(length: pointCapacity * 4, label: "\(label).XYZHigh")
        colorsBuffer = makeEmptyBuffer(length: pointCapacity * 4, label: "\(label).Colors")
        levelsBuffer = makeEmptyBuffer(length: pointCapacity, label: "\(label).Levels")
        batchesBuffer = makeEmptyBuffer(length: batchCapacity * MemoryLayout<RasterBatch>.stride, label: "\(label).Batches")
        sourcePointCapacity = pointCapacity
        sourceBatchCapacity = batchCapacity
    }

    /// Copy a packed cloud's arrays into the (already grown) source buffers.
    private func uploadPacked(_ packed: PackedPointCloud) {
        copyArray(packed.xyzLow, into: xyzLowBuffer)
        copyArray(packed.xyzMed, into: xyzMedBuffer)
        copyArray(packed.xyzHigh, into: xyzHighBuffer)
        copyArray(packed.colors, into: colorsBuffer)
        copyArray(packed.levels, into: levelsBuffer)
        copyArray(packed.batches, into: batchesBuffer)
    }

    private func copyArray<T>(_ array: [T], into buffer: MTLBuffer?) {
        guard let buffer, !array.isEmpty else { return }
        let byteCount = array.count * MemoryLayout<T>.stride
        precondition(byteCount <= buffer.length, "replace: source array exceeds buffer capacity")
        array.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: byteCount) }
    }

    // MARK: - Slot-pool init

    /// Creates an empty slot pool: `slotCapacity` batch slots, each
    /// `pointsPerBatch` points wide, that a streaming source pages in/out.
    /// - Parameter lodCapacity: cap on each LOD buffer set. Defaults to the pool's
    ///   full point capacity (`slotCapacity × pointsPerBatch`) so overflow is
    ///   impossible when every slot is resident. Pass a smaller value only as a
    ///   deliberate memory saver (see the wholesale init for the memory math).
    public init(
        context: Context,
        slotCapacity: Int,
        pointsPerBatch: Int,
        files: [RasterFile]? = nil,
        lodCapacity: Int? = nil,
        label: String = "PointRasteriserPointCloud"
    ) {
        precondition(slotCapacity > 0 && pointsPerBatch > 0, "slotCapacity and pointsPerBatch must be positive")
        self.isSlotPool = true
        self.batchCount = slotCapacity
        self.pointsPerBatch = pointsPerBatch
        self.totalPoints = slotCapacity * pointsPerBatch
        self.sourcePointCapacity = slotCapacity * pointsPerBatch
        self.sourceBatchCapacity = slotCapacity
        self.lodCapacity = max(1, min(slotCapacity * pointsPerBatch, lodCapacity ?? (slotCapacity * pointsPerBatch)))
        self.batchMirror = Array(repeating: RasterBatch(min: .zero, max: .zero, numPoints: 0, firstPoint: 0, fileIndex: 0, state: 0), count: slotCapacity)
        self.freeSlots = Array((0 ..< slotCapacity).reversed())
        self.residentBatchCount = 0
        self.residentPointCount = 0
        self.files = files ?? [RasterFile()]
        super.init(context: context, label: label)
        allocatePoolSourceBuffers()
        markAllSlotsDirty()
        flushBatchMirror()
        lodSetA = makeLODSet(tag: "A")
        rebuildFilesBuffer()
    }

    public required init(from decoder: any Decoder) throws {
        fatalError("init(from:) has not been implemented")
    }

    // MARK: - Slot-pool residency

    /// Reserve a **contiguous** run of `count` free slots (all non-resident),
    /// returning the first slot index or `nil` if no such run exists. The slots
    /// are removed from the free list and marked resident provisionally; a
    /// subsequent write (e.g. a GPU pack) fills them and ``commitBatchUpdates()``
    /// publishes the metadata.
    @discardableResult
    public func reserveContiguousSlots(count n: Int) -> Int? {
        precondition(isSlotPool, "reserveContiguousSlots requires a slot-pool cloud")
        guard n > 0, n <= batchCount else { return nil }
        var s = 0
        while s + n <= batchCount {
            var k = 0
            while k < n, batchMirror[s + k].state == 0 { k += 1 }
            if k == n {
                let reserved = Set(s ..< (s + n))
                freeSlots.removeAll { reserved.contains($0) }
                for slot in s ..< (s + n) { batchMirror[slot].state = 1 }
                markSlotsDirty(s ..< (s + n))
                return s
            }
            s += k + 1
        }
        return nil
    }

    /// Upload one or more batches into free slots. Each input `batches[i]`
    /// describes a contiguous run in the position/color/level blobs starting at
    /// `batches[i].firstPoint`; the run is copied into a fresh slot and its
    /// `firstPoint` rebased to `slot × pointsPerBatch`.
    ///
    /// - Parameters:
    ///   - positionsXYZLow/Med/High: `UInt32`-per-point axis packs.
    ///   - colors: packed RGBA `UInt32`-per-point.
    ///   - levels: `UInt8`-per-point LOD level.
    ///   - batches: metadata (one per chunk); `firstPoint`/`state` are rewritten.
    ///   - commit: re-upload the GPU-visible `batchesBuffer` before returning
    ///     (pass `false` for a batch of calls, then ``commitBatchUpdates()``).
    /// - Returns: the slot index chosen for each input batch.
    /// - Precondition: `freeSlotCount >= batches.count`.
    @discardableResult
    public func addBatches(
        positionsXYZLow: Data,
        positionsXYZMed: Data,
        positionsXYZHigh: Data,
        colors: Data,
        levels: Data,
        batches: [RasterBatch],
        commit: Bool = true
    ) -> [Int] {
        precondition(isSlotPool, "addBatches requires a slot-pool cloud")
        precondition(batches.count <= freeSlots.count, "addBatches: not enough free slots (have \(freeSlots.count), need \(batches.count))")
        guard !batches.isEmpty else { return [] }

        var assigned: [Int] = []
        assigned.reserveCapacity(batches.count)
        for batch in batches {
            let slot = freeSlots.removeLast()
            assigned.append(slot)

            let dstFirst = slot * pointsPerBatch
            let srcFirst = Int(batch.firstPoint)
            let count = Int(batch.numPoints)
            precondition(count <= pointsPerBatch, "batch numPoints (\(count)) exceeds slot capacity (\(pointsPerBatch))")

            copySlice(positionsXYZLow, srcOffsetPoints: srcFirst, dstOffsetPoints: dstFirst, count: count, stride: 4, into: xyzLowBuffer)
            copySlice(positionsXYZMed, srcOffsetPoints: srcFirst, dstOffsetPoints: dstFirst, count: count, stride: 4, into: xyzMedBuffer)
            copySlice(positionsXYZHigh, srcOffsetPoints: srcFirst, dstOffsetPoints: dstFirst, count: count, stride: 4, into: xyzHighBuffer)
            copySlice(colors, srcOffsetPoints: srcFirst, dstOffsetPoints: dstFirst, count: count, stride: 4, into: colorsBuffer)
            copySlice(levels, srcOffsetPoints: srcFirst, dstOffsetPoints: dstFirst, count: count, stride: 1, into: levelsBuffer)

            var slotBatch = batch
            slotBatch.firstPoint = UInt32(dstFirst)
            slotBatch.state = 1
            batchMirror[slot] = slotBatch
            markSlotDirty(slot)
            residentBatchCount += 1
            residentPointCount += count
        }
        if commit { flushBatchMirror() }
        return assigned
    }

    /// Free the given slots. Their ``RasterBatch/state`` flips to `0` so LODSelect
    /// skips them; the slots return to the free list for reuse.
    public func removeBatches(slots: [Int], commit: Bool = true) {
        precondition(isSlotPool, "removeBatches requires a slot-pool cloud")
        guard !slots.isEmpty else { return }
        for slot in slots {
            precondition(slot >= 0 && slot < batchCount, "removeBatches: slot \(slot) out of range")
            if batchMirror[slot].state == 0 { continue }
            residentPointCount -= Int(batchMirror[slot].numPoints)
            residentBatchCount -= 1
            batchMirror[slot].state = 0
            batchMirror[slot].numPoints = 0
            markSlotDirty(slot)
            freeSlots.append(slot)
        }
        if commit { flushBatchMirror() }
    }

    /// Mark every slot empty (cheap; keeps the GPU buffers).
    public func clearAllBatches() {
        precondition(isSlotPool, "clearAllBatches requires a slot-pool cloud")
        for slot in 0 ..< batchCount {
            batchMirror[slot].state = 0
            batchMirror[slot].numPoints = 0
        }
        freeSlots = Array((0 ..< batchCount).reversed())
        residentBatchCount = 0
        residentPointCount = 0
        markAllSlotsDirty()
        flushBatchMirror()
    }

    /// Publish `commit: false` add/remove calls to the GPU-visible `batchesBuffer`.
    public func commitBatchUpdates() {
        flushBatchMirror()
    }

    // MARK: - Stats (CPU-readable, no GPU stall)

    /// Number of points actually compacted into the front (published) LOD sweep
    /// — the clamped count the raster passes draw (`min(survivors, lodCapacity)`).
    /// When ``lodOverflowed`` this equals ``lodCapacity``.
    public var lodCount: Int {
        Int(front.stats.contents().load(fromByteOffset: 2 * MemoryLayout<UInt32>.stride, as: UInt32.self))
    }

    /// Points the front sweep dropped for want of LOD capacity (0 when the whole
    /// selection fit). A positive value means the image has scattered
    /// batch-shaped holes — raise `lodCapacity` at cloud init.
    public var lodOverflow: Int {
        Int(front.stats.contents().load(fromByteOffset: MemoryLayout<UInt32>.stride, as: UInt32.self))
    }

    /// `true` if the front sweep overflowed its LOD capacity.
    public var lodOverflowed: Bool { lodOverflow > 0 }

    /// os_log a capacity-overflow warning at most once per cloud (after a sweep
    /// completes and drains). Called each encode by ``PointRasteriser``.
    func logOverflowWarningIfNeeded() {
        guard !overflowWarned else { return }
        let dropped = lodOverflow
        guard dropped > 0 else { return }
        overflowWarned = true
        os_log(
            "%{public}@: LOD capacity exceeded — dropped ~%d points (capacity %d). Increase lodCapacity at cloud init to avoid scattered black holes.",
            log: Self.overflowLog, type: .error, label, dropped, lodCapacity
        )
    }

    private var overflowWarned = false
    private static let overflowLog = OSLog(subsystem: "SatinPointRasteriser", category: "LODCapacity")

    /// Progress of the in-flight back sweep in `[0, 1]` (1 = idle / complete).
    public var lodSweepProgress: Float {
        batchCount > 0 ? Float(sweepCursor) / Float(batchCount) : 1
    }

    /// Pack-order source index for a front-set LOD point (picking).
    public func frontLodSourceIndex(at lodIndex: Int) -> UInt32? {
        guard lodIndex >= 0, lodIndex < lodCapacity else { return nil }
        return front.sourceIndices.contents().load(fromByteOffset: lodIndex * MemoryLayout<UInt32>.stride, as: UInt32.self)
    }

    // MARK: - Amortized sweep lifecycle

    /// A per-frame slice of the LOD sweep.
    public struct LODChunk: Sendable {
        public let firstBatch: Int
        public let batchCount: Int
        public let startsSweep: Bool
        public let completesSweep: Bool
    }

    /// Plan this frame's LOD chunk given a per-frame point budget (0 = full
    /// sweep). Budget is approximate at batch granularity; the very first sweep
    /// always runs full so a freshly loaded cloud is never blank.
    public func planLODChunk(pointBudget: Int) -> LODChunk {
        let amortize = pointBudget > 0 && hasFrontData
        let firstBatch = sweepCursor
        var count = 0
        if !amortize {
            count = batchCount - firstBatch
        } else {
            var points = 0
            while firstBatch + count < batchCount, count == 0 || points < pointBudget {
                points += Int(batchMirror[firstBatch + count].numPoints)
                count += 1
            }
        }
        return LODChunk(
            firstBatch: firstBatch,
            batchCount: count,
            startsSweep: firstBatch == 0,
            completesSweep: firstBatch + count >= batchCount
        )
    }

    /// Advance the sweep cursor after encoding `chunk`; swap on completion.
    public func advanceSweep(after chunk: LODChunk) {
        if chunk.completesSweep {
            sweepCursor = 0
            hasFrontData = true
            if isDoubleBuffered { frontIsA.toggle() }
        } else {
            sweepCursor = chunk.firstBatch + chunk.batchCount
        }
    }

    /// Abandon the in-flight back sweep and restart from batch 0 next frame.
    public func restartLODSweep() {
        sweepCursor = 0
    }

    /// Lazily allocate the second LOD set so sweeps can double-buffer.
    public func ensureDoubleBuffered() {
        guard lodSetB == nil else { return }
        lodSetB = makeLODSet(tag: "B")
    }

    // MARK: - Files ring

    /// Per-frame per-camera transform push.
    public func updateFiles(viewProjection: simd_float4x4, modelMatrix: simd_float4x4, prevViewProjection: simd_float4x4? = nil) {
        guard !files.isEmpty, let filesBuffer else { return }
        var snapshot = files
        for index in snapshot.indices {
            let world = modelMatrix * files[index].world
            snapshot[index].transform = viewProjection * world
            snapshot[index].transformFrustum = viewProjection * world
            snapshot[index].world = world
            snapshot[index].prevTransform = (prevViewProjection ?? viewProjection) * world
        }

        filesSlotIndex = (filesSlotIndex + 1) % Self.filesBufferSlotCount
        filesBufferOffset = filesSlotIndex * filesSlotStride

        let byteCount = min(filesSlotStride, snapshot.count * MemoryLayout<RasterFile>.stride)
        snapshot.withUnsafeBytes { bytes in
            filesBuffer.contents().advanced(by: filesBufferOffset).copyMemory(from: bytes.baseAddress!, byteCount: byteCount)
        }
    }

    // MARK: - Allocation

    private func allocateWholesaleSourceBuffers(packed: PackedPointCloud) {
        batchesBuffer = makeBuffer(bytes: packed.batches, label: "\(label).Batches")
        xyzLowBuffer = makeBuffer(bytes: packed.xyzLow, label: "\(label).XYZLow")
        xyzMedBuffer = makeBuffer(bytes: packed.xyzMed, label: "\(label).XYZMed")
        xyzHighBuffer = makeBuffer(bytes: packed.xyzHigh, label: "\(label).XYZHigh")
        colorsBuffer = makeBuffer(bytes: packed.colors, label: "\(label).Colors")
        levelsBuffer = makeBuffer(bytes: packed.levels, label: "\(label).Levels")
    }

    private func allocatePoolSourceBuffers() {
        let points = totalPoints
        xyzLowBuffer = makeEmptyBuffer(length: points * 4, label: "\(label).XYZLow")
        xyzMedBuffer = makeEmptyBuffer(length: points * 4, label: "\(label).XYZMed")
        xyzHighBuffer = makeEmptyBuffer(length: points * 4, label: "\(label).XYZHigh")
        colorsBuffer = makeEmptyBuffer(length: points * 4, label: "\(label).Colors")
        levelsBuffer = makeEmptyBuffer(length: points, label: "\(label).Levels")
        batchesBuffer = makeEmptyBuffer(length: batchCount * MemoryLayout<RasterBatch>.stride, label: "\(label).Batches")
    }

    private func makeLODSet(tag: String) -> LODSet {
        let cap = lodCapacity
        let positions = context.device.makeBuffer(length: cap * LODCloudLayout.positionStride, options: .storageModePrivate)!
        positions.label = "\(label).LODPositions\(tag)"
        let colors = context.device.makeBuffer(length: cap * LODCloudLayout.colorStride, options: .storageModePrivate)!
        colors.label = "\(label).LODColors\(tag)"
        let sourceIndices = context.device.makeBuffer(length: cap * LODCloudLayout.sourceIndexStride, options: .storageModeShared)!
        sourceIndices.label = "\(label).LODSourceIndices\(tag)"
        let stats = context.device.makeBuffer(length: 3 * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        stats.label = "\(label).LODStats\(tag)"
        stats.contents().initializeMemory(as: UInt32.self, repeating: 0, count: 3)
        let dispatchArgs = context.device.makeBuffer(length: MemoryLayout<CRDispatchArgs>.stride, options: .storageModeShared)!
        dispatchArgs.label = "\(label).LODDispatchArgs\(tag)"
        dispatchArgs.contents().initializeMemory(as: UInt32.self, repeating: 0, count: 3)
        return LODSet(positions: positions, colors: colors, sourceIndices: sourceIndices, stats: stats, dispatchArgs: dispatchArgs)
    }

    private func rebuildFilesBuffer() {
        guard !files.isEmpty else {
            filesBuffer = nil; filesBufferOffset = 0; filesSlotIndex = -1; filesSlotStride = 0
            return
        }
        let perSlot = files.count * MemoryLayout<RasterFile>.stride
        filesSlotStride = perSlot
        let totalLength = perSlot * Self.filesBufferSlotCount
        guard let buffer = context.device.makeBuffer(length: totalLength, options: .storageModeShared) else {
            filesBuffer = nil; return
        }
        buffer.label = "\(label).Files"
        files.withUnsafeBytes { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: perSlot)
        }
        filesBuffer = buffer
        filesSlotIndex = -1
        filesBufferOffset = 0
    }

    private func makeBuffer<T>(bytes: [T], label: String) -> MTLBuffer? {
        guard !bytes.isEmpty else { return nil }
        let length = bytes.count * MemoryLayout<T>.stride
        let buffer = bytes.withUnsafeBytes { context.device.makeBuffer(bytes: $0.baseAddress!, length: length, options: .storageModeShared) }
        buffer?.label = label
        return buffer
    }

    private func makeEmptyBuffer(length: Int, label: String) -> MTLBuffer? {
        guard length > 0 else { return nil }
        let buffer = context.device.makeBuffer(length: length, options: .storageModeShared)
        buffer?.label = label
        return buffer
    }

    // MARK: - Batch mirror plumbing

    private func markSlotDirty(_ slot: Int) {
        if slot < dirtySlotLo { dirtySlotLo = slot }
        if slot > dirtySlotHi { dirtySlotHi = slot }
    }

    private func markSlotsDirty(_ range: Range<Int>) {
        guard !range.isEmpty else { return }
        markSlotDirty(range.lowerBound)
        markSlotDirty(range.upperBound - 1)
    }

    private func markAllSlotsDirty() { markSlotsDirty(0 ..< batchMirror.count) }

    private func flushBatchMirror() {
        guard let batchesBuffer, dirtySlotLo <= dirtySlotHi else { return }
        let stride = MemoryLayout<RasterBatch>.stride
        let lo = max(0, dirtySlotLo)
        let hi = min(dirtySlotHi, batchMirror.count - 1, batchesBuffer.length / stride - 1)
        dirtySlotLo = .max
        dirtySlotHi = -1
        guard lo <= hi else { return }
        let byteOffset = lo * stride
        let byteCount = (hi - lo + 1) * stride
        batchMirror.withUnsafeBytes { bytes in
            batchesBuffer.contents().advanced(by: byteOffset).copyMemory(from: bytes.baseAddress!.advanced(by: byteOffset), byteCount: byteCount)
        }
        // GPU-visible content changed — invalidate any content-caching consumer
        // (all CPU mutators funnel their commit through here).
        contentGeneration &+= 1
    }

    private func copySlice(_ source: Data, srcOffsetPoints: Int, dstOffsetPoints: Int, count: Int, stride: Int, into buffer: MTLBuffer?) {
        guard let buffer, count > 0 else { return }
        let byteCount = count * stride
        let srcByteOffset = srcOffsetPoints * stride
        let dstByteOffset = dstOffsetPoints * stride
        precondition(srcByteOffset + byteCount <= source.count, "addBatches: source data too small for batch")
        precondition(dstByteOffset + byteCount <= buffer.length, "addBatches: dest buffer too small")
        source.withUnsafeBytes { srcRaw in
            let src = srcRaw.baseAddress!.advanced(by: srcByteOffset)
            buffer.contents().advanced(by: dstByteOffset).copyMemory(from: src, byteCount: byteCount)
        }
    }
}

// MARK: - GPU-pack factory

public extension PointRasteriserPointCloud {
    /// Build a wholesale cloud by packing GPU-resident `positions`/`colors`
    /// buffers with `packer`, on a fresh command buffer that is committed and
    /// waited on before returning. The GPU replacement for
    /// ``init(context:packed:lodCapacity:label:)`` — no CPU pack, no Swift
    /// arrays. Mirrors Satin-ComputeRasteriser's `replacePackedPointCloud`.
    ///
    /// - Parameters:
    ///   - positions: `float3` positions (16-byte stride, `.storageModeShared`).
    ///   - colors: `float4` RGBA in `[0,1]` (16-byte stride, shared).
    ///   - count: number of points.
    ///   - shuffle: apply the whole-batch shuffle (default `true`, matching the CPU path).
    static func gpuPacked(
        context: Context,
        packer: GPUPacker,
        queue: MTLCommandQueue,
        positions: MTLBuffer,
        colors: MTLBuffer,
        count: Int,
        pointsPerBatch: Int = pointRasteriserThreadsPerGroup * 80,
        lodCapacity: Int? = nil,
        shuffle: Bool = true,
        label: String = "PointRasteriserPointCloud"
    ) -> PointRasteriserPointCloud {
        // Mint an empty cloud, then run the shared in-place GPU replace — one
        // pack/adopt code path for both first-build and later reloads.
        let cloud = PointRasteriserPointCloud(
            context: context, gpuPackPointCount: max(1, count),
            pointsPerBatch: pointsPerBatch, lodCapacity: lodCapacity, label: label
        )
        if count > 0 {
            cloud.replacePackedPointCloud(packer: packer, queue: queue, positions: positions, colors: colors, count: count, shuffle: shuffle)
        }
        return cloud
    }

    /// Build a wholesale cloud by GPU-packing point arrays: uploads `positions`
    /// (`SIMD3<Float>`) and `colors` (`SIMD4<Float>`, RGBA `[0,1]`) to shared
    /// buffers, then packs on the GPU. Convenience over the buffer overload.
    static func gpuPacked(
        context: Context,
        packer: GPUPacker,
        queue: MTLCommandQueue,
        positions: [SIMD3<Float>],
        colors: [SIMD4<Float>],
        pointsPerBatch: Int = pointRasteriserThreadsPerGroup * 80,
        lodCapacity: Int? = nil,
        shuffle: Bool = true,
        label: String = "PointRasteriserPointCloud"
    ) -> PointRasteriserPointCloud {
        precondition(positions.count == colors.count, "positions and colors must have the same count")
        let count = positions.count
        guard count > 0,
              let posBuf = context.device.makeBuffer(length: count * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared),
              let colBuf = context.device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)
        else {
            return PointRasteriserPointCloud(context: context, gpuPackPointCount: max(1, count), pointsPerBatch: pointsPerBatch, lodCapacity: lodCapacity, label: label)
        }
        positions.withUnsafeBytes { posBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        colors.withUnsafeBytes { colBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return gpuPacked(
            context: context, packer: packer, queue: queue,
            positions: posBuf, colors: colBuf, count: count,
            pointsPerBatch: pointsPerBatch, lodCapacity: lodCapacity, shuffle: shuffle, label: label
        )
    }
}
