import Foundation
import Metal
import Satin
import SatinPointRasteriser
import SatinPointRasteriserStreaming
import SwiftPDAL
import Testing
import simd

// Adapter unit tests driven by a public-API-only mock source (no real COPC
// file, no @testable). Now possible because SwiftPDAL exposes public memberwise
// inits for StreamingSourceInfo / ResidentChunk / StreamingDecodeStats.

/// Serializes these GPU-touching tests (each allocates a Metal context + slot
/// pool); avoids concurrent-Metal flakiness under swift-testing's parallel run.
nonisolated(unsafe) let streamingMockLock = NSLock()

/// Minimal in-memory `StreamingPointCloudSource` built from public API only.
private final class MockStreamingSource: StreamingPointCloudSource, @unchecked Sendable {
    let info: StreamingSourceInfo
    private let lock = NSLock()
    private var queued: [StreamingUpdate] = []
    private var stats = StreamingDecodeStats()
    private(set) var lastTargetScreenSize: Float?
    private(set) var lastBudget: Int?
    private(set) var closed = false

    init(info: StreamingSourceInfo) { self.info = info }

    func enqueue(_ u: StreamingUpdate) { lock.withLock { queued.append(u) } }
    func setStats(_ s: StreamingDecodeStats) { lock.withLock { stats = s } }

    func submit(view: StreamingCameraView) {}
    func setBudget(_ bytes: Int) { lock.withLock { lastBudget = bytes } }
    func pollLatest() -> StreamingUpdate? { lock.withLock { queued.isEmpty ? nil : queued.removeFirst() } }
    func nextUpdate() async -> StreamingUpdate? { pollLatest() }
    func cancel(_ chunkIDs: [ChunkID]) {}
    func close() { lock.withLock { closed = true } }
    func setTargetChunkScreenSize(_ pixels: Float) { lock.withLock { lastTargetScreenSize = pixels } }
    func decodeStats() -> StreamingDecodeStats { lock.withLock { stats } }
}

private func mockInfo(pointsPerBatch: Int = 256, maxDepth: Int = 6) -> StreamingSourceInfo {
    StreamingSourceInfo(
        bounds: Bounds(min: SIMD3<Float>(repeating: -1), max: SIMD3<Float>(repeating: 1)),
        originShift: SIMD3<Double>(0, 0, 0),
        totalPoints: 1_000_000,
        maxDepth: maxDepth,
        pointsPerBatch: pointsPerBatch,
        bytesPerPoint: 17,
        availableDimensions: []
    )
}

private func mockChunk(depth: Int, x: Int, points: Int) -> ResidentChunk {
    let id = ChunkID(depth: depth, x: x, y: 0, z: 0)
    let batch = StreamingRasterBatch(
        state: 1, min: SIMD3<Float>(repeating: -1), max: SIMD3<Float>(repeating: 1),
        numPoints: UInt32(points), firstPoint: 0, fileIndex: 0
    )
    return ResidentChunk(
        id: id, batches: [batch],
        xyzLow: Data(count: points * 4), xyzMed: Data(count: points * 4), xyzHigh: Data(count: points * 4),
        colors: Data(count: points * 4), levels: Data(count: points)
    )
}

private func makeCloud(_ device: MTLDevice, slotCapacity: Int, pointsPerBatch: Int) -> (Context, PointRasteriserPointCloud, PerspectiveCamera) {
    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)
    let cloud = PointRasteriserPointCloud(context: context, slotCapacity: slotCapacity, pointsPerBatch: pointsPerBatch, label: "MockPool")
    let cam = PerspectiveCamera(context: context, position: [0, 0, 3], near: 0.01, far: 100, fov: 45)
    cam.aspect = 1
    cam.lookAt(target: .zero)
    return (context, cloud, cam)
}

@Test func adapterKeepsSourcePinnedCoarseAndEvictsFine() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    streamingMockLock.lock(); defer { streamingMockLock.unlock() }
    let (_, cloud, cam) = makeCloud(device, slotCapacity: 32, pointsPerBatch: 256)
    let source = MockStreamingSource(info: mockInfo())
    let adapter = StreamingAdapter(source: source, cloud: cloud)
    adapter.coarsePinnedDepth = 2 // mirror the source option (telemetry only)
    let vp = SIMD2<Float>(256, 256)

    // Add 2 coarse (depth 0,1) + 2 fine (depth 3,4).
    source.enqueue(StreamingUpdate(added: [
        mockChunk(depth: 0, x: 0, points: 200),
        mockChunk(depth: 1, x: 0, points: 200),
        mockChunk(depth: 3, x: 0, points: 200),
        mockChunk(depth: 4, x: 0, points: 200),
    ], removed: []))
    adapter.update(camera: cam, viewport: vp)
    #expect(adapter.residentChunks == 4)
    #expect(adapter.pinnedResidentChunks == 2, "depth 0,1 should count as pinned coverage")

    // The source evicts ONLY the fine nodes (it never lists pinned in `removed`).
    source.enqueue(StreamingUpdate(added: [], removed: [
        ChunkID(depth: 3, x: 0, y: 0, z: 0),
        ChunkID(depth: 4, x: 0, y: 0, z: 0),
    ]))
    adapter.update(camera: cam, viewport: vp)
    #expect(adapter.residentChunks == 2, "fine nodes evicted")
    #expect(adapter.pinnedResidentChunks == 2, "coarse coverage held (never uncovered)")
    #expect(adapter.totalChunksEvicted == 2)
}

@Test func adapterUploadsInPriorityOrderAndRetriesSlotFull() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    streamingMockLock.lock(); defer { streamingMockLock.unlock() }
    // Pool holds only 2 slots; add 3 chunks in priority order A,B,C.
    let (_, cloud, cam) = makeCloud(device, slotCapacity: 2, pointsPerBatch: 256)
    let source = MockStreamingSource(info: mockInfo())
    let adapter = StreamingAdapter(source: source, cloud: cloud)
    let vp = SIMD2<Float>(256, 256)

    source.enqueue(StreamingUpdate(added: [
        mockChunk(depth: 3, x: 0, points: 200), // A
        mockChunk(depth: 3, x: 1, points: 200), // B
        mockChunk(depth: 3, x: 2, points: 200), // C
    ], removed: []))
    adapter.update(camera: cam, viewport: vp)
    // A + B fill the pool (priority order); C can't fit → parked + starvation.
    #expect(adapter.residentChunks == 2)
    #expect(adapter.pendingUploadCount == 1, "C should stay pending")
    #expect(adapter.starvedTickCount == 1, "slot-full should register starvation")

    // Free a slot (evict A) → the retried C uploads on the next tick.
    source.enqueue(StreamingUpdate(added: [], removed: [ChunkID(depth: 3, x: 0, y: 0, z: 0)]))
    adapter.update(camera: cam, viewport: vp)
    #expect(adapter.residentChunks == 2, "B + C now resident")
    #expect(adapter.pendingUploadCount == 0, "C uploaded after retry")
    #expect(adapter.totalChunksUploaded == 3, "all three eventually uploaded")
}

@Test func adapterSurfacesDecodeStatsAndRetarget() {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    streamingMockLock.lock(); defer { streamingMockLock.unlock() }
    let (_, cloud, cam) = makeCloud(device, slotCapacity: 8, pointsPerBatch: 256)
    let source = MockStreamingSource(info: mockInfo())
    let adapter = StreamingAdapter(source: source, cloud: cloud)
    let vp = SIMD2<Float>(256, 256)

    source.setStats(StreamingDecodeStats(pendingRequests: 7, inFlightDecodes: 3, decodedChunks: 10, decodedPoints: 100_000))
    adapter.update(camera: cam, viewport: vp)
    #expect(adapter.decodePendingRequests == 7)
    #expect(adapter.decodeInFlight == 3)

    // Second sample after a delay → nonzero decode-rate EMA from monotonic points.
    Thread.sleep(forTimeInterval: 0.08)
    source.setStats(StreamingDecodeStats(pendingRequests: 2, inFlightDecodes: 1, decodedChunks: 20, decodedPoints: 300_000))
    adapter.update(camera: cam, viewport: vp)
    #expect(adapter.decodePendingRequests == 2)
    #expect(adapter.decodedPointsPerSecond > 0, "rate should be derived from monotonic decodedPoints")

    // Live retarget passes through to the source.
    adapter.setTargetChunkScreenSize(64)
    #expect(source.lastTargetScreenSize == 64)
}
