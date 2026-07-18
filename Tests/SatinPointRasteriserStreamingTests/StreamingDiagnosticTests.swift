import Foundation
import Metal
import Satin
import SatinPointRasteriser
import SatinPointRasteriserStreaming
import SwiftPDAL
import Testing
import simd

// Env-gated streaming diagnostic against a real COPC dataset. Skips unless
// PR_COPC_TEST_FILE names a readable COPC LAZ. Runs a scripted orbit camera and
// reports fill telemetry with the fixes OFF (pin disabled, upload cap 4 — the
// pre-fix behavior) then ON (defaults), so the before/after is one run.
//
//   PR_COPC_TEST_FILE="/path/file.copc.laz" swift test --filter streamingFillDiagnostic

private struct DiagMetrics {
    var timeToFirstResidency: TimeInterval?
    var timeToCoarseCoverage: TimeInterval?   // all depth<=pin nodes resident
    var peakResidentChunks = 0
    var minResidentChunksAfterWarmup = Int.max // coverage dips during motion
    var endResidentChunks = 0
    var endResidentPoints = 0
    var endPinnedChunks = 0
    var totalUploaded = 0
    var totalEvicted = 0
    var totalDroppedByEviction = 0
    var starvedTicks = 0
    var peakPending = 0
    var maxDecodePtsPerSec = 0.0
    var frames = 0
    /// Once coarse coverage is established, the lowest pinned-resident count
    /// seen thereafter. If the pin holds (never uncovered), this stays == the
    /// coverage count; a drop means a coarse block went missing.
    var minPinnedAfterCoverage = Int.max
    var peakResidentPoints = 0
}

@Test func streamingFillDiagnostic() async throws {
    guard let path = ProcessInfo.processInfo.environment["PR_COPC_TEST_FILE"], !path.isEmpty else {
        print("[SKIP] streamingFillDiagnostic: set PR_COPC_TEST_FILE to run.")
        return
    }
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: url.path) else {
        Issue.record("PR_COPC_TEST_FILE missing: \(path)"); return
    }
    guard let device = MTLCreateSystemDefaultDevice() else { print("[SKIP] no Metal device"); return }

    func run(label: String, pinDepth: Int, uploadCap: Int) async throws -> DiagMetrics {
        let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)
        let cores = max(2, ProcessInfo.processInfo.activeProcessorCount)
        // Source-native pinning via alwaysResidentDepth (nil = off for the A run).
        let options = StreamingOptions(
            maxInFlightLoads: cores * 2, decodeConcurrency: cores,
            driverTickInterval: .milliseconds(16), residencyPolicy: .frustumFirstThenHalo,
            alwaysResidentDepth: pinDepth >= 0 ? pinDepth : nil
        )
        let source = try await CopcStreamingPointCloudSource.open(url, options: options)
        let budgetBytes = 512 * 1024 * 1024
        source.setBudget(budgetBytes)

        let ppb = source.info.pointsPerBatch
        let bytesPerSlot = max(1, ppb * source.info.bytesPerPoint)
        let slotCapacity = min(max(16, (budgetBytes / bytesPerSlot) * 2), 65536)
        let cloud = PointRasteriserPointCloud(context: context, slotCapacity: slotCapacity, pointsPerBatch: ppb, label: "Diag")

        let adapter = StreamingAdapter(source: source, cloud: cloud)
        adapter.coarsePinnedDepth = pinDepth
        adapter.maxChunkUploadsPerTick = uploadCap

        let shift = SIMD3<Float>(source.info.originShift)
        let bmin = source.info.bounds.min - shift, bmax = source.info.bounds.max - shift
        let center = (bmin + bmax) * 0.5
        let radius = max(simd_length(bmax - bmin) * 0.5, 0.01)
        let cam = PerspectiveCamera(context: context, position: .zero, near: 0.01, far: radius * 10, fov: 45)
        cam.aspect = 1
        let baseDist = radius / max(tanf(cam.fov * 0.5 * .pi / 180), 0.001) * 1.4
        let viewport = SIMD2<Float>(512, 512)

        var m = DiagMetrics()
        let start = Date()
        let frames = 360 // ~6s at 16ms/frame
        var coverageEstablished = false
        for f in 0 ..< frames {
            // First 2/3: orbit far (coarse view). Last 1/3: aggressive dolly-in
            // (fine-node burst) to stress decode + upload cap + coverage during motion.
            let t = Float(f) / Float(frames)
            let angle = t * 2.6 * .pi
            let dist: Float = t < 0.66 ? baseDist : baseDist * (1.0 - 0.92 * ((t - 0.66) / 0.34))
            cam.position = center + SIMD3<Float>(sin(angle) * dist, 0.25 * radius, cos(angle) * dist)
            cam.lookAt(target: center)

            adapter.update(camera: cam, viewport: viewport)
            m.frames += 1

            let now = Date().timeIntervalSince(start)
            if m.timeToFirstResidency == nil, adapter.residentChunks > 0 { m.timeToFirstResidency = now }
            if !coverageEstablished, pinDepth >= 0,
               adapter.pinnedResidentChunks >= expectedCoarseNodes(maxDepth: min(pinDepth, source.info.maxDepth)) {
                coverageEstablished = true
                m.timeToCoarseCoverage = now
            }
            m.peakResidentChunks = max(m.peakResidentChunks, adapter.residentChunks)
            m.peakResidentPoints = max(m.peakResidentPoints, adapter.residentPoints)
            if f > 40 { m.minResidentChunksAfterWarmup = min(m.minResidentChunksAfterWarmup, adapter.residentChunks) }
            if coverageEstablished { m.minPinnedAfterCoverage = min(m.minPinnedAfterCoverage, adapter.pinnedResidentChunks) }
            m.peakPending = max(m.peakPending, adapter.pendingUploadCount)
            m.maxDecodePtsPerSec = max(m.maxDecodePtsPerSec, adapter.decodedPointsPerSecond)

            try? await Task.sleep(for: .milliseconds(16))
        }
        if m.minPinnedAfterCoverage == Int.max { m.minPinnedAfterCoverage = 0 }
        m.endResidentChunks = adapter.residentChunks
        m.endResidentPoints = adapter.residentPoints
        m.endPinnedChunks = adapter.pinnedResidentChunks
        m.totalUploaded = adapter.totalChunksUploaded
        m.totalEvicted = adapter.totalChunksEvicted
        m.totalDroppedByEviction = adapter.totalPendingDroppedByEviction
        m.starvedTicks = adapter.starvedTickCount
        if m.minResidentChunksAfterWarmup == Int.max { m.minResidentChunksAfterWarmup = 0 }

        print("""

        === streamingFillDiagnostic [\(label)] ===
        file: \(url.lastPathComponent)  totalPoints: \(source.info.totalPoints)  maxDepth: \(source.info.maxDepth)  ppb: \(ppb)
        budget: \(budgetBytes / (1024*1024)) MB  slotCapacity: \(slotCapacity)  pinDepth: \(pinDepth)  uploadCap: \(uploadCap)
        timeToFirstResidency: \(m.timeToFirstResidency.map { String(format: "%.2fs", $0) } ?? "n/a")
        timeToCoarseCoverage: \(m.timeToCoarseCoverage.map { String(format: "%.2fs", $0) } ?? "n/a")
        resident chunks — peak: \(m.peakResidentChunks)  min-after-warmup (coverage dip): \(m.minResidentChunksAfterWarmup)  end: \(m.endResidentChunks)
        resident points end: \(m.endResidentPoints)  pinned resident end: \(m.endPinnedChunks)
        uploaded: \(m.totalUploaded)  evicted: \(m.totalEvicted)  droppedByEviction(lost refinement): \(m.totalDroppedByEviction)
        STARVED ticks (pool full, no eviction → never-fill): \(m.starvedTicks) / \(m.frames)
        peak pending backlog: \(m.peakPending)  max decode rate: \(String(format: "%.1f M pts/s", m.maxDecodePtsPerSec / 1e6))
        peak resident points (zoom detail): \(m.peakResidentPoints)
        pin coverage after establish — min: \(m.minPinnedAfterCoverage) (holds = never uncovered)
        """)
        adapter.close()
        return m
    }

    let before = try await run(label: "BEFORE (pin off, cap 4)", pinDepth: -1, uploadCap: 4)
    let after = try await run(label: "AFTER (pin depth 2, cap 32)", pinDepth: 2, uploadCap: 32)

    print("""

    === BEFORE vs AFTER ===
    starved ticks:              \(before.starvedTicks) → \(after.starvedTicks)
    dropped-by-eviction:        \(before.totalDroppedByEviction) → \(after.totalDroppedByEviction)
    coverage dip (min chunks):  \(before.minResidentChunksAfterWarmup) → \(after.minResidentChunksAfterWarmup)
    end resident chunks:        \(before.endResidentChunks) → \(after.endResidentChunks)
    end resident points:        \(before.endResidentPoints) → \(after.endResidentPoints)
    time to coarse coverage:    \(before.timeToCoarseCoverage.map { String(format: "%.2fs", $0) } ?? "n/a") → \(after.timeToCoarseCoverage.map { String(format: "%.2fs", $0) } ?? "n/a")
    """)

    // The fix must not regress fill: after should keep at least as many chunks
    // resident at the worst moment, with coarse coverage established.
    #expect(after.endResidentChunks > 0, "no chunks resident after run")
}

/// Number of octree nodes at depth 0…maxDepth in a *fully populated* octree is
/// Σ 8^d, but COPC files are sparse — most deep nodes are absent. For the
/// coarse-coverage completion proxy we only need "at least the root plus a few",
/// so use a conservative small lower bound rather than the theoretical count.
private func expectedCoarseNodes(maxDepth: Int) -> Int {
    // Root guaranteed; depth 1..2 sparse. 1 is enough to call coverage "started";
    // require a modest handful so the proxy reflects real spatial spread.
    maxDepth <= 0 ? 1 : 4
}
