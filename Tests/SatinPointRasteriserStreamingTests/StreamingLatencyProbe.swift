import Foundation
import Metal
import Satin
import SatinPointRasteriser
import SatinPointRasteriserStreaming
import SwiftPDAL
import Testing
import simd

// ============================================================================
// THROWAWAY INVESTIGATION INSTRUMENTATION (Slice-11 diagnosis, env-gated).
// Not shipped behavior — measures the streaming chunk lifecycle latency budget
// and directly tests the "frustum culling is wrong / infrequent" hypothesis.
//   PR_COPC_TEST_FILE="/…/ipogeo_cleaned.copc.laz" \
//     swift test --no-parallel --filter streamingLatencyProbe
// ============================================================================

@Test func streamingLatencyProbe() async throws {
    guard let path = ProcessInfo.processInfo.environment["PR_COPC_TEST_FILE"], !path.isEmpty else {
        print("[SKIP] streamingLatencyProbe: set PR_COPC_TEST_FILE to run."); return
    }
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: url.path) else { Issue.record("missing \(path)"); return }
    guard let device = MTLCreateSystemDefaultDevice() else { print("[SKIP] no device"); return }

    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)
    let cores = max(2, ProcessInfo.processInfo.activeProcessorCount)
    let options = StreamingOptions(
        maxInFlightLoads: cores * 2, decodeConcurrency: cores,
        driverTickInterval: .milliseconds(16), residencyPolicy: .frustumFirstThenHalo,
        alwaysResidentDepth: 2
    )
    let source = try await CopcStreamingPointCloudSource.open(url, options: options)
    let budget = 1024 * 1024 * 1024 // GUI default
    source.setBudget(budget)

    let ppb = source.info.pointsPerBatch
    let slotCapacity = min(max(16, (budget / max(1, ppb * source.info.bytesPerPoint)) * 2), 65536)
    let cloud = PointRasteriserPointCloud(context: context, slotCapacity: slotCapacity, pointsPerBatch: ppb, label: "Probe")
    let rasteriser = PointRasteriser(context: context)
    rasteriser.setup()
    rasteriser.addPointCloud(cloud)
    rasteriser.resize(size: (512, 512), scaleFactor: 1)
    let adapter = StreamingAdapter(source: source, cloud: cloud)
    adapter.coarsePinnedDepth = 2

    let shift = SIMD3<Float>(source.info.originShift)
    let bmin = source.info.bounds.min - shift, bmax = source.info.bounds.max - shift
    let center = (bmin + bmax) * 0.5
    let radius = max(simd_length(bmax - bmin) * 0.5, 0.01)
    let vp = SIMD2<Float>(512, 512)
    let cam = PerspectiveCamera(context: context, position: .zero, near: 0.01, far: radius * 12, fov: 45)
    cam.aspect = 1
    let dist = radius / max(tanf(cam.fov * 0.5 * .pi / 180), 0.001) * 1.4

    // 60 Hz frame pump: submit view every "frame" (mirrors the GUI's per-frame
    // adapter.update → source.submit), which is latest-wins into the 16 ms driver.
    func pump(seconds: Double, place: (Int) -> Void) async {
        let frames = Int(seconds * 60)
        for f in 0 ..< frames { place(f); adapter.update(camera: cam, viewport: vp); try? await Task.sleep(for: .milliseconds(16)) }
    }
    func faceFrom(_ pos: SIMD3<Float>) { cam.position = pos; cam.lookAt(target: center) }

    // ---- Warm up on region A (front) ----
    let front = center + SIMD3<Float>(0, 0.2 * radius, dist)
    await pump(seconds: 2.0) { _ in faceFrom(front) }

    // ---- (A) FRUSTUM CORRECTNESS: facing the cloud vs facing away ----
    faceFrom(front); adapter.update(camera: cam, viewport: vp)
    try? await Task.sleep(for: .milliseconds(120))
    let facing = await source._debugSnapshot()
    // Face 180° away: position on the far side looking outward.
    let away = center + SIMD3<Float>(0, 0.2 * radius, dist)
    cam.position = away
    cam.lookAt(target: center + SIMD3<Float>(0, 0, dist * 4)) // look past/away from the cloud
    await pump(seconds: 0.8) { _ in }
    let facingAway = await source._debugSnapshot()

    print("""

    === (A) FRUSTUM CORRECTNESS ===
    facing cloud:  candidates=\(facing.candidates)  frustumVisible=\(facing.frustumVisible)  wanted=\(facing.wanted)  resident=\(facing.resident)
    facing away:   candidates=\(facingAway.candidates)  frustumVisible=\(facingAway.frustumVisible)  wanted=\(facingAway.wanted)  resident=\(facingAway.resident)
    → culling ratio facing/away frustumVisible: \(facing.frustumVisible) vs \(facingAway.frustumVisible)  (away << facing ⇒ frustum works)
    cache hits/misses: \(facing.cacheHits)/\(facing.cacheMisses)
    """)

    // ---- Continuous-motion re-score cadence: pan every frame for 2 s ----
    let before = await source._debugSnapshot()
    let panStart = Date()
    await pump(seconds: 2.0) { f in
        let a = Float(f) / 120.0 * 1.5 * .pi
        faceFrom(center + SIMD3<Float>(sin(a) * dist, 0.2 * radius, cos(a) * dist))
    }
    let after = await source._debugSnapshot()
    let panDur = Date().timeIntervalSince(panStart)
    let rescores = after.cacheMisses - before.cacheMisses
    print("""

    === (B) RE-SCORE CADENCE (continuous pan, camera changes every frame) ===
    duration: \(String(format: "%.2fs", panDur))  wanted-set re-scores: \(rescores)  ⇒ \(String(format: "%.1f", Double(rescores) / panDur)) re-scores/s
    cache hits during pan: \(after.cacheHits - before.cacheHits)
    """)

    // ---- (C) PER-STAGE LATENCY: snap A→B, time each stage ----
    faceFrom(front)
    await pump(seconds: 1.5) { _ in }               // settle on A
    let s0 = source.decodeStats()
    let d0 = await source._debugSnapshot()
    let baseDecodedChunks = s0.decodedChunks
    let baseResident = adapter.residentChunks
    let baseMisses = d0.cacheMisses

    // Snap to region B (opposite side, dolly in for fine detail).
    let sideB = center + SIMD3<Float>(dist * 0.8, 0.2 * radius, -dist * 0.5)
    let t0 = Date()
    var tRescore: Double?, tFirstDecode: Double?, tFirstResident: Double?, tDetail: Double?
    var peakInFlight = 0, peakPending = 0, peakUploadBacklog = 0
    var lastDecoded = s0.decodedPoints
    let residentPointsBase = adapter.residentPoints
    for f in 0 ..< 360 { // up to 6 s
        faceFrom(sideB)
        adapter.update(camera: cam, viewport: vp)
        let now = Date().timeIntervalSince(t0)
        let st = source.decodeStats()
        peakInFlight = max(peakInFlight, st.inFlightDecodes)
        peakPending = max(peakPending, st.pendingRequests)
        peakUploadBacklog = max(peakUploadBacklog, adapter.pendingUploadCount)
        lastDecoded = st.decodedPoints
        if tRescore == nil {
            let snap = await source._debugSnapshot()
            if snap.cacheMisses > baseMisses { tRescore = now }
        }
        if tFirstDecode == nil, st.decodedChunks > baseDecodedChunks { tFirstDecode = now }
        if tFirstResident == nil, adapter.residentChunks > baseResident { tFirstResident = now }
        if tDetail == nil, adapter.residentPoints > residentPointsBase + 3_000_000 { tDetail = now }
        if tDetail != nil, f > 60 { break }
        try? await Task.sleep(for: .milliseconds(16))
    }
    let windowSec = Date().timeIntervalSince(t0)
    let decodedInWindow = lastDecoded - s0.decodedPoints
    let decodeRate = Double(decodedInWindow) / windowSec

    print("""

    === (C) PER-STAGE LATENCY (camera snap A→B → new detail resident) ===
    budget: \(budget / (1024*1024)) MB  decodeConcurrency: \(cores)  maxInFlightLoads: \(cores*2)  uploadCap: \(adapter.maxChunkUploadsPerTick)
    t_rescore (camera→wanted recomputed):        \(fmt(tRescore))
    t_first_decode (→ first new chunk decoded):  \(fmt(tFirstDecode))   [rescore + queue + lazperf]
    t_first_resident (→ uploaded to slot pool):  \(fmt(tFirstResident)) [+ adapter tick + slot memcpy]
    t_detail (→ +3M new points resident):        \(fmt(tDetail))
    window: \(String(format: "%.2fs", windowSec))  decoded in window: \(decodedInWindow) pts  ⇒ \(String(format: "%.1f M pts/s", decodeRate / 1e6))
    peak decode in-flight: \(peakInFlight)  peak decode pending(queued): \(peakPending)  peak adapter upload backlog: \(peakUploadBacklog)
    resident-by-depth (end): \((await source._debugSnapshot()).residentByDepth)
    """)

    adapter.close()
    #expect(facing.frustumVisible >= 0)
}

private func fmt(_ t: Double?) -> String { t.map { String(format: "%.0f ms", $0 * 1000) } ?? "n/a" }
