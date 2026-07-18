import Foundation
import Metal
import Satin
import SatinPointRasteriser
import SatinPointRasteriserStreaming
import SwiftPDAL
import Testing
import simd

// MARK: - Env-gated smoke test against a real COPC dataset.
//
// Skips (not fails) unless `PR_COPC_TEST_FILE` names a readable COPC LAZ:
//   PR_COPC_TEST_FILE=/path/to/file.copc.laz \
//     swift test --filter streamingRealFileSmokeTest
//
// Opens the file through `StreamingAdapter` with a deliberately modest
// resident budget (so the test doesn't need to pull in the whole file),
// drives ticks until a handful of batches are resident, then encodes one
// offscreen frame and asserts the pipeline actually produced visible output
// (nonzero `lodCount` + covered pixels) — i.e. streamed data reaches the
// rasteriser end-to-end, not just the adapter's bookkeeping.

@Test func streamingRealFileSmokeTest() async throws {
    guard let path = ProcessInfo.processInfo.environment["PR_COPC_TEST_FILE"], !path.isEmpty else {
        print("[SKIP] streamingRealFileSmokeTest: set PR_COPC_TEST_FILE to a COPC path to run this test.")
        return
    }
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: url.path) else {
        Issue.record("PR_COPC_TEST_FILE points to a missing file: \(path)")
        return
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("[SKIP] streamingRealFileSmokeTest: no Metal device available.")
        return
    }

    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm, depthPixelFormat: .depth32Float)

    let cores = max(2, ProcessInfo.processInfo.activeProcessorCount)
    let options = StreamingOptions(
        maxInFlightLoads: cores * 2,
        decodeConcurrency: cores,
        driverTickInterval: .milliseconds(16),
        residencyPolicy: .distanceOnly
    )
    let openStart = Date()
    let source = try await CopcStreamingPointCloudSource.open(url, options: options)
    let openDuration = Date().timeIntervalSince(openStart)

    // Modest budget: enough to pull in a handful of chunks without needing
    // the whole (potentially multi-GB) file.
    let budgetBytes = 96 * 1024 * 1024
    source.setBudget(budgetBytes)

    let pointsPerBatch = source.info.pointsPerBatch
    let bytesPerSlot = max(1, pointsPerBatch * source.info.bytesPerPoint)
    // 2x headroom over the byte budget, same rationale as the example app:
    // every chunk occupies whole slots (its last batch is partial), so a
    // pool sized 1:1 to the budget exhausts before the source stops
    // admitting.
    let slotCapacity = max(16, (budgetBytes / bytesPerSlot) * 2)

    let cloud = PointRasteriserPointCloud(
        context: context,
        slotCapacity: slotCapacity,
        pointsPerBatch: pointsPerBatch,
        label: "StreamingSmokeTest"
    )

    let rasteriser = PointRasteriser(context: context)
    rasteriser.setup()
    rasteriser.addPointCloud(cloud)
    let width = 256, height = 256
    rasteriser.resize(size: (Float(width), Float(height)), scaleFactor: 1)

    let adapter = StreamingAdapter(source: source, cloud: cloud)
    adapter.maxChunkUploadsPerTick = 16

    let originShift = SIMD3<Float>(source.info.originShift)
    let boundsMin = source.info.bounds.min - originShift
    let boundsMax = source.info.bounds.max - originShift
    let center = (boundsMin + boundsMax) * 0.5
    let radius = max(simd_length(boundsMax - boundsMin) * 0.5, 0.01)

    let camera = PerspectiveCamera(context: context, position: .zero, near: 0.01, far: 1000, fov: 45)
    camera.aspect = Float(width) / Float(height)
    let distance = radius / max(tanf(camera.fov * 0.5 * .pi / 180), 0.001) * 1.5
    camera.position = center + SIMD3<Float>(0, 0, distance)
    camera.near = max(distance - radius * 3, 0.001)
    camera.far = distance + radius * 4
    camera.lookAt(target: center)

    let viewport = SIMD2<Float>(Float(width), Float(height))

    // Drive ticks until a handful of batches are resident (or timeout).
    let minResidentBatches = 4
    let deadline = Date().addingTimeInterval(60)
    let driveStart = Date()
    var timeToFirstResidency: TimeInterval?
    while Date() < deadline, cloud.residentBatchCount < minResidentBatches {
        adapter.update(camera: camera, viewport: viewport)
        if timeToFirstResidency == nil, cloud.residentBatchCount > 0 {
            timeToFirstResidency = Date().timeIntervalSince(driveStart)
        }
        try? await Task.sleep(for: .milliseconds(50))
    }
    // One more tick so any chunks the last poll queued get uploaded.
    adapter.update(camera: camera, viewport: viewport)
    let driveDuration = Date().timeIntervalSince(driveStart)

    print("""
    === streamingRealFileSmokeTest ===
    file: \(url.lastPathComponent)
    totalPoints: \(source.info.totalPoints)  maxDepth: \(source.info.maxDepth)  pointsPerBatch: \(pointsPerBatch)
    slotCapacity: \(slotCapacity)  budget: \(budgetBytes / (1024 * 1024)) MB
    open duration: \(String(format: "%.3f", openDuration))s
    time to first residency: \(timeToFirstResidency.map { String(format: "%.3fs", $0) } ?? "n/a")
    drive duration: \(String(format: "%.3f", driveDuration))s
    residentChunks: \(adapter.residentChunks)  residentBatches: \(cloud.residentBatchCount)  residentPoints: \(cloud.residentPointCount)
    lastError: \(adapter.lastError ?? "none")
    """)

    #expect(cloud.residentBatchCount > 0, "no batches became resident within the timeout")

    // Offscreen render of one frame with the streamed cloud.
    guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
        Issue.record("failed to create command buffer")
        return
    }
    let viewport4 = simd_float4(0, 0, Float(width), Float(height))
    rasteriser.update(renderContext: context, camera: camera, viewport: viewport4, index: 0)
    rasteriser.encode(commandBuffer)

    guard let outputTexture = rasteriser.outputTexture,
          let rgbaStaging = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
          let blit = commandBuffer.makeBlitCommandEncoder()
    else {
        Issue.record("failed to set up readback")
        return
    }
    blit.copy(
        from: outputTexture, sourceSlice: 0, sourceLevel: 0,
        sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
        sourceSize: MTLSize(width: width, height: height, depth: 1),
        to: rgbaStaging, destinationOffset: 0,
        destinationBytesPerRow: width * 4, destinationBytesPerImage: width * height * 4
    )
    blit.endEncoding()
    commandBuffer.commit()
    await commandBuffer.completed()

    let rgba = [UInt8](unsafeUninitializedCapacity: width * height * 4) { buf, count in
        memcpy(buf.baseAddress!, rgbaStaging.contents(), width * height * 4)
        count = width * height * 4
    }
    var coveredPixels = 0
    for i in 0 ..< (width * height) where rgba[i * 4 + 3] > 0 { coveredPixels += 1 }

    print("lodCount: \(cloud.lodCount)  coveredPixels: \(coveredPixels) / \(width * height)")

    #expect(cloud.lodCount > 0, "streamed cloud produced zero LOD survivors")
    #expect(coveredPixels > 0, "streamed cloud produced zero covered pixels")

    adapter.close()
}
