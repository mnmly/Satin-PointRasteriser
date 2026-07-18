#if os(macOS)
import Foundation
import Observation
import SatinPointRasteriser
import simd

/// Mirror of `SwiftPDAL.ResidencyPolicy` that doesn't require importing
/// SwiftPDAL into this state type (which should stay lightweight even when
/// the streaming module is unavailable).
enum StreamingResidencyChoice: String, CaseIterable, Hashable {
    /// SwiftPDAL's `frustumFirstThenHalo` — frustum-visible chunks first,
    /// then surround the camera with leftover budget.
    case halo
    /// SwiftPDAL's `distanceOnly` — no frustum gate, fill nearest-first.
    /// Matches the "load it all" feel of non-streaming renderers.
    case distance
}

/// Observable mirror of ``PointRasteriserConfiguration`` (plus DoF + status
/// fields) driving the example app's settings UI. The renderer rebuilds a
/// fresh ``PointRasteriserConfiguration`` from this state every frame and
/// assigns it to `rasteriser.configuration` (see
/// `PointRasteriserExampleRenderer.update()`), so every slider/toggle here
/// takes effect on the next frame with no extra plumbing.
///
/// `applyDisplacement` / `applyTint` / `tintAlphaIsCoverage` are deliberately
/// **not** mirrored here — they're owned by whichever ``DisplacementPass`` /
/// ``TintPass`` is currently encoding (flipped automatically by `encode`/`disable`),
/// so a config push from this state always leaves them at their struct default
/// and lets the pass's own encode call (which runs later in the frame, right
/// before the rasteriser consumes them) win. LOD capacity is fixed per-cloud at
/// ``PointRasteriserPointCloud`` creation (defaulting to the full source count so
/// overflow can't happen), so it isn't a per-frame config knob.
@Observable
final class PointRasteriserExampleState {
    // MARK: - Status

    var status: String = "Fixture (64³ cube grid)"
    var errorMessage: String?
    var isLoading: Bool = false

    // MARK: - Streaming (COPC)

    /// `true` once a COPC source is loaded; drives streaming-only UI (budget
    /// slider, residency toggle, telemetry overlay). `false` in fixture/PLY mode.
    var isStreaming: Bool = false
    /// Resident chunk/point/free-slot telemetry, refreshed each frame from the
    /// active ``StreamingAdapter``(s). Zero in non-streaming mode.
    var streamingChunks: Int = 0
    var streamingPoints: Int = 0
    var streamingFreeSlots: Int = 0
    /// Guaranteed-coverage coarse chunks pinned resident (never evicted).
    var streamingPinnedChunks: Int = 0
    /// Decoded chunks queued but not yet uploaded (integration backlog).
    var streamingPendingUploads: Int = 0
    /// Ticks where a decoded chunk couldn't upload (slot pool full → never-fill risk).
    var streamingStarvedTicks: Int = 0
    /// Decode/publish throughput, millions of points per second.
    var streamingDecodeMPS: Double = 0
    /// Source decode-queue back-pressure (nodes scheduled, not yet decoding).
    var streamingDecodePending: Int = 0
    /// Source nodes currently mid-decode.
    var streamingDecodeInFlight: Int = 0
    /// Live residency detail target (screen px per node; smaller = more detail).
    var streamingTargetChunkPx: Float = 256
    /// Total streaming budget in MB, split equally across concurrently loaded
    /// COPC sources.
    var streamingBudgetMB: Int = 1024
    var streamingResidency: StreamingResidencyChoice = .halo

    // MARK: - Render mode

    var renderMode: RenderMode = .highQualityAverage
    var enableSimdAggregation: Bool = true

    // MARK: - Point sizing

    var pointSizeMode: PointSizeMode = .screenSpace
    var minimumPointSize: Float = 1.0
    var maximumPointSize: Float = 6.0
    var pointSizeScale: Float = 5.0
    /// Analytic per-point edge antialiasing (soft ~1px disc silhouette).
    var pointEdgeAntialiasing: Bool = false

    // MARK: - LOD & culling

    var enableFrustumCulling: Bool = true
    var enableCLOD: Bool = true
    var lodBias: Int = 0
    var enableLODDither: Bool = true
    /// Amortization budget in **source points/frame**; `0` = full sweep every frame.
    var lodPointsPerFrame: Int = 0

    // MARK: - LOD sweep telemetry (read-only; refreshed each frame from the front cloud)

    var lodSweepProgress: Float = 1
    var lodCount: Int = 0
    var lodOverflow: Int = 0
    var lodOverflowed: Bool = false
    /// `true` when last frame's full-sweep LODSelect was skipped (static scene).
    var lodSelectSkipped: Bool = false

    // MARK: - Rejection / hole fill / depth test

    var enablePointRejection: Bool = false
    var rejectionConeThreshold: Float = 0.5
    var holeFillIterations: Int = 0
    var depthTolerance: Float = 0.01

    // MARK: - Scene / debug

    var writesSceneDepth: Bool = true
    var backgroundColor: SIMD4<Float> = .zero
    var colorizeChunks: Bool = false
    var colorizeOverdraw: Bool = false

    // MARK: - Motion blur

    var motionBlur: Float = 0.0
    var motionBlurSamples: Int = 8
    var motionBlurMaxSpread: Float = 64.0

    // MARK: - Sine-wave displacement demo ('D' key / toggle)

    var sineDisplacementEnabled: Bool = false
    /// Wave amplitude as a **fraction of the loaded cloud's radius**, so the same
    /// slider is visible on a unit-cube fixture and a 100 m COPC scene alike. The
    /// renderer multiplies this by the cloud radius before binding it to the kernel.
    var sineDisplacementAmplitude: Float = 0.08
    /// Wave frequency in **cycles across the cloud's radius** (also scale-normalized
    /// by the renderer), so the sine stays coherent regardless of world extent.
    var sineDisplacementFrequency: Float = 7.0

    // MARK: - Depth of field (DisplacementPass jitter + TintPass coverage recipe)

    var dofEnabled: Bool = false
    /// Translucent defocus via weighted-blended OIT (``TintPass/alphaIsCoverage``).
    var dofTranslucent: Bool = true
    /// Scatter out-of-focus points (``DisplacementPass``).
    var dofJitter: Bool = true
    /// Focus on the loaded cloud's centre rather than ``dofFocus``.
    var dofAutoFocus: Bool = true
    var dofFocus: Float = 1.0
    var dofFocusMax: Float = 10
    var dofBand: Float = 0.04
    var dofFalloff: Float = 0.25
    var dofScatter: Float = 0.05
    var dofMaxDefocus: Float = 0.85

    /// Builds a fresh ``PointRasteriserConfiguration`` from the current state.
    /// `applyDisplacement`/`applyTint`/`tintAlphaIsCoverage`/`lodCapacity` are
    /// intentionally left at their struct defaults; see the type doc comment.
    func makeConfiguration() -> PointRasteriserConfiguration {
        PointRasteriserConfiguration(
            renderMode: renderMode,
            depthTolerance: depthTolerance,
            backgroundColor: backgroundColor,
            enableFrustumCulling: enableFrustumCulling,
            lodBias: lodBias,
            enableCLOD: enableCLOD,
            enableLODDither: enableLODDither,
            holeFillIterations: holeFillIterations,
            colorizeChunks: colorizeChunks,
            colorizeOverdraw: colorizeOverdraw,
            pointSizeMode: pointSizeMode,
            minimumPointSize: minimumPointSize,
            maximumPointSize: maximumPointSize,
            pointSizeScale: pointSizeScale,
            writesSceneDepth: writesSceneDepth,
            motionBlur: motionBlur,
            motionBlurSamples: motionBlurSamples,
            motionBlurMaxSpread: motionBlurMaxSpread,
            enableSimdAggregation: enableSimdAggregation,
            lodPointsPerFrame: lodPointsPerFrame,
            enablePointRejection: enablePointRejection,
            rejectionConeThreshold: rejectionConeThreshold,
            pointEdgeAntialiasing: pointEdgeAntialiasing
        )
    }
}
#endif
