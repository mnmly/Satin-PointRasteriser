import Foundation
import simd

/// Rasterization strategy for resolving overlapping points per pixel.
public enum RenderMode: Int32, CaseIterable, Hashable, Sendable {
    /// Depth-tested color averaging (Σcolor/count) — the default, highest
    /// quality. All visible clouds contribute.
    case highQualityAverage = 0
    /// Nearest surviving point per pixel (no averaging). Single cloud only (the
    /// first visible cloud). Uses a single 64-bit packed-atomic pass on Apple9+
    /// hardware, or a portable two-pass 32-bit fallback.
    case nearestPoint = 1
}

/// How a point's screen footprint (splat radius) is derived. Mirrors
/// Satin-ComputeRasteriser's `PointSizeMode`; the raw values match the Metal
/// `pointFootprintRadius` `pointSizeMode` branch (0 = screen space, 1 = world).
public enum PointSizeMode: Int32, CaseIterable, Hashable, Sendable {
    /// `pointSizeScale / length(viewSpacePosition)` clamped to `[min, max]`
    /// pixels. `pointSizeScale` reads as "pixels at one unit of view distance";
    /// under an orthographic projection size is a constant `pointSizeScale`.
    case screenSpace = 0
    /// Perspective projection of a world-space sphere radius. `pointSizeScale`
    /// is the sphere radius in scene units; FOV and screen height affect size.
    case worldSpace = 1
}

/// Runtime configuration for a ``PointRasteriser``. Values are pushed to the
/// pass processors via the ``PointRasteriser``'s setter→needsUpdate path
/// (mirrors Satin-ComputeRasteriser's `ComputeRasteriserConfiguration`).
public struct PointRasteriserConfiguration: Sendable {
    /// Rasterization strategy; see ``RenderMode``.
    public var renderMode: RenderMode
    /// Reduce within each simdgroup before hitting device atomics in the depth
    /// and color passes (one leader per distinct target pixel performs the
    /// combined max/add). Compile-time specialized via the `PR_SIMD_AGGREGATION`
    /// shader define; results are identical with it on or off (max/add are
    /// order-independent). Default reflects the Slice-4 benchmark on this class
    /// of hardware.
    public var enableSimdAggregation: Bool
    /// Footprint sizing mode; see ``PointSizeMode``.
    public var pointSizeMode: PointSizeMode
    /// Lower clamp (pixels) on the computed footprint diameter.
    public var minimumPointSize: Float
    /// Upper clamp (pixels) on the computed footprint diameter.
    public var maximumPointSize: Float
    /// Scale applied before clamping; interpreted per ``pointSizeMode``.
    public var pointSizeScale: Float
    /// Frustum-cull each source batch's AABB in the LODSelect pass.
    public var enableFrustumCulling: Bool
    /// Continuous LOD master switch. When `false`, the LODSelect pass writes a
    /// sentinel threshold so every resident point survives regardless of level.
    public var enableCLOD: Bool
    /// Shifts the CLOD pixel-size ramp (higher = more detail kept). Ignored
    /// when ``enableCLOD`` is `false`.
    public var lodBias: Int
    /// Per-point hash dither on the CLOD keep-test (smooths the LOD boundary).
    public var enableLODDither: Bool
    /// Amortize LOD selection across frames: at most this many **source** points
    /// per cloud are compacted per frame, resuming from a persistent cursor, and
    /// the raster passes read the last *completed* sweep from a double-buffered
    /// LOD set. `0` (the default) full-sweeps every frame with no extra memory.
    /// The budget is approximate at batch granularity (whole batches are
    /// processed until it is exceeded). Enabling it costs **2× LOD memory** and
    /// leaves empty space at frame edges when the camera moves faster than a
    /// sweep completes (the reference's accepted tradeoff). The cloud's very
    /// first sweep always runs un-amortized so a freshly loaded cloud is not
    /// blank while its first sweep builds.
    public var lodPointsPerFrame: Int
    /// Color the composited output; the resolve writes `rgb` with alpha 0 where
    /// no cloud lands so the composite leaves the scene untouched there.
    public var backgroundColor: SIMD4<Float>
    /// Fraction the color pass nudges its depth test toward the camera, so a
    /// point exactly at the winning surface still contributes.
    public var depthTolerance: Float
    /// Enable the VAST 2011 screen-space occlusion operator in the merged
    /// reject+resolve pass, discarding far points that leak through gaps in a
    /// nearer surface. See ``rejectionConeThreshold``.
    public var enablePointRejection: Bool
    /// Minimum empty-cone half-angle, in **radians**, for a point to be kept.
    /// A point is rejected when the largest eye-ward cone about it that is empty
    /// of closer neighbors is narrower than this. Meaningful range ~`0 … π/2`;
    /// higher values reject more aggressively. Ignored under an orthographic
    /// projection's parallel rays where the operator degenerates (the axis is a
    /// constant, so a nearer neighbor's angle still discriminates, but tune with
    /// care). Only applied when ``enablePointRejection`` is `true`.
    public var rejectionConeThreshold: Float
    /// Number of neighbor-average hole-fill iterations after resolve (0 = off).
    /// Each iteration expands valid color + depth into empty (α < 0.5) pixels.
    public var holeFillIterations: Int
    /// Debug: tint the color pass by source chunk instead of point color.
    public var colorizeChunks: Bool
    /// Debug: accumulate a constant per covered pixel so `count` reveals overdraw.
    public var colorizeOverdraw: Bool
    /// If `true`, the composite writes the cloud's per-pixel reverse-Z depth into
    /// the render pass depth attachment and depth-tests (`.greaterEqual`), so
    /// Satin meshes inter-occlude with the cloud. `false` = always-on-top overlay.
    public var writesSceneDepth: Bool
    /// Add per-point displacement in the depth + color passes; see full docs above.
    public var applyDisplacement: Bool
    /// Mix per-point tint in the color pass; see full docs above.
    public var applyTint: Bool
    /// Translucent-defocus / weighted-blended-OIT mode; see full docs above.
    public var tintAlphaIsCoverage: Bool
    /// Motion-blur shutter strength (0 = off).
    public var motionBlur: Float
    /// Maximum motion-blur sub-samples along the velocity vector.
    public var motionBlurSamples: Int
    /// Clamp on the motion-blur smear length in pixels.
    public var motionBlurMaxSpread: Float

    /// Creates a configuration; defaults render the fixture clouds sensibly.
    // Parameter order intentionally matches Satin-ComputeRasteriser's
    // `ComputeRasteriserConfiguration.init` for the shared fields, with the
    // PointRasteriser-only additions appended at the end. Swift requires
    // provided arguments to follow declaration order even when all are
    // defaulted, so this ordering is what lets CR-style call sites (a subset of
    // labels in CR order) compile unchanged against PointRasteriser.
    public init(
        renderMode: RenderMode = .highQualityAverage,
        depthTolerance: Float = 0.01,
        backgroundColor: SIMD4<Float> = SIMD4<Float>(0, 0, 0, 0),
        enableFrustumCulling: Bool = true,
        lodBias: Int = 0,
        enableCLOD: Bool = true,
        enableLODDither: Bool = true,
        holeFillIterations: Int = 0,
        colorizeChunks: Bool = false,
        colorizeOverdraw: Bool = false,
        pointSizeMode: PointSizeMode = .screenSpace,
        minimumPointSize: Float = 1.0,
        maximumPointSize: Float = 5.0,
        pointSizeScale: Float = 5.0,
        applyDisplacement: Bool = false,
        applyTint: Bool = false,
        tintAlphaIsCoverage: Bool = false,
        writesSceneDepth: Bool = true,
        motionBlur: Float = 0.0,
        motionBlurSamples: Int = 8,
        motionBlurMaxSpread: Float = 64.0,
        // PointRasteriser-only additions (appended so CR-style calls keep working):
        // SIMD-group pre-aggregation, amortized-LOD per-frame budget, and the
        // VAST 2011 point-rejection controls.
        enableSimdAggregation: Bool = true,
        lodPointsPerFrame: Int = 0,
        enablePointRejection: Bool = true,
        rejectionConeThreshold: Float = 0.5
    ) {
        self.renderMode = renderMode
        self.enableSimdAggregation = enableSimdAggregation
        self.pointSizeMode = pointSizeMode
        self.minimumPointSize = minimumPointSize
        self.maximumPointSize = maximumPointSize
        self.pointSizeScale = pointSizeScale
        self.enableFrustumCulling = enableFrustumCulling
        self.enableCLOD = enableCLOD
        self.lodBias = lodBias
        self.enableLODDither = enableLODDither
        self.lodPointsPerFrame = lodPointsPerFrame
        self.backgroundColor = backgroundColor
        self.depthTolerance = depthTolerance
        self.enablePointRejection = enablePointRejection
        self.rejectionConeThreshold = rejectionConeThreshold
        self.holeFillIterations = holeFillIterations
        self.colorizeChunks = colorizeChunks
        self.colorizeOverdraw = colorizeOverdraw
        self.writesSceneDepth = writesSceneDepth
        self.applyDisplacement = applyDisplacement
        self.applyTint = applyTint
        self.tintAlphaIsCoverage = tintAlphaIsCoverage
        self.motionBlur = motionBlur
        self.motionBlurSamples = motionBlurSamples
        self.motionBlurMaxSpread = motionBlurMaxSpread
    }
}
