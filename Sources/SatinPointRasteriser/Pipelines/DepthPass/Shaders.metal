// Depth pass over the compacted LOD point cloud: one thread per surviving LOD
// point. Read its object-space position, optionally add per-point displacement
// (looked up in pack order via lodSourceIndices), project through the file
// transform, splat the footprint, and atomic_fetch_max the reverse-Z depth.
//
// With PR_SIMD_AGGREGATION defined AND no per-point features active, the center
// pixel write is reduced within the simdgroup (order-independent max → identical
// buffer). Displacement / translucent-defocus paths take the plain footprint
// loop (they need per-point NaN culls / depth skips the aggregation can't fold).
#include "../Common.metal"

// ⚠️ No inline `//` comments INSIDE this struct — Satin's uniform parser drops
// the field after a comment (set() silently no-ops). Document fields above it.
struct DepthPassUniforms {
    int2 screenSize;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    int pointSizeMode;
    float minimumPointSize;
    float maximumPointSize;
    float pointSizeScale;
    int applyDisplacement;
    int applyTint;
    int tintAlphaIsCoverage;
};

kernel void depthPassUpdate(
    constant DepthPassUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const packed_float3 *lodPositions [[buffer(ComputeBufferCustom0)]],
    device const RasterFile *files [[buffer(ComputeBufferCustom1)]],
    device RasterPixel *pixels [[buffer(ComputeBufferCustom2)]],
    device const uint *lodStats [[buffer(ComputeBufferCustom3)]],
    device const uint *lodSourceIndices [[buffer(ComputeBufferCustom4)]],
    device const float3 *displacements [[buffer(ComputeBufferCustom5)]],
    device const float4 *tints [[buffer(ComputeBufferCustom6)]],
    uint gid [[thread_position_in_grid]],
    uint simdLane [[thread_index_in_simdgroup]]
) {
    const uint count = lodStats[2];
    const bool coverageMode = (uniforms.applyTint != 0 && uniforms.tintAlphaIsCoverage != 0);
    const bool anyFeature = (uniforms.applyDisplacement != 0) || coverageMode;

#if PR_SIMD_AGGREGATION
    if (!anyFeature) {
        // --- Aggregated fast path (no per-point features) ---
        bool valid = gid < count;
        int2 pixelCoord = int2(0);
        uint depth = 0u;
        int radius = 0;
        if (valid) {
            const float3 point = float3(lodPositions[gid]);
            const RasterFile file = files[0];
            float ndcZ;
            if (projectLODPoint(point, file, uniforms.screenSize, pixelCoord, depth, ndcZ)) {
                radius = pointFootprintRadius(point, file, uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize, uniforms.pointSizeMode, uniforms.minimumPointSize, uniforms.maximumPointSize, uniforms.pointSizeScale);
            } else { valid = false; }
        }
        const uint SENT = 0xffffffffu;
        const uint myPix = valid ? uint(pixelCoord.y * uniforms.screenSize.x + pixelCoord.x) : SENT;
        uint reduced = depth;
        uint leader = simdLane;
        for (uint o = 0u; o < 32u; o++) {
            const uint oPix = simd_shuffle(myPix, ushort(o));
            const uint oDepth = simd_shuffle(depth, ushort(o));
            if (oPix == myPix && myPix != SENT) {
                leader = min(leader, o);
                if (o != simdLane) { reduced = max(reduced, oDepth); }
            }
        }
        if (myPix != SENT && simdLane == leader) {
            atomic_fetch_max_explicit((device atomic_uint *)&pixels[myPix].depth, reduced, memory_order_relaxed);
        }
        if (valid) {
            for (int oy = -radius; oy <= radius; oy++) {
                for (int ox = -radius; ox <= radius; ox++) {
                    if (ox == 0 && oy == 0) { continue; }
                    if (!insidePointFootprint(int2(ox, oy), radius)) { continue; }
                    const int2 target = pixelCoord + int2(ox, oy);
                    if (target.x < 0 || target.x >= uniforms.screenSize.x || target.y < 0 || target.y >= uniforms.screenSize.y) { continue; }
                    atomic_fetch_max_explicit((device atomic_uint *)&pixels[uint(target.y * uniforms.screenSize.x + target.x)].depth, depth, memory_order_relaxed);
                }
            }
        }
        return;
    }
#endif

    // --- Plain path (per-point displacement / coverage) ---
    if (gid >= count) { return; }
    const uint pointIndex = lodSourceIndices[gid];
    float3 point = float3(lodPositions[gid]);
    const RasterFile file = files[0];

    if (uniforms.applyDisplacement != 0) {
        point += displacements[pointIndex];
        if (any(isnan(point))) { return; } // NaN displacement = drop the point
    }
    // Translucent defocus: the discard-sentinel point must not occlude, so skip
    // its depth write (all other points still write depth as the nearest surface).
    if (coverageMode && tints[pointIndex].a < 0.0) { return; }

    int2 pixelCoord; uint depth; float ndcZ;
    if (!projectLODPoint(point, file, uniforms.screenSize, pixelCoord, depth, ndcZ)) { return; }

    const int radius = pointFootprintRadius(point, file, uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize, uniforms.pointSizeMode, uniforms.minimumPointSize, uniforms.maximumPointSize, uniforms.pointSizeScale);
    for (int oy = -radius; oy <= radius; oy++) {
        for (int ox = -radius; ox <= radius; ox++) {
            if (!insidePointFootprint(int2(ox, oy), radius)) { continue; }
            const int2 target = pixelCoord + int2(ox, oy);
            if (target.x < 0 || target.x >= uniforms.screenSize.x || target.y < 0 || target.y >= uniforms.screenSize.y) { continue; }
            atomic_fetch_max_explicit((device atomic_uint *)&pixels[uint(target.y * uniforms.screenSize.x + target.x)].depth, depth, memory_order_relaxed);
        }
    }
}
