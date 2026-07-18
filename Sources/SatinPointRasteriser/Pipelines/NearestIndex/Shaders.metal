// Portable nearest-point fallback, pass 2 of 2: for each covered pixel whose
// settled depth matches this point's, record the point via atomic_fetch_min on
// the lodIndex (ties resolve to the smallest lodIndex). Note the 64-bit fast
// path instead breaks ties toward the LARGEST lodIndex (see NearestWinner64);
// the images agree on depth everywhere and on color for all but the vanishing
// fraction of pixels where two points land at the exact same reverse-Z depth.
#include "../Common.metal"

// ⚠️ No inline `//` comments INSIDE this struct — Satin's uniform parser drops
// the field after a comment (set() silently no-ops). Document fields above it.
struct NearestIndexUniforms {
    int2 screenSize;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    int pointSizeMode;
    float minimumPointSize;
    float maximumPointSize;
    float pointSizeScale;
};

kernel void nearestIndexUpdate(
    constant NearestIndexUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const packed_float3 *lodPositions [[buffer(ComputeBufferCustom0)]],
    device const RasterFile *files [[buffer(ComputeBufferCustom1)]],
    device const uint *depths [[buffer(ComputeBufferCustom2)]],
    device atomic_uint *indices [[buffer(ComputeBufferCustom3)]],
    device const uint *lodStats [[buffer(ComputeBufferCustom4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint count = lodStats[2];
    if (gid >= count) { return; }

    const float3 point = float3(lodPositions[gid]);
    const RasterFile file = files[0];

    int2 pixelCoord; uint depthUint; float ndcZ;
    if (!projectLODPoint(point, file, uniforms.screenSize, pixelCoord, depthUint, ndcZ)) { return; }
    if (depthUint == 0u) { return; }

    const int radius = pointFootprintRadius(
        point, file,
        uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize,
        uniforms.pointSizeMode,
        uniforms.minimumPointSize, uniforms.maximumPointSize, uniforms.pointSizeScale
    );

    for (int oy = -radius; oy <= radius; oy++) {
        for (int ox = -radius; ox <= radius; ox++) {
            if (!insidePointFootprint(int2(ox, oy), radius)) { continue; }
            const int2 target = pixelCoord + int2(ox, oy);
            if (target.x < 0 || target.x >= uniforms.screenSize.x || target.y < 0 || target.y >= uniforms.screenSize.y) { continue; }
            const uint pixelIndex = uint(target.y * uniforms.screenSize.x + target.x);
            if (depthUint == depths[pixelIndex]) {
                atomic_fetch_min_explicit(&indices[pixelIndex], gid, memory_order_relaxed);
            }
        }
    }
}
