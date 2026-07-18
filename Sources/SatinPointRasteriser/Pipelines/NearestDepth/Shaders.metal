// Portable nearest-point fallback, pass 1 of 2: settle the nearest reverse-Z
// depth per pixel with a 32-bit atomic_max over the LOD buffer (one thread per
// LOD point). Pass 2 (NearestIndex) then records the winning point.
#include "../Common.metal"

// ⚠️ No inline `//` comments INSIDE this struct — Satin's uniform parser drops
// the field after a comment (set() silently no-ops). Document fields above it.
struct NearestDepthUniforms {
    int2 screenSize;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    int pointSizeMode;
    float minimumPointSize;
    float maximumPointSize;
    float pointSizeScale;
};

kernel void nearestDepthUpdate(
    constant NearestDepthUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const packed_float3 *lodPositions [[buffer(ComputeBufferCustom0)]],
    device const RasterFile *files [[buffer(ComputeBufferCustom1)]],
    device atomic_uint *depths [[buffer(ComputeBufferCustom2)]],
    device const uint *lodStats [[buffer(ComputeBufferCustom3)]],
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
            atomic_fetch_max_explicit(&depths[pixelIndex], depthUint, memory_order_relaxed);
        }
    }
}
