// Apple9+ 64-bit nearest-point fast path: a SINGLE pass over the LOD buffer that
// replaces both the depth and index passes. For each covered pixel it does one
// atomic_max on a `device atomic_ulong*` with the packed value
//   (ulong(reverseZDepthUint) << 32) | ulong(lodIndex)
// so the max simultaneously selects the nearest depth (high word) and, among
// ties, the largest lodIndex (low word). The split pass then unpacks winners.
//
// Gated on __HAVE_ATOMIC_ULONG_MIN_MAX__ (MSL 3.1, Apple9+), the same recipe
// RasteriserCapabilities' probe validates. A no-op stub keeps the library
// compiling on toolchains/devices without the extension; the Swift side only
// dispatches this when RasteriserCapabilities.use64BitAtomics is true.
#include "../Common.metal"

// ⚠️ No inline `//` comments INSIDE this struct — Satin's uniform parser drops
// the field after a comment (set() silently no-ops). Document fields above it.
struct NearestWinner64Uniforms {
    int2 screenSize;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    int pointSizeMode;
    float minimumPointSize;
    float maximumPointSize;
    float pointSizeScale;
};

#if defined(__HAVE_ATOMIC_ULONG_MIN_MAX__)

kernel void nearestWinner64Update(
    constant NearestWinner64Uniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const packed_float3 *lodPositions [[buffer(ComputeBufferCustom0)]],
    device const RasterFile *files [[buffer(ComputeBufferCustom1)]],
    device atomic_ulong *winner [[buffer(ComputeBufferCustom2)]],
    device const uint *lodStats [[buffer(ComputeBufferCustom3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint count = lodStats[2];
    if (gid >= count) { return; }

    const float3 point = float3(lodPositions[gid]);
    const RasterFile file = files[0];

    int2 pixelCoord; uint depthUint; float ndcZ;
    if (!projectLODPoint(point, file, uniforms.screenSize, pixelCoord, depthUint, ndcZ)) { return; }
    // Never pack depth 0 (that is the "empty" sentinel in the winner buffer).
    if (depthUint == 0u) { return; }

    const ulong packed = (ulong(depthUint) << 32) | ulong(gid);
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
            atomic_max_explicit(&winner[pixelIndex], packed, memory_order_relaxed);
        }
    }
}

#else

// Fallback stub (device lacks 64-bit buffer atomics). Never dispatched by the
// Swift side in that case; present only so the library links.
kernel void nearestWinner64Update(
    constant NearestWinner64Uniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const packed_float3 *lodPositions [[buffer(ComputeBufferCustom0)]],
    device const RasterFile *files [[buffer(ComputeBufferCustom1)]],
    device ulong *winner [[buffer(ComputeBufferCustom2)]],
    device const uint *lodStats [[buffer(ComputeBufferCustom3)]],
    uint gid [[thread_position_in_grid]]
) {
    (void)uniforms; (void)lodPositions; (void)files; (void)winner; (void)lodStats; (void)gid;
}

#endif
