// LOD compaction ("LOD point cloud" generation), the Magnopus architecture's
// amortized front end. One threadgroup per source batch (PR_THREADS_PER_GROUP
// threads), a full sweep every frame in this slice: frustum-cull the batch AABB
// (whole-threadgroup early-out), pick the pixel-footprint precision level
// (0/1/2 → 30/20/10-bit decode), then per surviving point (CLOD threshold +
// hashUnit dither) decode its object-space position and append
// {position, color, pack-order index} into the cloud's compacted SoA LOD
// buffers via a single atomic append. Downstream depth/color passes rasterize
// straight from those buffers instead of re-decoding + re-LOD-testing the
// source every pass (the sibling Satin-ComputeRasteriser's per-frame cost).
#include "../Common.metal"

// ⚠️ No inline `//` comments INSIDE this struct — Satin's uniform parser drops
// the field after a comment (set() silently no-ops). Document fields above it.
struct LODSelectUniforms {
    int2 screenSize;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    int enableFrustumCulling;
    int lodBias;
    int enableCLOD;
    int lodDither;
};

kernel void lodselectUpdate(
    constant LODSelectUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const RasterBatch *batches [[buffer(ComputeBufferCustom0)]],
    device const uint *xyzLow [[buffer(ComputeBufferCustom1)]],
    device const uint *xyzMed [[buffer(ComputeBufferCustom2)]],
    device const uint *xyzHigh [[buffer(ComputeBufferCustom3)]],
    device const RasterFile *files [[buffer(ComputeBufferCustom4)]],
    device const uint *colors [[buffer(ComputeBufferCustom5)]],
    device const uchar *levels [[buffer(ComputeBufferCustom6)]],
    device packed_float3 *lodPositions [[buffer(ComputeBufferCustom7)]],
    device uint *lodColors [[buffer(ComputeBufferCustom8)]],
    device uint *lodSourceIndices [[buffer(ComputeBufferCustom9)]],
    // lodStats[0] = append cursor / survivor count, lodStats[1] = overflow count
    // (points dropped for want of capacity), lodStats[2] = clamped count written
    // by the finalize pass.
    device atomic_uint *lodStats [[buffer(ComputeBufferCustom10)]],
    constant uint &lodCapacity [[buffer(12)]],
    // Index of the first source batch this dispatch processes; the threadgroup
    // at grid position `g` handles batch `firstBatch + g`. Non-zero when the
    // sweep is amortized across frames (resuming from a persistent cursor).
    constant uint &firstBatch [[buffer(13)]],
    uint slot [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    const RasterBatch batch = batches[firstBatch + slot];
    // Residency gate (fixtures are always resident; kept for streaming parity).
    if (batch.state == 0) {
        return;
    }
    const RasterFile file = files[batch.fileIndex];

    if (uniforms.enableFrustumCulling != 0) {
        if (!intersectsFrustum(file.transformFrustum, batchMin(batch), batchMax(batch))) {
            return;
        }
    }

    const float pixelSize = pixelSizeOnScreen(batch, file, uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);
    int level;
    if (pixelSize < 100.0) level = 4;
    else if (pixelSize < 200.0) level = 3;
    else if (pixelSize < 500.0) level = 2;
    else if (pixelSize < 10000.0) level = 1;
    else level = 0;

    // CLOD off: sentinel that's always ≥ dither(1.0) + pointLevel(7) + 0.5, so
    // the per-point keep-test never culls.
    const float lodThreshold = (uniforms.enableCLOD != 0)
        ? max(0.0, lodThresholdFromPixelSize(pixelSize, uniforms.lodBias))
        : 99.0;

    // Survivor prefix: level-bucketed batches store points level-ascending with
    // cumulative counts in lodCum01..67, so the keep-test can only pass within
    // cum[Lmax]. Legacy batches (lodCum67 == 0) walk the full numPoints range.
    uint activePoints = batch.numPoints;
    if (batch.lodCum67 != 0u) {
        const int lmax = clamp(int(floor(lodThreshold + 0.5)), 0, 7);
        const uint words[4] = { batch.lodCum01, batch.lodCum23, batch.lodCum45, batch.lodCum67 };
        const uint word = words[lmax >> 1];
        activePoints = ((lmax & 1) != 0) ? (word >> 16) : (word & 0xffffu);
    }
    if (activePoints == 0u) {
        return;
    }

    const uint pointsPerThread = (activePoints + PR_THREADS_PER_GROUP - 1u) / PR_THREADS_PER_GROUP;
    for (uint i = 0; i < pointsPerThread; i++) {
        const uint localIndex = i * PR_THREADS_PER_GROUP + lid;
        if (localIndex >= activePoints) {
            continue;
        }

        const uint pointIndex = batch.firstPoint + localIndex;
        const float pointLevel = float(uint(levels[pointIndex]) & 0x7u);
        const float dither = (uniforms.lodDither != 0) ? hashUnit(pointIndex) : 0.5;
        if (dither >= lodThreshold - pointLevel + 0.5) {
            continue;
        }

        const float3 point = decodePoint(pointIndex, batch, xyzLow, xyzMed, xyzHigh, level);

        // Reserve a slot in the compacted LOD buffers. Plain per-point atomic
        // append (a per-simdgroup ballot reservation is a slice-4 optimization).
        const uint dst = atomic_fetch_add_explicit(&lodStats[0], 1u, memory_order_relaxed);
        if (dst >= lodCapacity) {
            atomic_fetch_add_explicit(&lodStats[1], 1u, memory_order_relaxed);
            continue;
        }
        lodPositions[dst] = point;
        // Color stored as-is (packed RGBA8); the point's LOD level byte is
        // preserved in the alpha channel of the source color word.
        lodColors[dst] = colors[pointIndex];
        lodSourceIndices[dst] = pointIndex;
    }
}
