// Color pass over the compacted LOD point cloud: one thread per surviving LOD
// point. Re-projects (same transform as the depth pass) and accumulates color.
//
// Three per-point features (looked up in pack order via lodSourceIndices),
// ported from Satin-ComputeRasteriser: displacement (add + NaN cull), tint
// (mix / discard sentinel / translucent-defocus coverage), and motion blur
// (sweep sub-splats along screen-space velocity). Translucent defocus and motion
// blur route through weighted-blended OIT (Σcolor·α, Σα, no depth test); the
// plain path is the depth-tested integer color sum.
//
// PR_SIMD_AGGREGATION reduces the center pixel within the simdgroup only on the
// plain, feature-free path (order-independent integer add → identical buffer).
#include "../Common.metal"

// ⚠️ No inline `//` comments INSIDE this struct — Satin's uniform parser drops
// the field after a comment (set() silently no-ops). Document fields above it.
struct ColorPassUniforms {
    int2 screenSize;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    int pointSizeMode;
    float minimumPointSize;
    float maximumPointSize;
    float pointSizeScale;
    float depthTolerance;
    int colorizeChunks;
    int colorizeOverdraw;
    int applyDisplacement;
    int applyTint;
    int tintAlphaIsCoverage;
    float motionBlur;
    int motionBlurSamples;
    float motionBlurMaxSpread;
    int antialiasEdges;
};

kernel void colorPassUpdate(
    constant ColorPassUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const packed_float3 *lodPositions [[buffer(ComputeBufferCustom0)]],
    device const uint *lodColors [[buffer(ComputeBufferCustom1)]],
    device const RasterFile *files [[buffer(ComputeBufferCustom2)]],
    device RasterPixel *pixels [[buffer(ComputeBufferCustom3)]],
    device const uint *lodStats [[buffer(ComputeBufferCustom4)]],
    device const uint *lodSourceIndices [[buffer(ComputeBufferCustom5)]],
    device const float3 *displacements [[buffer(ComputeBufferCustom6)]],
    device const float4 *tints [[buffer(ComputeBufferCustom7)]],
    device const float3 *prevDisplacements [[buffer(ComputeBufferCustom8)]],
    uint gid [[thread_position_in_grid]],
    uint simdLane [[thread_index_in_simdgroup]]
) {
    const uint count = lodStats[2];
    const bool coverageMode = (uniforms.applyTint != 0 && uniforms.tintAlphaIsCoverage != 0);
    const bool motionBlurOn = uniforms.motionBlur > 0.0;
    const bool anyFeature = (uniforms.applyDisplacement != 0) || (uniforms.applyTint != 0) || motionBlurOn || (uniforms.antialiasEdges != 0);

#if PR_SIMD_AGGREGATION
    if (!anyFeature) {
        // --- Aggregated fast path (plain depth-tested color, no features) ---
        bool valid = gid < count;
        int2 pixelCoord = int2(0);
        int radius = 0;
        float ndcZ = 0.0;
        uint r = 0u, g = 0u, b = 0u;
        if (valid) {
            const float3 point = float3(lodPositions[gid]);
            const RasterFile file = files[0];
            uint depthUint;
            if (projectLODPoint(point, file, uniforms.screenSize, pixelCoord, depthUint, ndcZ)) {
                radius = pointFootprintRadius(point, file, uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize, uniforms.pointSizeMode, uniforms.minimumPointSize, uniforms.maximumPointSize, uniforms.pointSizeScale);
                uint color = lodColors[gid];
                if (uniforms.colorizeChunks != 0) { color = (lodSourceIndices[gid] >> 13) * 1234567u; }
                else if (uniforms.colorizeOverdraw != 0) { color = 0x00010101u; }
                r = color & 0xffu; g = (color >> 8) & 0xffu; b = (color >> 16) & 0xffu;
            } else { valid = false; }
        }
        const uint SENT = 0xffffffffu;
        uint centerPix = SENT;
        if (valid) {
            const uint ci = uint(pixelCoord.y * uniforms.screenSize.x + pixelCoord.x);
            const uint cd = pixels[ci].depth;
            if (cd != 0u && ndcZ >= uintToDepthReverseZ(cd) * (1.0 - uniforms.depthTolerance)) { centerPix = ci; }
        }
        const uint mr = (centerPix != SENT) ? r : 0u;
        const uint mg = (centerPix != SENT) ? g : 0u;
        const uint mb = (centerPix != SENT) ? b : 0u;
        const uint mc = (centerPix != SENT) ? 1u : 0u;
        uint sr = mr, sg = mg, sb = mb, sc = mc;
        uint leader = simdLane;
        for (uint o = 0u; o < 32u; o++) {
            const uint oPix = simd_shuffle(centerPix, ushort(o));
            if (oPix == centerPix && centerPix != SENT) {
                leader = min(leader, o);
                if (o != simdLane) {
                    sr += simd_shuffle(mr, ushort(o)); sg += simd_shuffle(mg, ushort(o));
                    sb += simd_shuffle(mb, ushort(o)); sc += simd_shuffle(mc, ushort(o));
                }
            }
        }
        if (centerPix != SENT && simdLane == leader) {
            atomic_fetch_add_explicit((device atomic_uint *)&pixels[centerPix].red, sr, memory_order_relaxed);
            atomic_fetch_add_explicit((device atomic_uint *)&pixels[centerPix].green, sg, memory_order_relaxed);
            atomic_fetch_add_explicit((device atomic_uint *)&pixels[centerPix].blue, sb, memory_order_relaxed);
            atomic_fetch_add_explicit((device atomic_uint *)&pixels[centerPix].count, sc, memory_order_relaxed);
        }
        if (valid) {
            for (int oy = -radius; oy <= radius; oy++) {
                for (int ox = -radius; ox <= radius; ox++) {
                    if (ox == 0 && oy == 0) { continue; }
                    if (!insidePointFootprint(int2(ox, oy), radius)) { continue; }
                    const int2 target = pixelCoord + int2(ox, oy);
                    if (target.x < 0 || target.x >= uniforms.screenSize.x || target.y < 0 || target.y >= uniforms.screenSize.y) { continue; }
                    const uint pixelIndex = uint(target.y * uniforms.screenSize.x + target.x);
                    const uint cd = pixels[pixelIndex].depth;
                    if (cd == 0u) { continue; }
                    if (ndcZ < uintToDepthReverseZ(cd) * (1.0 - uniforms.depthTolerance)) { continue; }
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].red, r, memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].green, g, memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].blue, b, memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].count, 1u, memory_order_relaxed);
                }
            }
        }
        return;
    }
#endif

    // --- Plain path with per-point features ---
    if (gid >= count) { return; }
    const uint pointIndex = lodSourceIndices[gid];
    const float3 base = float3(lodPositions[gid]);
    float3 point = base;
    const RasterFile file = files[0];

    if (uniforms.applyDisplacement != 0) {
        point += displacements[pointIndex];
        if (any(isnan(point))) { return; }
    }

    int2 pixelCoord; uint depthUint; float ndcZ;
    if (!projectLODPoint(point, file, uniforms.screenSize, pixelCoord, depthUint, ndcZ)) { return; }

    uint color = lodColors[gid];
    if (uniforms.colorizeChunks != 0) { color = (lodSourceIndices[gid] >> 13) * 1234567u; }
    else if (uniforms.colorizeOverdraw != 0) { color = 0x00010101u; }
    uint r = color & 0xffu;
    uint g = (color >> 8) & 0xffu;
    uint b = (color >> 16) & 0xffu;

    const bool oit = coverageMode || motionBlurOn;
    float coverage = 1.0;
    if (uniforms.applyTint != 0) {
        const float4 tint = tints[pointIndex];
        if (tint.a < 0.0) { return; } // discard sentinel
        if (coverageMode) {
            coverage = saturate(1.0 - tint.a);
        } else {
            const float w = saturate(tint.a);
            const float3 orig = float3(float(r), float(g), float(b)) * (1.0 / 255.0);
            const float3 mixed = mix(orig, saturate(tint.rgb), w);
            r = uint(saturate(mixed.x) * 255.0 + 0.5);
            g = uint(saturate(mixed.y) * 255.0 + 0.5);
            b = uint(saturate(mixed.z) * 255.0 + 0.5);
        }
    }
    const float a = coverage;
    if (oit && a <= 0.0) { return; }

    const int radius = pointFootprintRadius(point, file, uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize, uniforms.pointSizeMode, uniforms.minimumPointSize, uniforms.maximumPointSize, uniforms.pointSizeScale);

    // Motion blur: sweep the splat back toward its previous screen position
    // (camera velocity via prevTransform, plus displacement velocity).
    int mbSamples = 1;
    float2 mbStep = float2(0.0);
    if (motionBlurOn) {
        float3 prevPoint = base;
        if (uniforms.applyDisplacement != 0) { prevPoint += prevDisplacements[pointIndex]; }
        const float4 prevClip = file.prevTransform * float4(prevPoint, 1.0);
        if (prevClip.w > 0.0) {
            const float3 prevNdc = prevClip.xyz / prevClip.w;
            int2 prevPixel = int2((prevNdc.xy * 0.5 + 0.5) * float2(uniforms.screenSize));
            prevPixel.y = uniforms.screenSize.y - 1 - prevPixel.y;
            float2 vel = float2(pixelCoord - prevPixel) * uniforms.motionBlur;
            float len = length(vel);
            if (len > uniforms.motionBlurMaxSpread) { vel *= uniforms.motionBlurMaxSpread / len; len = uniforms.motionBlurMaxSpread; }
            if (len > 0.75) {
                mbSamples = clamp(int(ceil(len)), 1, uniforms.motionBlurSamples);
                mbStep = vel / float(max(mbSamples - 1, 1));
            }
        }
    }

    const bool aa = uniforms.antialiasEdges != 0;
    const float aN = a / float(mbSamples);
    const float kS = 4096.0;
    // Constant fixed-point contribution for the OIT path (per-pixel weight = aN).
    const uint caR0 = uint(float(r) * (1.0 / 255.0) * aN * kS + 0.5);
    const uint caG0 = uint(float(g) * (1.0 / 255.0) * aN * kS + 0.5);
    const uint caB0 = uint(float(b) * (1.0 / 255.0) * aN * kS + 0.5);
    const uint caA0 = uint(aN * kS + 0.5);
    // Normalized color for the edge-AA path (per-pixel coverage weight).
    const float3 cN = float3(float(r), float(g), float(b)) * (1.0 / 255.0);

    for (int s = 0; s < mbSamples; s++) {
        const int2 center = pixelCoord - int2(round(mbStep * float(s)));
        for (int oy = -radius; oy <= radius; oy++) {
            for (int ox = -radius; ox <= radius; ox++) {
                if (!insidePointFootprint(int2(ox, oy), radius)) { continue; }
                const int2 target = center + int2(ox, oy);
                if (target.x < 0 || target.x >= uniforms.screenSize.x || target.y < 0 || target.y >= uniforms.screenSize.y) { continue; }
                const uint pixelIndex = uint(target.y * uniforms.screenSize.x + target.x);

                // Opaque accumulation (incl. edge-AA) is depth-tested; OIT is not.
                if (!oit) {
                    const uint cd = pixels[pixelIndex].depth;
                    if (cd == 0u) { continue; }
                    if (ndcZ < uintToDepthReverseZ(cd) * (1.0 - uniforms.depthTolerance)) { continue; }
                }

                if (aa) {
                    const float cov = pointFootprintCoverage(int2(ox, oy), radius);
                    if (cov <= 0.0) { continue; }
                    const float w = cov * aN;
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].red, uint(cN.x * w * kS + 0.5), memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].green, uint(cN.y * w * kS + 0.5), memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].blue, uint(cN.z * w * kS + 0.5), memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].count, uint(w * kS + 0.5), memory_order_relaxed);
                } else if (oit) {
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].red, caR0, memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].green, caG0, memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].blue, caB0, memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].count, caA0, memory_order_relaxed);
                } else {
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].red, r, memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].green, g, memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].blue, b, memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].count, 1u, memory_order_relaxed);
                }
            }
        }
    }
}
