// Merged point-rejection + resolve (Magnopus optimization): one thread per
// pixel. For each pixel with accumulated data, run the VAST 2011 screen-space
// occlusion operator against a PR_REJECT_RADIUS neighborhood of the settled
// depth buffer, then — if the point survives — resolve it (Σcolor/count →
// output, reverse-Z depth → depth). Rejected points write transparent + 0
// depth, so background/nearer geometry shows through the gap.
//
// VAST 2011 operator (Dobrev/Rosenthal/Linsen, "Real-time Rendering of Massive
// Unstructured Raw Point Clouds using Screen-space Operators"): a point is
// visible if there is a wide empty cone, apex at the point, opening toward the
// eye. For each *closer* neighbor point Pn we measure the angle between the
// eye-ward axis (eye − Pc) and (Pn − Pc); the largest empty cone half-angle is
// the minimum of those angles. A far point leaking through a gap in a nearer
// surface has closer neighbors sitting almost directly between it and the eye
// (small angle → narrow cone → rejected); a genuine surface point's closer
// neighbors are tangential (near-90° angle → wide cone → kept).
#include "../Common.metal"

// ⚠️ No inline `//` comments INSIDE this struct — Satin's uniform parser drops
// the field after a comment (set() silently no-ops). Document fields above it.
// `invProjectionMatrix` maps clip → view for depth reconstruction.
// `rejectionConeThreshold` is the minimum empty-cone half-angle in radians.
// `isOrthographic` selects a constant eye-ward axis (parallel rays).
struct ResolveUniforms {
    int2 screenSize;
    float4 backgroundColor;
    float4x4 invProjectionMatrix;
    int enablePointRejection;
    float rejectionConeThreshold;
    int isOrthographic;
    int coverageEnabled;
};

// Reconstruct a view-space position from a buffer pixel + its reverse-Z depth.
// The depth/color passes flip Y when writing the pixel buffer, so undo that
// here; a consistent reconstruction across center + neighbors is what the cone
// test needs (a global flip preserves angles regardless).
static inline float3 reconstructViewPosition(uint2 pix, uint depthUint, int2 screenSize, float4x4 invProjection) {
    const float ndcZ = uintToDepthReverseZ(depthUint);
    const int screenY = screenSize.y - 1 - int(pix.y);
    const float2 ndcXY = float2(
        (float(pix.x) + 0.5) / float(screenSize.x),
        (float(screenY) + 0.5) / float(screenSize.y)
    ) * 2.0 - 1.0;
    const float4 view = invProjection * float4(ndcXY, ndcZ, 1.0);
    return view.xyz / view.w;
}

kernel void resolveUpdate(
    constant ResolveUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const RasterPixel *pixels [[buffer(ComputeBufferCustom0)]],
    texture2d<float, access::write> outputTexture [[texture(ComputeTextureCustom0)]],
    texture2d<float, access::write> depthTexture [[texture(ComputeTextureCustom1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) {
        return;
    }

    const int2 screenSize = uniforms.screenSize;
    const uint pixelIndex = gid.y * uint(screenSize.x) + gid.x;
    const RasterPixel pixel = pixels[pixelIndex];

    if (pixel.count == 0u) {
        outputTexture.write(float4(uniforms.backgroundColor.rgb, 0.0), gid);
        depthTexture.write(float4(0.0), gid);
        return;
    }

    // Weighted-blended OIT resolve (translucent defocus / motion blur). Fixed
    // point S=4096: red/green/blue = Σ(colour·α·S), count = Σ(α·S). Colour is the
    // α-weighted average (S cancels); pixel alpha is the additive revealage
    // 1 − e^(−Σα). No point rejection here (all depth layers accumulated).
    if (uniforms.coverageEnabled != 0) {
        const float sumA = float(pixel.count) * (1.0 / 4096.0);
        const float3 rgb = float3(pixel.red, pixel.green, pixel.blue) / float(pixel.count);
        outputTexture.write(float4(rgb, saturate(1.0 - exp(-sumA))), gid);
        depthTexture.write(float4(uintToDepthReverseZ(pixel.depth)), gid);
        return;
    }

    // Point rejection (VAST 2011 empty-cone operator).
    if (uniforms.enablePointRejection != 0 && pixel.depth != 0u) {
        const float3 pc = reconstructViewPosition(gid, pixel.depth, screenSize, uniforms.invProjectionMatrix);
        // Eye-ward axis: eye − Pc (eye at origin in view space). Orthographic
        // projection has parallel rays, so the axis is constant view +Z.
        const float3 axisToEye = (uniforms.isOrthographic != 0)
            ? float3(0.0, 0.0, 1.0)
            : normalize(-pc);

        float coneHalfAngle = M_PI_F;
        bool hadCloserNeighbor = false;
        const int radius = PR_REJECT_RADIUS;
        for (int oy = -radius; oy <= radius; oy++) {
            for (int ox = -radius; ox <= radius; ox++) {
                if (ox == 0 && oy == 0) { continue; }
                const int2 s = int2(gid) + int2(ox, oy);
                if (s.x < 0 || s.x >= screenSize.x || s.y < 0 || s.y >= screenSize.y) { continue; }
                const RasterPixel np = pixels[uint(s.y) * uint(screenSize.x) + uint(s.x)];
                if (np.count == 0u || np.depth == 0u) { continue; }
                // Reverse-Z: larger depth uint == nearer the eye. Only closer
                // neighbors can occlude this point.
                if (np.depth <= pixel.depth) { continue; }

                const float3 pn = reconstructViewPosition(uint2(s), np.depth, screenSize, uniforms.invProjectionMatrix);
                const float3 w = pn - pc;
                const float wl = length(w);
                if (wl < 1e-6) { continue; }
                const float angle = acos(clamp(dot(axisToEye, w / wl), -1.0, 1.0));
                coneHalfAngle = min(coneHalfAngle, angle);
                hadCloserNeighbor = true;
            }
        }

        if (hadCloserNeighbor && coneHalfAngle < uniforms.rejectionConeThreshold) {
            outputTexture.write(float4(uniforms.backgroundColor.rgb, 0.0), gid);
            depthTexture.write(float4(0.0), gid);
            return;
        }
    }

    const float invCount = 1.0 / float(pixel.count);
    const float3 rgb = float3(pixel.red, pixel.green, pixel.blue) * invCount / 255.0;
    outputTexture.write(float4(rgb, 1.0), gid);
    depthTexture.write(float4(uintToDepthReverseZ(pixel.depth)), gid);
}
