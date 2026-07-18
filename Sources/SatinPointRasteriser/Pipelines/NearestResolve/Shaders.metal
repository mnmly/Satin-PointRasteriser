// Nearest-point reject + resolve: one thread per pixel. Reads the settled uint
// depth + winning lodIndex buffers (populated by either the 64-bit fast path via
// the split pass, or the portable two-pass fallback), runs the same VAST 2011
// occlusion operator as the high-quality resolve against the depth neighborhood,
// and — for survivors — writes the winner's color (no averaging) + its reverse-Z
// depth. Downstream hole fill + composite are identical to the HQ path.
#include "../Common.metal"

// ⚠️ No inline `//` comments INSIDE this struct — Satin's uniform parser drops
// the field after a comment (set() silently no-ops). Document fields above it.
struct NearestResolveUniforms {
    int2 screenSize;
    float4 backgroundColor;
    float4x4 invProjectionMatrix;
    int enablePointRejection;
    float rejectionConeThreshold;
    int isOrthographic;
};

// Reconstruct a view-space position from a buffer pixel + its reverse-Z depth
// (mirrors the HQ resolve; undoes the depth pass's Y flip).
static inline float3 reconstructViewPositionU(uint2 pix, uint depthUint, int2 screenSize, float4x4 invProjection) {
    const float ndcZ = uintToDepthReverseZ(depthUint);
    const int screenY = screenSize.y - 1 - int(pix.y);
    const float2 ndcXY = float2(
        (float(pix.x) + 0.5) / float(screenSize.x),
        (float(screenY) + 0.5) / float(screenSize.y)
    ) * 2.0 - 1.0;
    const float4 view = invProjection * float4(ndcXY, ndcZ, 1.0);
    return view.xyz / view.w;
}

kernel void nearestResolveUpdate(
    constant NearestResolveUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const uint *depths [[buffer(ComputeBufferCustom0)]],
    device const uint *indices [[buffer(ComputeBufferCustom1)]],
    device const uint *lodColors [[buffer(ComputeBufferCustom2)]],
    texture2d<float, access::write> outputTexture [[texture(ComputeTextureCustom0)]],
    texture2d<float, access::write> depthTexture [[texture(ComputeTextureCustom1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) {
        return;
    }

    const int2 screenSize = uniforms.screenSize;
    const uint pixelIndex = gid.y * uint(screenSize.x) + gid.x;
    const uint centerDepth = depths[pixelIndex];
    const uint winner = indices[pixelIndex];

    if (centerDepth == 0u || winner == 0xffffffffu) {
        outputTexture.write(float4(uniforms.backgroundColor.rgb, 0.0), gid);
        depthTexture.write(float4(0.0), gid);
        return;
    }

    // Point rejection (VAST 2011 empty-cone operator) over the depth neighborhood.
    if (uniforms.enablePointRejection != 0) {
        const float3 pc = reconstructViewPositionU(gid, centerDepth, screenSize, uniforms.invProjectionMatrix);
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
                const uint nd = depths[uint(s.y) * uint(screenSize.x) + uint(s.x)];
                if (nd == 0u || nd <= centerDepth) { continue; }
                const float3 pn = reconstructViewPositionU(uint2(s), nd, screenSize, uniforms.invProjectionMatrix);
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

    const uint rgba = lodColors[winner];
    const float3 rgb = float3(
        float((rgba >> 0u) & 0xffu),
        float((rgba >> 8u) & 0xffu),
        float((rgba >> 16u) & 0xffu)
    ) / 255.0;
    outputTexture.write(float4(rgb, 1.0), gid);
    depthTexture.write(float4(uintToDepthReverseZ(centerDepth)), gid);
}
