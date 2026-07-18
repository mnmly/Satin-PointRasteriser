// Neighbor-average hole filling, run for N ping-ponged iterations. Valid pixels
// (α ≥ 0.5) pass through unchanged; empty pixels adopt the mean color AND mean
// reverse-Z depth of their valid 3×3 neighbors, so the composited depth stays
// consistent with the filled color. Ported from Satin-ComputeRasteriser's
// HoleFill (which fills color only) and extended to depth per the Magnopus
// write-up. Color and depth are ping-ponged together by the caller.
#include "../Common.metal"

struct HoleFillUniforms {
    int2 screenSize;
};

kernel void holeFillUpdate(
    constant HoleFillUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    texture2d<float, access::read> inputColor [[texture(ComputeTextureCustom0)]],
    texture2d<float, access::read> inputDepth [[texture(ComputeTextureCustom1)]],
    texture2d<float, access::write> outputColor [[texture(ComputeTextureCustom2)]],
    texture2d<float, access::write> outputDepth [[texture(ComputeTextureCustom3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) {
        return;
    }

    const float4 center = inputColor.read(gid);
    const float centerDepth = inputDepth.read(gid).r;
    if (center.a >= 0.5) {
        outputColor.write(center, gid);
        outputDepth.write(float4(centerDepth), gid);
        return;
    }

    float3 sumColor = float3(0.0);
    float sumDepth = 0.0;
    float count = 0.0;
    for (int oy = -1; oy <= 1; oy++) {
        for (int ox = -1; ox <= 1; ox++) {
            if (ox == 0 && oy == 0) { continue; }
            const int2 sample = int2(gid) + int2(ox, oy);
            if (sample.x < 0 || sample.x >= uniforms.screenSize.x || sample.y < 0 || sample.y >= uniforms.screenSize.y) {
                continue;
            }
            const float4 neighbour = inputColor.read(uint2(sample));
            if (neighbour.a >= 0.5) {
                sumColor += neighbour.rgb;
                sumDepth += inputDepth.read(uint2(sample)).r;
                count += 1.0;
            }
        }
    }

    if (count > 0.0) {
        outputColor.write(float4(sumColor / count, 1.0), gid);
        outputDepth.write(float4(sumDepth / count), gid);
    } else {
        outputColor.write(center, gid);
        outputDepth.write(float4(centerDepth), gid);
    }
}
