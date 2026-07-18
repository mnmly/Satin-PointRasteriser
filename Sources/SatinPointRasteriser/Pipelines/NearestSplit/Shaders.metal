// Unpack the 64-bit winner buffer into the shared uint depth + index buffers, so
// the nearest reject+resolve pass reads the same representation as the portable
// fallback path (which writes those two buffers directly).
#include "../Common.metal"

struct NearestSplitUniforms {
    int pixelCount;
};

kernel void nearestSplitUpdate(
    constant NearestSplitUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const ulong *winner [[buffer(ComputeBufferCustom0)]],
    device uint *depths [[buffer(ComputeBufferCustom1)]],
    device uint *indices [[buffer(ComputeBufferCustom2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(uniforms.pixelCount)) { return; }
    const ulong packed = winner[gid];
    if (packed == 0ul) {
        depths[gid] = 0u;
        indices[gid] = 0xffffffffu;
        return;
    }
    depths[gid] = uint(packed >> 32);
    indices[gid] = uint(packed & 0xffffffffu);
}
