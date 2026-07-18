// Zero the screen-sized RasterPixel accumulation buffer before each frame's
// depth/color passes.
#include "../Common.metal"

struct ClearUniforms {
    int pixelCount;
};

kernel void clearUpdate(
    constant ClearUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device RasterPixel *pixels [[buffer(ComputeBufferCustom0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(uniforms.pixelCount)) {
        return;
    }

    pixels[gid].depth = 0u;
    pixels[gid].red = 0u;
    pixels[gid].green = 0u;
    pixels[gid].blue = 0u;
    pixels[gid].count = 0u;
    pixels[gid].weight = 0u;
    pixels[gid].padding = uint2(0);
}
