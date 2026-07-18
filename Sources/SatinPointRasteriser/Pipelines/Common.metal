#include <metal_stdlib>
#include "PointRasteriserTypes.h"
using namespace metal;

struct RasterPlane {
    float3 normal;
    float offset;
};

inline uint depthToUintReverseZ(float ndcZ) {
    return uint(saturate(ndcZ) * float(PR_MAX_DEPTH - 1u));
}

inline float uintToDepthReverseZ(uint depth) {
    return float(depth) / float(PR_MAX_DEPTH - 1u);
}

// Project an object-space LOD point through its file transform to a buffer pixel
// coordinate (Y flipped to match the depth/color/nearest passes) and reverse-Z
// depth. Returns false when the point is behind the camera or off-screen.
inline bool projectLODPoint(
    float3 point,
    RasterFile file,
    int2 screenSize,
    thread int2 &pixelCoord,
    thread uint &depthUint,
    thread float &ndcZ
) {
    const float4 clip = file.transform * float4(point, 1.0);
    if (clip.w <= 0.0) { return false; }
    const float3 ndc = clip.xyz / clip.w;
    if (any(ndc.xy < -1.0) || any(ndc.xy > 1.0) || ndc.z < 0.0 || ndc.z > 1.0) { return false; }
    int2 pc = int2((ndc.xy * 0.5 + 0.5) * float2(screenSize));
    if (pc.x < 0 || pc.x >= screenSize.x || pc.y < 0 || pc.y >= screenSize.y) { return false; }
    pc.y = screenSize.y - 1 - pc.y;
    pixelCoord = pc;
    depthUint = depthToUintReverseZ(ndc.z);
    ndcZ = ndc.z;
    return true;
}

inline int pointFootprintRadius(
    float3 point,
    RasterFile file,
    float4x4 viewMatrix,
    float4x4 projectionMatrix,
    int2 screenSize,
    int pointSizeMode,
    float minimumPointSize,
    float maximumPointSize,
    float pointSizeScale
) {
    const float3 worldPoint = (file.world * float4(point, 1.0)).xyz;
    const float4 viewPos = viewMatrix * float4(worldPoint, 1.0);

    // Orthographic apparent size must not vary with depth. projectionMatrix[3][3]
    // (column 3, row 3 — Metal's float4x4 is column-major) is 1.0 for an
    // orthographic projection and 0.0 for a perspective one.
    const bool ortho = (projectionMatrix[3][3] == 1.0);

    float pointSize;
    if (pointSizeMode == 1) {
        // World space: pointSizeScale is a metric radius projected to pixels, so
        // the on-screen size varies with depth — clamp it to the Min…Max pixel
        // band (Min floored to 1px) to bound near/far extremes.
        const float lo = max(min(minimumPointSize, maximumPointSize), 1.0);
        const float hi = max(max(minimumPointSize, maximumPointSize), lo);
        const float focal = float(screenSize.y) * 0.5 * projectionMatrix[1][1];
        if (ortho) {
            pointSize = 2.0 * pointSizeScale * focal;
        } else {
            const float viewZ = max(-viewPos.z, 0.000001);
            pointSize = 2.0 * pointSizeScale * focal / viewZ;
        }
        if (!isfinite(pointSize)) { pointSize = lo; }
        pointSize = clamp(pointSize, lo, hi);
    } else {
        // Screen space: pointSizeScale IS the on-screen px diameter, uniform for
        // every point regardless of depth. Min/Max do not apply here — there is
        // no depth-varying size to bound, and clamping would let a stale
        // Min==Max override Scale entirely. The radius encoding below already
        // floors at 1px and caps the footprint at ~33px.
        pointSize = pointSizeScale;
    }
    return clamp(int(ceil(max(pointSize - 1.0, 0.0) * 0.5)), 0, 16);
}

inline bool insidePointFootprint(int2 offset, int radius) {
    if (radius <= 0) {
        return offset.x == 0 && offset.y == 0;
    }

    const float2 p = float2(offset);
    const float r = float(radius) + 0.5;
    return dot(p, p) <= r * r;
}

inline float3 batchMin(RasterBatch batch) {
    return float3(batch.minX, batch.minY, batch.minZ);
}

inline float3 batchMax(RasterBatch batch) {
    return float3(batch.maxX, batch.maxY, batch.maxZ);
}

inline float3 batchSize(RasterBatch batch) {
    return max(batchMax(batch) - batchMin(batch), float3(0.000001));
}

inline uint unpack10(uint encoded, uint shift) {
    return (encoded >> shift) & PR_MASK_10BIT;
}

inline float3 decodePoint(uint pointIndex, RasterBatch batch, device const uint *xyzLow, device const uint *xyzMed, device const uint *xyzHigh, int level) {
    const float3 wgMin = batchMin(batch);
    const float3 wgSize = batchSize(batch);

    if (level == 0) {
        const uint low = xyzLow[pointIndex];
        const uint med = xyzMed[pointIndex];
        const uint high = xyzHigh[pointIndex];
        const uint x = (unpack10(low, 0) << 20) | (unpack10(med, 0) << 10) | unpack10(high, 0);
        const uint y = (unpack10(low, 10) << 20) | (unpack10(med, 10) << 10) | unpack10(high, 10);
        const uint z = (unpack10(low, 20) << 20) | (unpack10(med, 20) << 10) | unpack10(high, 20);
        return float3(x, y, z) * (wgSize / PR_STEPS_30BIT) + wgMin;
    }

    if (level == 1) {
        const uint low = xyzLow[pointIndex];
        const uint med = xyzMed[pointIndex];
        const uint x = (unpack10(low, 0) << 10) | unpack10(med, 0);
        const uint y = (unpack10(low, 10) << 10) | unpack10(med, 10);
        const uint z = (unpack10(low, 20) << 10) | unpack10(med, 20);
        return float3(x, y, z) * (wgSize / PR_STEPS_20BIT) + wgMin;
    }

    const uint low = xyzLow[pointIndex];
    const uint x = unpack10(low, 0);
    const uint y = unpack10(low, 10);
    const uint z = unpack10(low, 20);
    return float3(x, y, z) * (wgSize / PR_STEPS_10BIT) + wgMin;
}

inline RasterPlane makePlane(float x, float y, float z, float w) {
    const float nLength = max(length(float3(x, y, z)), 0.000001);
    RasterPlane plane;
    plane.normal = float3(x, y, z) / nLength;
    plane.offset = w / nLength;
    return plane;
}

inline bool intersectsFrustum(float4x4 m, float3 wgMin, float3 wgMax) {
    RasterPlane planes[6] = {
        makePlane(m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0]),
        makePlane(m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0]),
        makePlane(m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1]),
        makePlane(m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1]),
        makePlane(m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2]),
        // Near plane uses Metal's z ∈ [0,1] clip convention (just the z row),
        // not OpenGL's z ∈ [-1,1] form (w + z rows). The latter over-culls near
        // the camera, badly so for orthographic projections where the w row is a
        // pure constant offset rather than depth-dependent.
        makePlane(m[0][2], m[1][2], m[2][2], m[3][2]),
    };

    for (int i = 0; i < 6; i++) {
        const RasterPlane plane = planes[i];
        const float3 p = float3(
            plane.normal.x > 0.0 ? wgMax.x : wgMin.x,
            plane.normal.y > 0.0 ? wgMax.y : wgMin.y,
            plane.normal.z > 0.0 ? wgMax.z : wgMin.z
        );
        if (dot(plane.normal, p) + plane.offset < 0.0) {
            return false;
        }
    }
    return true;
}

inline float pixelSizeOnScreen(RasterBatch batch, RasterFile file, float4x4 viewMatrix, float4x4 projectionMatrix, int2 imageSize) {
    const float3 wgMin = batchMin(batch);
    const float3 wgMax = batchMax(batch);
    const float3 wgCenter = (wgMin + wgMax) * 0.5;
    const float wgRadius = distance(wgMin, wgMax);

    const float4 viewCenter = viewMatrix * file.world * float4(wgCenter, 1.0);
    const float4 viewEdge = viewCenter + float4(wgRadius, 0.0, 0.0, 0.0);
    float4 projCenter = projectionMatrix * viewCenter;
    float4 projEdge = projectionMatrix * viewEdge;

    projCenter.xy /= max(abs(projCenter.w), 0.000001);
    projEdge.xy /= max(abs(projEdge.w), 0.000001);

    const float2 screenCenter = float2(imageSize) * (projCenter.xy + 1.0) * 0.5;
    const float2 screenEdge = float2(imageSize) * (projEdge.xy + 1.0) * 0.5;
    return distance(screenEdge, screenCenter);
}

inline int precisionLevel(RasterBatch batch, RasterFile file, float4x4 viewMatrix, float4x4 projectionMatrix, int2 imageSize) {
    const float pixelSize = pixelSizeOnScreen(batch, file, viewMatrix, projectionMatrix, imageSize);
    if (pixelSize < 100.0) { return 4; }
    if (pixelSize < 200.0) { return 3; }
    if (pixelSize < 500.0) { return 2; }
    if (pixelSize < 10000.0) { return 1; }
    return 0;
}

// Continuous LOD threshold: log2-of-pixelSize ramp with a baseline that keeps
// typical viewing distances at full detail. lodBias shifts the ramp.
inline float lodThresholdFromPixelSize(float pixelSize, int lodBias) {
    return log2(max(pixelSize, 1.0) / 25.0) + 3.0 + float(lodBias);
}

// Cheap deterministic per-point hash → [0, 1). Used for LOD-threshold dither.
inline float hashUnit(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return float(x) * (1.0 / 4294967296.0);
}
