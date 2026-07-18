import Foundation

/// Auto-generated Metal preambles prepended to user Displacement / Tint sketch
/// kernels before compile. They expose the same `scr_*` helpers, `RasterBatch`,
/// slot macros, and `SCR_*_KERNEL_BUFFERS` declarations as
/// Satin-ComputeRasteriser, so existing sketches compile unchanged.
///
/// Internal difference (transparent to sketches): `scr_resolve*Thread`
/// binary-searches the batch table by `firstPoint` because this package packs
/// points contiguously rather than in fixed-stride slots.
enum ScrSketchPreamble {
    /// Shared helper body (decode + colour) parameterised by nothing — identical
    /// across both passes.
    private static let decodeHelpers = """
    #define SCR_STEPS_30BIT 1073741824.0f
    #define SCR_STEPS_20BIT 1048576.0f
    #define SCR_STEPS_10BIT 1024.0f
    #define SCR_MASK_10BIT  1023u

    typedef struct {
        int  state;
        float minX, minY, minZ;
        float maxX, maxY, maxZ;
        uint numPoints;
        uint firstPoint;
        uint fileIndex;
        uint scr_padding3, scr_padding4, scr_padding5, scr_padding6, scr_padding7, scr_padding8;
    } RasterBatch;

    inline uint scr_unpack10(uint encoded, uint shift) {
        return (encoded >> shift) & SCR_MASK_10BIT;
    }

    inline float3 scr_decodePointAt(
        uint pointIndex,
        RasterBatch batch,
        device const uint *xyzLow,
        device const uint *xyzMed,
        device const uint *xyzHigh,
        device const uchar *levels)
    {
        const float3 wgMin = float3(batch.minX, batch.minY, batch.minZ);
        const float3 wgSize = max(
            float3(batch.maxX, batch.maxY, batch.maxZ) - wgMin,
            float3(1e-6));
        const int level = int(uint(levels[pointIndex]) & 0x7u);
        if (level == 0) {
            const uint low  = xyzLow[pointIndex];
            const uint med  = xyzMed[pointIndex];
            const uint high = xyzHigh[pointIndex];
            const uint x = (scr_unpack10(low,  0) << 20) | (scr_unpack10(med,  0) << 10) | scr_unpack10(high,  0);
            const uint y = (scr_unpack10(low, 10) << 20) | (scr_unpack10(med, 10) << 10) | scr_unpack10(high, 10);
            const uint z = (scr_unpack10(low, 20) << 20) | (scr_unpack10(med, 20) << 10) | scr_unpack10(high, 20);
            return float3(x, y, z) * (wgSize / SCR_STEPS_30BIT) + wgMin;
        }
        if (level == 1) {
            const uint low = xyzLow[pointIndex];
            const uint med = xyzMed[pointIndex];
            const uint x = (scr_unpack10(low,  0) << 10) | scr_unpack10(med,  0);
            const uint y = (scr_unpack10(low, 10) << 10) | scr_unpack10(med, 10);
            const uint z = (scr_unpack10(low, 20) << 10) | scr_unpack10(med, 20);
            return float3(x, y, z) * (wgSize / SCR_STEPS_20BIT) + wgMin;
        }
        const uint low = xyzLow[pointIndex];
        const uint x = scr_unpack10(low,  0);
        const uint y = scr_unpack10(low, 10);
        const uint z = scr_unpack10(low, 20);
        return float3(x, y, z) * (wgSize / SCR_STEPS_10BIT) + wgMin;
    }

    // Decode a point's native colour (packed RGBA8, little-endian) to 0..1 rgba.
    inline float4 scr_decodeColorAt(uint pointIndex, device const uint *colors) {
        const uint c = colors[pointIndex];
        return float4(float( c        & 0xffu),
                      float((c >>  8) & 0xffu),
                      float((c >> 16) & 0xffu),
                      float((c >> 24) & 0xffu)) * (1.0 / 255.0);
    }
    """

    /// The `scr_resolve*Thread` body: binary-search the batch table by
    /// `firstPoint` (contiguous packing) — id is the pack-order point index.
    private static func resolveThread(fn: String, infoType: String) -> String {
        """
        inline bool \(fn)(
            uint id,
            constant \(infoType) &info,
            device const RasterBatch *batches,
            thread RasterBatch &batch,
            thread uint &pointIndex,
            thread uint &localOffset)
        {
            if (id >= info.scr_totalPoints) return false;
            uint lo = 0u;
            uint hi = info.scr_batchCount; // exclusive
            while (lo + 1u < hi) {
                const uint mid = (lo + hi) >> 1u;
                if (batches[mid].firstPoint <= id) lo = mid; else hi = mid;
            }
            batch = batches[lo];
            if (batch.state == 0) return false;
            localOffset = id - batch.firstPoint;
            if (localOffset >= batch.numPoints) return false; // thread in a partially-filled slot's unwritten tail
            pointIndex = id;
            return true;
        }
        """
    }

    static let displacement: String = """
    // Auto-generated displacement preamble — SatinPointRasteriser.
    #include <metal_stdlib>
    using namespace metal;

    \(decodeHelpers)

    #define SCR_DISP_BUF_BATCHES   0
    #define SCR_DISP_BUF_XYZ_LOW   1
    #define SCR_DISP_BUF_XYZ_MED   2
    #define SCR_DISP_BUF_XYZ_HIGH  3
    #define SCR_DISP_BUF_LEVELS    4
    #define SCR_DISP_BUF_OUT       5
    #define SCR_DISP_BUF_INFO      6
    #define SCR_DISP_BUF_COLORS    7
    #define SCR_DISP_BUF_USER0     8
    #define SCR_DISP_BUF_USER1     9
    #define SCR_DISP_BUF_USER2     10
    #define SCR_DISP_BUF_USER3     11
    #define SCR_DISP_BUF_USER4     12
    #define SCR_DISP_BUF_USER5     13
    #define SCR_DISP_BUF_USER6     14
    #define SCR_DISP_BUF_USER7     15

    typedef struct {
        uint pointsPerBatch;
        uint scr_batchCount;
        uint scr_totalPoints;
        uint scr_reserved;
    } ScrDisplacementInfo;

    \(resolveThread(fn: "scr_resolveDisplacementThread", infoType: "ScrDisplacementInfo"))

    #define SCR_DISPLACEMENT_KERNEL_BUFFERS \\
        device const RasterBatch *batches            [[buffer(SCR_DISP_BUF_BATCHES)]], \\
        device const uint        *xyzLow             [[buffer(SCR_DISP_BUF_XYZ_LOW)]], \\
        device const uint        *xyzMed             [[buffer(SCR_DISP_BUF_XYZ_MED)]], \\
        device const uint        *xyzHigh            [[buffer(SCR_DISP_BUF_XYZ_HIGH)]], \\
        device const uchar       *levels             [[buffer(SCR_DISP_BUF_LEVELS)]], \\
        device       float3      *displacements      [[buffer(SCR_DISP_BUF_OUT)]], \\
        constant     ScrDisplacementInfo &_scrInfo   [[buffer(SCR_DISP_BUF_INFO)]], \\
        device const uint        *colors             [[buffer(SCR_DISP_BUF_COLORS)]]

    """

    static let tint: String = """
    // Auto-generated tint preamble — SatinPointRasteriser.
    #include <metal_stdlib>
    using namespace metal;

    \(decodeHelpers)

    #define SCR_TINT_BUF_BATCHES   0
    #define SCR_TINT_BUF_XYZ_LOW   1
    #define SCR_TINT_BUF_XYZ_MED   2
    #define SCR_TINT_BUF_XYZ_HIGH  3
    #define SCR_TINT_BUF_LEVELS    4
    #define SCR_TINT_BUF_OUT       5
    #define SCR_TINT_BUF_INFO      6
    #define SCR_TINT_BUF_COLORS    7
    #define SCR_TINT_BUF_USER0     8
    #define SCR_TINT_BUF_USER1     9
    #define SCR_TINT_BUF_USER2     10
    #define SCR_TINT_BUF_USER3     11
    #define SCR_TINT_BUF_USER4     12
    #define SCR_TINT_BUF_USER5     13
    #define SCR_TINT_BUF_USER6     14
    #define SCR_TINT_BUF_USER7     15

    typedef struct {
        uint pointsPerBatch;
        uint scr_batchCount;
        uint scr_totalPoints;
        uint scr_reserved;
    } ScrTintInfo;

    \(resolveThread(fn: "scr_resolveTintThread", infoType: "ScrTintInfo"))

    #define SCR_TINT_KERNEL_BUFFERS \\
        device const RasterBatch *batches            [[buffer(SCR_TINT_BUF_BATCHES)]], \\
        device const uint        *xyzLow             [[buffer(SCR_TINT_BUF_XYZ_LOW)]], \\
        device const uint        *xyzMed             [[buffer(SCR_TINT_BUF_XYZ_MED)]], \\
        device const uint        *xyzHigh            [[buffer(SCR_TINT_BUF_XYZ_HIGH)]], \\
        device const uchar       *levels             [[buffer(SCR_TINT_BUF_LEVELS)]], \\
        device       float4      *tints              [[buffer(SCR_TINT_BUF_OUT)]], \\
        constant     ScrTintInfo &_scrInfo           [[buffer(SCR_TINT_BUF_INFO)]], \\
        device const uint        *colors             [[buffer(SCR_TINT_BUF_COLORS)]]

    """
}
