#ifndef PointRasteriserTypes_h
#define PointRasteriserTypes_h

#define PR_THREADS_PER_GROUP 128
#define PR_STEPS_30BIT 1073741824.0f
#define PR_STEPS_20BIT 1048576.0f
#define PR_STEPS_10BIT 1024.0f
#define PR_MASK_10BIT 1023u
#define PR_MAX_DEPTH 0xffffffffu
// Radius of the point-rejection neighborhood in the merged reject+resolve pass:
// a (2*R+1)² window, i.e. 7×7 at R=3. The Magnopus write-up found larger grids
// counterproductive (cost grows quadratically while the occlusion decision is
// already settled by the near neighbors), so keep this at 3.
#define PR_REJECT_RADIUS 3

typedef struct {
    int state;
    float minX;
    float minY;
    float minZ;
    float maxX;
    float maxY;
    float maxZ;
    uint numPoints;
    uint firstPoint;
    uint fileIndex;
    // Cumulative LOD level counts, two uint16 per word (low level in the low
    // half): lodCum01 = cum0 | cum1<<16 … lodCum67 = cum6 | cum7<<16, where
    // cum[L] = points in the batch with level <= L. cum7 == numPoints for a
    // bucketed batch, so lodCum67 == 0 is the legacy sentinel (draw the full
    // numPoints range). Mirrors RasterBatch.padding3..6 on the Swift side.
    uint lodCum01;
    uint lodCum23;
    uint lodCum45;
    uint lodCum67;
    uint padding7;
    uint padding8;
} RasterBatch;

typedef struct {
    float4x4 transform;
    float4x4 transformFrustum;
    float4x4 world;
    float4x4 prevTransform;
} RasterFile;

typedef struct {
    uint depth;
    uint red;
    uint green;
    uint blue;
    uint count;
    uint weight;   // Σ(coverage·255) for translucent-defocus accumulation (else 0)
    uint2 padding;
} RasterPixel;

typedef struct {
    uint batchIndex;
    int level;
    float lodThreshold;
    uint activePoints;   // LOD survivor prefix length computed by cullUpdate
} VisibleBatch;

typedef struct {
    uint threadgroupsX;
    uint threadgroupsY;
    uint threadgroupsZ;
} CRDispatchArgs;

// Compacted LOD point cloud entry (SoA, see LODCloudLayout on the Swift
// side): one CLOD-surviving point emitted by the (future) LODSelect pass.
// Stored as three parallel buffers rather than one AoS struct so the depth
// pass can read positions without touching colors/indices:
//   device packed_float3 *lodPositions   (12 B stride, object-space)
//   device uint          *lodColors      (4 B stride, packed RGBA8)
//   device uint          *lodSourceIndices (4 B stride, pack-order index)
// No typedef needed here — kernels declare the three buffers directly.

#endif
