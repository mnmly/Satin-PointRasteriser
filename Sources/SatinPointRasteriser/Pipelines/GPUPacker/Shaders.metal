#include <metal_stdlib>
using namespace metal;

// GPU mirror of PackedPointCloudFixtures.pack() (see PackedPointCloudFixtures.swift).
// Stages: global bounds -> Morton key -> LSD radix sort (4x8-bit) -> gather
// positions -> LOD voxel occupancy -> per-batch AABB -> per-batch stable level
// bucketing -> shuffled finalize (gather + quantize + batch-record emit).
//
// Decode-identical to the CPU path (same Morton order up to equal-key tie
// order, same per-batch quantization, same LOD levels, same stable
// level-ascending point order within each batch, same cumulative level counts
// in PackRasterBatch p3..p6), PLUS the CPU path's batch shuffle: the finalize
// stage relocates whole batches per a CPU-computed permutation so no separate
// shuffle pass is needed. See PackedPointCloudFixtures.shuffleBatches().
//
// Portable Metal: the radix sort and LOD occupancy use only 32-bit atomics.

struct PackParams {
    uint count;
    uint pointsPerBatch;
    uint numBatches;
    uint numTiles;
    uint tileSize;
    uint shift;       // radix pass: bit offset of the 8-bit digit
    uint maxLevel;    // lodLevels - 1
    uint level;       // current LOD level being claimed
    float coarseVoxelDivisions;
    float lodVoxelScale; // 0.5^level
    uint slotBase;    // pool slot index of this chunk's first batch (0 for the
                      // wholesale pack); makes per-batch firstPoint absolute.
};

// RasterBatch — must match PointRasteriserTypes.h byte-for-byte (the renderer
// reads the same MTLBuffer this kernel writes). p3..p6 carry the cumulative LOD
// level counts (8 x uint16, cum[L] = points with level <= L; see
// RasterBatch.lodCumulativeCounts) written by packBucketLOD; p7/p8 = 0.
struct PackRasterBatch {
    int  state;
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    uint numPoints;
    uint firstPoint;
    uint fileIndex;
    uint p3, p4, p5, p6, p7, p8;
};

// ---- float<->orderable-uint for atomic min/max bounds ----------------------
inline uint floatFlip(float f) {
    uint x = as_type<uint>(f);
    uint mask = (uint)(-(int)(x >> 31)) | 0x80000000u;
    return x ^ mask;
}
inline float floatUnflip(uint u) {
    uint mask = ((u >> 31) - 1u) | 0x80000000u;
    return as_type<float>(u ^ mask);
}

inline uint spread10(uint v) {
    uint x = v & 0x3ffu;
    x = (x | (x << 16)) & 0x030000ffu;
    x = (x | (x << 8))  & 0x0300f00fu;
    x = (x | (x << 4))  & 0x030c30c3u;
    x = (x | (x << 2))  & 0x09249249u;
    return x;
}

inline float3 boundsMin(device const uint *b) { return float3(floatUnflip(b[0]), floatUnflip(b[1]), floatUnflip(b[2])); }
inline float3 boundsMax(device const uint *b) { return float3(floatUnflip(b[3]), floatUnflip(b[4]), floatUnflip(b[5])); }

// ============================ bounds ========================================

kernel void packBoundsInit(device atomic_uint *bounds [[buffer(0)]],
                           uint tid [[thread_position_in_grid]]) {
    if (tid < 3) atomic_store_explicit(&bounds[tid], 0xffffffffu, memory_order_relaxed);
    else if (tid < 6) atomic_store_explicit(&bounds[tid], 0u, memory_order_relaxed);
}

kernel void packBounds(device const float3 *positions [[buffer(0)]],
                       device atomic_uint *bounds [[buffer(1)]],
                       constant PackParams &p [[buffer(2)]],
                       uint tid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint gridSize [[threads_per_grid]],
                       uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float3 sMin[256];
    threadgroup float3 sMax[256];
    float3 mn = float3(FLT_MAX);
    float3 mx = float3(-FLT_MAX);
    for (uint i = tid; i < p.count; i += gridSize) {
        float3 q = positions[i];
        mn = min(mn, q);
        mx = max(mx, q);
    }
    sMin[lid] = mn;
    sMax[lid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sMin[lid] = min(sMin[lid], sMin[lid + s]);
            sMax[lid] = max(sMax[lid], sMax[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        atomic_fetch_min_explicit(&bounds[0], floatFlip(sMin[0].x), memory_order_relaxed);
        atomic_fetch_min_explicit(&bounds[1], floatFlip(sMin[0].y), memory_order_relaxed);
        atomic_fetch_min_explicit(&bounds[2], floatFlip(sMin[0].z), memory_order_relaxed);
        atomic_fetch_max_explicit(&bounds[3], floatFlip(sMax[0].x), memory_order_relaxed);
        atomic_fetch_max_explicit(&bounds[4], floatFlip(sMax[0].y), memory_order_relaxed);
        atomic_fetch_max_explicit(&bounds[5], floatFlip(sMax[0].z), memory_order_relaxed);
    }
}

// ============================ morton ========================================

kernel void packMorton(device const float3 *positions [[buffer(0)]],
                       device uint *keys [[buffer(1)]],
                       device uint *indices [[buffer(2)]],
                       device const uint *bounds [[buffer(3)]],
                       constant PackParams &p [[buffer(4)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    float3 bmin = boundsMin(bounds);
    float3 bmax = boundsMax(bounds);
    float3 extent = max(bmax - bmin, float3(0.000001));
    float3 scale = float3(1023.0) / extent;
    float3 n = clamp((positions[i] - bmin) * scale, float3(0.0), float3(1023.0));
    uint kx = spread10(uint(n.x));
    uint ky = spread10(uint(n.y));
    uint kz = spread10(uint(n.z));
    keys[i] = (kx << 2) | (ky << 1) | kz;
    indices[i] = i;
}

// ============================ radix sort ====================================
// digit-major histogram + per-digit cross-tile exclusive scan + stable
// serial-per-tile scatter. One thread per tile keeps histogram and scatter in
// identical iteration order, guaranteeing a stable partition. Because the CPU
// reference (PackedPointCloudFixtures.mortonOrder) tie-breaks equal Morton keys
// by ascending original index, this stable LSD sort — which preserves the
// identity input order across equal keys — reproduces the CPU order exactly.

kernel void radixHistogram(device const uint *keysIn [[buffer(0)]],
                           device uint *hist [[buffer(1)]],
                           constant PackParams &p [[buffer(2)]],
                           uint tile [[thread_position_in_grid]]) {
    if (tile >= p.numTiles) return;
    uint local[256];
    for (uint d = 0; d < 256; d++) local[d] = 0;
    uint start = tile * p.tileSize;
    uint end = min(start + p.tileSize, p.count);
    for (uint i = start; i < end; i++) {
        local[(keysIn[i] >> p.shift) & 0xffu]++;
    }
    for (uint d = 0; d < 256; d++) hist[d * p.numTiles + tile] = local[d];
}

// One threadgroup per digit. Exclusive-scan hist[d*numTiles + (0..numTiles)]
// cooperatively, writing per-tile global base offset into tileOffset and the
// digit total into digitTotal.
kernel void radixScanPerDigit(device const uint *hist [[buffer(0)]],
                              device uint *tileOffset [[buffer(1)]],
                              device uint *digitTotal [[buffer(2)]],
                              constant PackParams &p [[buffer(3)]],
                              uint d [[threadgroup_position_in_grid]],
                              uint lid [[thread_position_in_threadgroup]],
                              uint tgsize [[threads_per_threadgroup]]) {
    threadgroup uint temp[256];
    threadgroup uint carry;
    if (lid == 0) carry = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = d * p.numTiles;
    for (uint chunk = 0; chunk < p.numTiles; chunk += tgsize) {
        uint t = chunk + lid;
        uint v = (t < p.numTiles) ? hist[base + t] : 0u;
        temp[lid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint o = 1; o < tgsize; o <<= 1) {
            uint add = (lid >= o) ? temp[lid - o] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            temp[lid] += add;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint incl = temp[lid];
        if (t < p.numTiles) tileOffset[base + t] = carry + (incl - v);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == tgsize - 1) carry += incl;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) digitTotal[d] = carry;
}

kernel void radixDigitBase(device const uint *digitTotal [[buffer(0)]],
                           device uint *digitBase [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint acc = 0;
    for (uint d = 0; d < 256; d++) {
        digitBase[d] = acc;
        acc += digitTotal[d];
    }
}

kernel void radixScatter(device const uint *keysIn [[buffer(0)]],
                         device const uint *indicesIn [[buffer(1)]],
                         device uint *keysOut [[buffer(2)]],
                         device uint *indicesOut [[buffer(3)]],
                         device const uint *tileOffset [[buffer(4)]],
                         device const uint *digitBase [[buffer(5)]],
                         constant PackParams &p [[buffer(6)]],
                         uint tile [[thread_position_in_grid]]) {
    if (tile >= p.numTiles) return;
    uint cursor[256];
    for (uint d = 0; d < 256; d++) cursor[d] = digitBase[d] + tileOffset[d * p.numTiles + tile];
    uint start = tile * p.tileSize;
    uint end = min(start + p.tileSize, p.count);
    for (uint i = start; i < end; i++) {
        uint k = keysIn[i];
        uint dig = (k >> p.shift) & 0xffu;
        uint pos = cursor[dig]++;
        keysOut[pos] = k;
        indicesOut[pos] = indicesIn[i];
    }
}

// ============================ gather (positions) ============================
// Reorder positions into Morton order for the LOD and per-batch AABB stages.
// Colors/quantization are deferred to packShuffleFinalize, which gathers
// straight from the source buffers via the bucketed permutation.

kernel void packGatherPos(device const float3 *positions [[buffer(0)]],
                          device const uint *indices [[buffer(1)]],
                          device float3 *sortedPos [[buffer(2)]],
                          constant PackParams &p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    sortedPos[i] = positions[indices[i]];
}

// ============================ batch AABB ====================================
// One threadgroup per batch. Writes the per-batch AABB + numPoints into a
// scratch PackRasterBatch array (Morton/old order). packBucketLOD patches
// p3..p6; packShuffleFinalize reads these and emits the final (shuffled) batch
// record. Runs on the Morton-ordered positions — the AABB is
// permutation-invariant within the batch slice.

kernel void packBatchAABB(device const float3 *sortedPos [[buffer(0)]],
                          device PackRasterBatch *batches [[buffer(1)]],
                          constant PackParams &p [[buffer(2)]],
                          uint b [[threadgroup_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float3 sMin[256];
    threadgroup float3 sMax[256];
    uint start = b * p.pointsPerBatch;
    uint end = min(start + p.pointsPerBatch, p.count);
    float3 mn = float3(FLT_MAX);
    float3 mx = float3(-FLT_MAX);
    for (uint i = start + lid; i < end; i += tgsize) {
        float3 q = sortedPos[i];
        mn = min(mn, q);
        mx = max(mx, q);
    }
    sMin[lid] = mn;
    sMax[lid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sMin[lid] = min(sMin[lid], sMin[lid + s]);
            sMax[lid] = max(sMax[lid], sMax[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        PackRasterBatch rb;
        rb.state = 1;
        rb.minX = sMin[0].x; rb.minY = sMin[0].y; rb.minZ = sMin[0].z;
        rb.maxX = sMax[0].x; rb.maxY = sMax[0].y; rb.maxZ = sMax[0].z;
        rb.numPoints = end - start;
        rb.firstPoint = (p.slotBase + b) * p.pointsPerBatch;
        rb.fileIndex = 0;
        rb.p3 = rb.p4 = rb.p5 = rb.p6 = rb.p7 = rb.p8 = 0;
        batches[b] = rb;
    }
}

// ============================ LOD ===========================================
// Dense voxel occupancy grid. atomic_min(grid[cell], sortedIndex) makes the
// lowest sorted index in each voxel win — exactly the CPU "first in sorted
// order to occupy the voxel" rule. Grid dims are derived from global bounds.

inline void lodCellDims(device const uint *bounds, constant PackParams &p,
                        thread float3 &bmin, thread float &inv,
                        thread uint &dx, thread uint &dy) {
    bmin = boundsMin(bounds);
    float3 bmax = boundsMax(bounds);
    float3 extent = max(bmax - bmin, float3(0.000001));
    float longest = max(extent.x, max(extent.y, extent.z));
    float voxel = (longest / p.coarseVoxelDivisions) * p.lodVoxelScale;
    inv = 1.0 / max(voxel, 0.000001);
    dx = uint(floor(extent.x * inv)) + 1u;
    dy = uint(floor(extent.y * inv)) + 1u;
}

kernel void packLODInit(device uchar *levels [[buffer(0)]],
                        constant PackParams &p [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    levels[i] = uchar(p.maxLevel);
}

kernel void packLODClaim(device const float3 *sortedPos [[buffer(0)]],
                         device const uchar *levels [[buffer(1)]],
                         device atomic_uint *grid [[buffer(2)]],
                         device const uint *bounds [[buffer(3)]],
                         constant PackParams &p [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    if (uint(levels[i]) != p.maxLevel) return;
    float3 bmin; float inv; uint dx, dy;
    lodCellDims(bounds, p, bmin, inv, dx, dy);
    // local is bit-identical to the CPU's (pos-boundsMin)*invVoxel, so floor()
    // matches the CPU cell exactly. No upper clamp — a clamp would merge a
    // boundary point into a different cell than the CPU's raw floor. cx<=dx-1
    // and cy<=dy-1 hold because local<=extent*inv and dx=floor(extent*inv)+1.
    float3 local = (sortedPos[i] - bmin) * inv;
    uint cx = uint(floor(local.x));
    uint cy = uint(floor(local.y));
    uint cz = uint(floor(local.z));
    uint lin = cx + dx * (cy + dy * cz);
    atomic_fetch_min_explicit(&grid[lin], i, memory_order_relaxed);
}

kernel void packLODAssign(device const float3 *sortedPos [[buffer(0)]],
                          device uchar *levels [[buffer(1)]],
                          device const uint *grid [[buffer(2)]],
                          device const uint *bounds [[buffer(3)]],
                          constant PackParams &p [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    if (uint(levels[i]) != p.maxLevel) return;
    float3 bmin; float inv; uint dx, dy;
    lodCellDims(bounds, p, bmin, inv, dx, dy);
    float3 local = (sortedPos[i] - bmin) * inv;
    uint cx = uint(floor(local.x));
    uint cy = uint(floor(local.y));
    uint cz = uint(floor(local.z));
    uint lin = cx + dx * (cy + dy * cz);
    if (grid[lin] == i) levels[i] = uchar(p.level);
}

// ============================ LOD bucketing =================================
// One thread per batch: stable 8-bucket counting scatter by LOD level over the
// batch's slice, mirroring the CPU bucketSortBatchSlicesByLevel. Serial
// iteration in Morton order guarantees stability within each level. Emits the
// bucketed source-index permutation (consumed by packShuffleFinalize), the
// final-order per-point levels (Morton/old positions), and patches the
// cumulative level counts into the scratch PackRasterBatch.

kernel void packBucketLOD(device const uchar *mortonLevels [[buffer(0)]],
                          device const uint *indicesIn [[buffer(1)]],
                          device uint *indicesOut [[buffer(2)]],
                          device uchar *levelsOut [[buffer(3)]],
                          device PackRasterBatch *batches [[buffer(4)]],
                          constant PackParams &p [[buffer(5)]],
                          uint b [[thread_position_in_grid]]) {
    if (b >= p.numBatches) return;
    uint start = b * p.pointsPerBatch;
    uint end = min(start + p.pointsPerBatch, p.count);
    uint cursor[8];
    for (uint l = 0; l < 8; l++) cursor[l] = 0;
    for (uint i = start; i < end; i++) cursor[mortonLevels[i] & 7u]++;
    uint acc = 0;
    for (uint l = 0; l < 8; l++) {
        uint c = cursor[l];
        cursor[l] = acc;
        acc += c;
    }
    for (uint i = start; i < end; i++) {
        uchar lvl = mortonLevels[i];
        uint dst = start + cursor[lvl & 7u]++;
        indicesOut[dst] = indicesIn[i];
        levelsOut[dst] = lvl;
    }
    // After the scatter cursor[L] == points with level <= L (cum[7] == numPoints).
    batches[b].p3 = cursor[0] | (cursor[1] << 16);
    batches[b].p4 = cursor[2] | (cursor[3] << 16);
    batches[b].p5 = cursor[4] | (cursor[5] << 16);
    batches[b].p6 = cursor[6] | (cursor[7] << 16);
}

// ============================ shuffled finalize =============================
// One threadgroup per FINAL (shuffled) batch j. shuffleMap[j] = (oldBatch,
// newFirstPoint): the finalize gathers old batch `oldBatch`'s bucketed points
// straight from the source position/color buffers, quantizes them against that
// batch's AABB, and writes them contiguously at newFirstPoint — folding the
// CPU path's whole-batch shuffle into the single output pass. Emits the final
// batch record (old metadata with firstPoint rebased to newFirstPoint).

kernel void packShuffleFinalize(device const float3 *positions [[buffer(0)]],
                                device const float4 *colorsIn [[buffer(1)]],
                                device const uint *bucketedIndices [[buffer(2)]],
                                device const uchar *bucketedLevels [[buffer(3)]],
                                device const PackRasterBatch *oldBatches [[buffer(4)]],
                                device const uint2 *shuffleMap [[buffer(5)]],
                                device uint *xyzLow [[buffer(6)]],
                                device uint *xyzMed [[buffer(7)]],
                                device uint *xyzHigh [[buffer(8)]],
                                device uint *colorsOut [[buffer(9)]],
                                device uchar *levelsOut [[buffer(10)]],
                                device PackRasterBatch *batchesOut [[buffer(11)]],
                                constant PackParams &p [[buffer(12)]],
                                uint j [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint tgsize [[threads_per_threadgroup]]) {
    if (j >= p.numBatches) return;
    uint2 m = shuffleMap[j];
    uint oldB = m.x;
    uint newFirst = m.y;
    PackRasterBatch rb = oldBatches[oldB];
    uint oldStart = oldB * p.pointsPerBatch;
    uint cnt = rb.numPoints;
    float3 bmin = float3(rb.minX, rb.minY, rb.minZ);
    float3 size = max(float3(rb.maxX, rb.maxY, rb.maxZ) - bmin, float3(0.000001));

    for (uint k = lid; k < cnt; k += tgsize) {
        uint src = bucketedIndices[oldStart + k];
        uint dst = newFirst + k;

        float3 q = positions[src];
        float3 n = clamp((q - bmin) / size, float3(0.0), float3(0.99999994));
        // Match CPU EXACTLY: normalized * Float(pointRasteriserSteps30Bit - 1),
        // where Float(1073741823) rounds to 2^30 in f32. float3(...) forces a
        // float32 multiply (a bare literal can promote to double and diverge).
        uint3 qu = uint3(n * float3(1073741823.0));
        uint xl = (qu.x >> 20) & 1023u, yl = (qu.y >> 20) & 1023u, zl = (qu.z >> 20) & 1023u;
        uint xm = (qu.x >> 10) & 1023u, ym = (qu.y >> 10) & 1023u, zm = (qu.z >> 10) & 1023u;
        uint xh = qu.x & 1023u, yh = qu.y & 1023u, zh = qu.z & 1023u;
        xyzLow[dst]  = xl | (yl << 10) | (zl << 20);
        xyzMed[dst]  = xm | (ym << 10) | (zm << 20);
        xyzHigh[dst] = xh | (yh << 10) | (zh << 20);

        float4 c = clamp(colorsIn[src], 0.0, 1.0);
        uint r = uint(c.x * 255.0);
        uint g = uint(c.y * 255.0);
        uint bl = uint(c.z * 255.0);
        uint a = uint(c.w * 255.0);
        colorsOut[dst] = r | (g << 8) | (bl << 16) | (a << 24);

        levelsOut[dst] = bucketedLevels[oldStart + k];
    }

    if (lid == 0) {
        rb.firstPoint = newFirst;
        rb.state = 1;
        batchesOut[j] = rb;
    }
}
