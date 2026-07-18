// Single-thread finalize: read the LOD survivor count the LODSelect pass
// appended, clamp it to the cloud's LOD capacity, and write the indirect
// dispatch args (ceil(count / PR_THREADS_PER_GROUP) threadgroups) for the
// point-count-driven depth/color passes.
#include "../Common.metal"

kernel void loddispatchFinalizeUpdate(
    device atomic_uint *lodStats [[buffer(ComputeBufferCustom0)]],
    device CRDispatchArgs *args [[buffer(ComputeBufferCustom1)]],
    constant uint &lodCapacity [[buffer(ComputeBufferCustom2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }
    const uint raw = atomic_load_explicit(&lodStats[0], memory_order_relaxed);
    // On overflow the append cursor runs past capacity; clamp so the depth/color
    // passes never read out of bounds. lodStats[2] is the count both passes read.
    const uint clamped = min(raw, lodCapacity);
    atomic_store_explicit(&lodStats[2], clamped, memory_order_relaxed);
    args[0].threadgroupsX = (clamped + PR_THREADS_PER_GROUP - 1u) / PR_THREADS_PER_GROUP;
    args[0].threadgroupsY = 1u;
    args[0].threadgroupsZ = 1u;
}
