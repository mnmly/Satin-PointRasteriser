#include <metal_stdlib>
using namespace metal;

// 64-bit buffer atomic min/max (`atomic_max_explicit`/`atomic_min_explicit` on
// `atomic_ulong`) require MSL 3.1 and are only enabled by the compiler when
// `__HAVE_ATOMIC_ULONG_MIN_MAX__` is defined (Apple9+ GPUs, e.g. M5-class).
// Note this is the `atomic_max_explicit`/`atomic_min_explicit` overload set,
// NOT `atomic_fetch_max_explicit` — the fetch-and-return-old-value family
// only supports 32-bit int/uint in this Metal version, even under MSL 3.1.
// Gated so this file still compiles on toolchains/devices without the
// extension; on hardware that can't run it, `RasteriserCapabilities.verify64BitAtomics`
// simply reports `false` without dispatching.
#if defined(__HAVE_ATOMIC_ULONG_MIN_MAX__)

// Applies `atomic_max_explicit` to `result` for each of `count` input values —
// the minimal probe RasteriserCapabilities uses to confirm 64-bit buffer
// atomics actually work on the current device/driver, not just that the
// shader compiles.
kernel void probeAtomicULongMax(
    device atomic_ulong *result [[buffer(0)]],
    constant ulong *inputs [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    atomic_max_explicit(result, inputs[gid], memory_order_relaxed);
}

#else

// Fallback stub so the library still compiles on toolchains/devices without
// 64-bit buffer atomic support; leaves `result` untouched so the caller's
// max-of-inputs check fails and `verify64BitAtomics` reports `false`.
kernel void probeAtomicULongMax(
    device uint *result [[buffer(0)]],
    constant uint *inputs [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    (void)result;
    (void)inputs;
    (void)count;
    (void)gid;
}

#endif
