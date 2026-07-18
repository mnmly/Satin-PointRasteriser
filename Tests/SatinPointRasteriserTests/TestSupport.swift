import Foundation

/// Serializes GPU-touching tests. swift-testing runs tests in parallel by
/// default; spinning up many Metal `Context`s + command queues concurrently on
/// one device intermittently faults the driver (SIGSEGV). GPU tests take this
/// lock for their duration so only one exercises the GPU at a time, while the
/// pure-CPU tests (layout, packer, PLY parsing) keep running in parallel.
nonisolated(unsafe) let gpuTestLock = NSLock()
