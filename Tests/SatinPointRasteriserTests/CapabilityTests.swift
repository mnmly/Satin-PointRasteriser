import Foundation
import Metal
import Testing
@testable import SatinPointRasteriser

// This machine (M5 Max) is Apple9-family, so these assertions are pinned to
// `true`. On a GPU-less CI host `MTLCreateSystemDefaultDevice()` returns nil
// and the test returns early rather than failing the run.
@Test func capabilitiesReportApple9AndPassTheEmpirical64BitAtomicProbe() {
    guard let device = MTLCreateSystemDefaultDevice() else {
        return
    }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }

    let capabilities = RasteriserCapabilities(device: device)
    #expect(capabilities.supportsApple9 == true, "expected an Apple9+ (M5-class) GPU on this machine")
    #expect(capabilities.use64BitAtomics == true, "use64BitAtomics should default to supportsApple9")
    #expect(!capabilities.gpuFamilyDescription.isEmpty)

    #expect(RasteriserCapabilities.verify64BitAtomics(device: device) == true)
}

@Test func capabilitiesUse64BitAtomicsOverrideIsHonored() {
    guard let device = MTLCreateSystemDefaultDevice() else {
        return
    }
    gpuTestLock.lock(); defer { gpuTestLock.unlock() }
    let forcedOff = RasteriserCapabilities(device: device, use64BitAtomics: false)
    #expect(forcedOff.use64BitAtomics == false)
    #expect(forcedOff.supportsApple9 == device.supportsFamily(.apple9))
}
