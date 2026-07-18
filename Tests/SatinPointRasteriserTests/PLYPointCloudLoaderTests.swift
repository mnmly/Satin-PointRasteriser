import Foundation
import simd
import Testing
@testable import SatinPointRasteriser

/// Appends `bytes` little-endian into `data`.
private func appendLE<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
    var le = value.littleEndian
    withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
}

private func appendLE(_ value: Float, to data: inout Data) {
    appendLE(value.bitPattern, to: &data)
}

@Test func asciiPLYLoaderPacksVertexPositionsAndColors() throws {
    let ply = """
    ply
    format ascii 1.0
    element vertex 3
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    0 0 0 255 0 0
    1 0 0 0 255 0
    0 1 0 0 0 255
    """

    let packed = try PLYPointCloudLoader.parse(Data(ply.utf8), pointsPerBatch: 2)
    #expect(packed.pointCount == 3)
    #expect(packed.batchCount == 2)
    let rgb = Set(packed.colors.map { $0 & 0x00ffffff })
    #expect(rgb == [0x0000_00ff, 0x0000_ff00, 0x00ff_0000])
    #expect(packed.levels.count == 3)
}

@Test func asciiPLYLoaderWithoutColorDefaultsToOpaqueWhite() throws {
    let ply = """
    ply
    format ascii 1.0
    element vertex 2
    property float x
    property float y
    property float z
    end_header
    0 0 0
    1 1 1
    """

    let packed = try PLYPointCloudLoader.parse(Data(ply.utf8), pointsPerBatch: 8)
    #expect(packed.pointCount == 2)
    for color in packed.colors {
        #expect(color == 0xffff_ffff)
    }
}

@Test func binaryLittleEndianPLYLoaderParsesFloatVertices() throws {
    var body = Data()
    let points: [SIMD3<Float>] = [
        SIMD3<Float>(0, 0, 0),
        SIMD3<Float>(1, 2, 3),
        SIMD3<Float>(-1, -2, -3),
        SIMD3<Float>(4.5, 5.5, 6.5),
    ]
    for p in points {
        appendLE(p.x, to: &body)
        appendLE(p.y, to: &body)
        appendLE(p.z, to: &body)
    }

    let header = """
    ply
    format binary_little_endian 1.0
    element vertex 4
    property float x
    property float y
    property float z
    end_header

    """
    var data = Data(header.utf8)
    data.append(body)

    let packed = try PLYPointCloudLoader.parse(data, pointsPerBatch: 8, shuffleBatches: false)
    #expect(packed.pointCount == 4)
    let decodedPositions = Set(packed.orderedPositions.map { "\($0.x),\($0.y),\($0.z)" })
    let expectedPositions = Set(points.map { "\($0.x),\($0.y),\($0.z)" })
    #expect(decodedPositions == expectedPositions)
}

@Test func binaryLittleEndianPLYLoaderParsesUCharColors() throws {
    var body = Data()
    // vertex 1: position + red
    appendLE(Float(0), to: &body)
    appendLE(Float(0), to: &body)
    appendLE(Float(0), to: &body)
    body.append(255) // red
    body.append(0) // green
    body.append(0) // blue
    // vertex 2: position + green
    appendLE(Float(1), to: &body)
    appendLE(Float(0), to: &body)
    appendLE(Float(0), to: &body)
    body.append(0)
    body.append(255)
    body.append(0)

    let header = """
    ply
    format binary_little_endian 1.0
    element vertex 2
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header

    """
    var data = Data(header.utf8)
    data.append(body)

    let packed = try PLYPointCloudLoader.parse(data, pointsPerBatch: 8)
    #expect(packed.pointCount == 2)
    let rgb = Set(packed.colors.map { $0 & 0x00ffffff })
    #expect(rgb == [0x0000_00ff, 0x0000_ff00])
}

@Test func colorPropertyAliasesAreRecognized() throws {
    // r/g/b and diffuse_red/diffuse_green/diffuse_blue are aliases for red/green/blue.
    let rgbShort = """
    ply
    format ascii 1.0
    element vertex 1
    property float x
    property float y
    property float z
    property uchar r
    property uchar g
    property uchar b
    end_header
    0 0 0 255 128 64
    """
    let diffuse = """
    ply
    format ascii 1.0
    element vertex 1
    property float x
    property float y
    property float z
    property uchar diffuse_red
    property uchar diffuse_green
    property uchar diffuse_blue
    end_header
    0 0 0 255 128 64
    """

    let packedShort = try PLYPointCloudLoader.parse(Data(rgbShort.utf8), pointsPerBatch: 8)
    let packedDiffuse = try PLYPointCloudLoader.parse(Data(diffuse.utf8), pointsPerBatch: 8)
    #expect(packedShort.colors == packedDiffuse.colors)
    #expect((packedShort.colors[0] & 0x00ffffff) == 0x0040_80ff)
}

@Test func scalarTypesNormalizeColorsPerTypeRange() throws {
    // ushort red = 65535 should normalize to the same 8-bit value as uchar red = 255.
    let ply = """
    ply
    format ascii 1.0
    element vertex 1
    property float x
    property float y
    property float z
    property ushort red
    property ushort green
    property ushort blue
    end_header
    0 0 0 65535 0 0
    """
    let packed = try PLYPointCloudLoader.parse(Data(ply.utf8), pointsPerBatch: 8)
    #expect((packed.colors[0] & 0x00ffffff) == 0x0000_00ff)
}

@Test func loaderThrowsOnListProperty() {
    let ply = """
    ply
    format ascii 1.0
    element vertex 1
    property float x
    property float y
    property float z
    property list uchar int vertex_indices
    end_header
    0 0 0 0
    """
    #expect(throws: PLYPointCloudLoaderError.self) {
        try PLYPointCloudLoader.parse(Data(ply.utf8))
    }
}

@Test func loaderThrowsOnBigEndianFormat() {
    let ply = """
    ply
    format binary_big_endian 1.0
    element vertex 1
    property float x
    property float y
    property float z
    end_header
    """
    #expect(throws: PLYPointCloudLoaderError.self) {
        try PLYPointCloudLoader.parse(Data(ply.utf8))
    }
}

@Test func loaderThrowsOnMissingVertexProperty() {
    let ply = """
    ply
    format ascii 1.0
    element vertex 1
    property float x
    property float y
    end_header
    0 0
    """
    #expect(throws: PLYPointCloudLoaderError.self) {
        try PLYPointCloudLoader.parse(Data(ply.utf8))
    }
}

@Test func loaderThrowsOnTruncatedAsciiBody() {
    let ply = """
    ply
    format ascii 1.0
    element vertex 3
    property float x
    property float y
    property float z
    end_header
    0 0 0
    1 1 1
    """
    #expect(throws: PLYPointCloudLoaderError.self) {
        try PLYPointCloudLoader.parse(Data(ply.utf8))
    }
}

@Test func loaderThrowsOnTruncatedBinaryBody() {
    let header = """
    ply
    format binary_little_endian 1.0
    element vertex 4
    property float x
    property float y
    property float z
    end_header

    """
    var data = Data(header.utf8)
    // Only one vertex worth of bytes instead of four.
    appendLE(Float(0), to: &data)
    appendLE(Float(0), to: &data)
    appendLE(Float(0), to: &data)

    #expect(throws: PLYPointCloudLoaderError.self) {
        try PLYPointCloudLoader.parse(data)
    }
}

@Test func loaderThrowsOnInvalidASCIIValue() {
    let ply = """
    ply
    format ascii 1.0
    element vertex 1
    property float x
    property float y
    property float z
    end_header
    0 notanumber 0
    """
    #expect(throws: PLYPointCloudLoaderError.self) {
        try PLYPointCloudLoader.parse(Data(ply.utf8))
    }
}

@Test func loaderShuffleBatchesParameterDefaultsToTrueAndCanBeDisabled() throws {
    // Enough points across enough batches (>= 6) to make the shuffle observable.
    var body = ""
    var lcg: UInt64 = 0x1234_5678
    func next() -> Float {
        lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
        return Float(lcg >> 40) / Float(1 << 24)
    }
    let count = 1600
    for _ in 0 ..< count {
        body += "\(next() * 10) \(next() * 10) \(next() * 10)\n"
    }
    let header = """
    ply
    format ascii 1.0
    element vertex \(count)
    property float x
    property float y
    property float z
    end_header

    """
    let data = Data((header + body).utf8)

    let shuffled = try PLYPointCloudLoader.parse(data, pointsPerBatch: 64)
    let unshuffled = try PLYPointCloudLoader.parse(data, pointsPerBatch: 64, shuffleBatches: false)
    #expect(unshuffled.batchCount >= 6, "fixture must produce >= 6 batches to exercise the shuffle")

    let shuffledMins = shuffled.batches.map { SIMD3<Float>($0.minX, $0.minY, $0.minZ) }
    let unshuffledMins = unshuffled.batches.map { SIMD3<Float>($0.minX, $0.minY, $0.minZ) }
    #expect(shuffledMins != unshuffledMins, "shuffleBatches should default to true and permute batch order")
    #expect(shuffled.pointCount == unshuffled.pointCount)
}

@Test func loadedPositionsRoundTripThroughPackingWithinQuantizationEpsilon() throws {
    var body = ""
    var lcg: UInt64 = 0x5EED
    func next() -> Float {
        lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
        return Float(lcg >> 40) / Float(1 << 24)
    }
    let count = 500
    for _ in 0 ..< count {
        body += "\(next() * 10 - 5) \(next() * 10 - 5) \(next() * 10 - 5) 128 64 32\n"
    }
    let header = """
    ply
    format ascii 1.0
    element vertex \(count)
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header

    """
    let data = Data((header + body).utf8)

    let packed = try PLYPointCloudLoader.parse(data, pointsPerBatch: 64, shuffleBatches: false)
    #expect(packed.pointCount == count)
    #expect(!packed.orderedPositions.isEmpty)

    for batch in packed.batches {
        let batchMin = SIMD3<Float>(batch.minX, batch.minY, batch.minZ)
        let batchMax = SIMD3<Float>(batch.maxX, batch.maxY, batch.maxZ)
        let batchSize = max(batchMax - batchMin, SIMD3<Float>(repeating: 0.000001))
        let epsilon = simd_length(batchSize) / Float(pointRasteriserSteps30Bit) * 2 + 1e-5

        for localIndex in 0 ..< Int(batch.numPoints) {
            let pointIndex = Int(batch.firstPoint) + localIndex
            let decoded = PackedPointCloudFixtures.decodePosition30Bit(pointIndex: pointIndex, batch: batch, packed: packed)
            let original = packed.orderedPositions[pointIndex]
            let error = simd_length(decoded - original)
            #expect(error <= epsilon, "point \(pointIndex) decoded error \(error) exceeds epsilon \(epsilon)")
        }
    }
}

@Test func loadReadsPLYFileFromDisk() throws {
    let ply = """
    ply
    format ascii 1.0
    element vertex 2
    property float x
    property float y
    property float z
    end_header
    0 0 0
    1 1 1
    """
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".ply")
    try Data(ply.utf8).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }

    let packed = try PLYPointCloudLoader.load(url: url, pointsPerBatch: 8)
    #expect(packed.pointCount == 2)
}
