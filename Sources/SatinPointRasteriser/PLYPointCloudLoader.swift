import Foundation
import simd

/// Errors thrown while parsing a PLY point-cloud file.
public enum PLYPointCloudLoaderError: LocalizedError {
    /// The file does not start with a recognizable PLY header, or `end_header` was never found.
    case invalidHeader
    /// The header declared a `format` this loader does not support (only `ascii 1.0` and `binary_little_endian 1.0` are supported).
    case unsupportedFormat(String)
    /// The header never declared a `vertex` element.
    case missingVertexElement
    /// A required vertex property (e.g. `x`, `y`, or `z`) is missing.
    case missingProperty(String)
    /// A vertex property uses an unsupported type, such as a `list` property.
    case unsupportedProperty(String)
    /// The file ended before all declared vertex data could be read.
    case truncatedBody
    /// An ASCII vertex line contained a value that could not be parsed as the declared property type.
    case invalidASCIIValue(line: Int, property: String)

    public var errorDescription: String? {
        switch self {
        case .invalidHeader:
            return "The file is not a valid PLY file."
        case let .unsupportedFormat(format):
            return "Unsupported PLY format: \(format)."
        case .missingVertexElement:
            return "The PLY file does not contain a vertex element."
        case let .missingProperty(property):
            return "The PLY file is missing required property '\(property)'."
        case let .unsupportedProperty(property):
            return "Unsupported PLY vertex property '\(property)'."
        case .truncatedBody:
            return "The PLY file ended before all vertex data could be read."
        case let .invalidASCIIValue(line, property):
            return "Could not parse property '\(property)' on vertex line \(line)."
        }
    }
}

/// Loads point clouds from PLY files (`ascii 1.0` and `binary_little_endian 1.0`) into a ``PackedPointCloud``.
///
/// Requires `x`/`y`/`z` vertex properties; optional `red`/`green`/`blue` (or `r`/`g`/`b`,
/// `diffuse_red`/`diffuse_green`/`diffuse_blue`) properties are read and normalized to `[0, 1]`
/// per their declared scalar type. `list` properties and big-endian formats are not supported.
public enum PLYPointCloudLoader {
    /// Loads and packs a PLY point cloud from a file at `url`.
    ///
    /// - Parameters:
    ///   - url: File location of the `.ply` file.
    ///   - pointsPerBatch: Target number of points per packed batch. Defaults to
    ///     ``pointRasteriserThreadsPerGroup`` × 80.
    ///   - shuffleBatches: Forwarded to ``PackedPointCloudFixtures/pack(positions:colors:pointsPerBatch:lodLevels:coarseVoxelDivisions:shuffleBatches:)``;
    ///     `true` (the default) applies the inter-wave-contention batch shuffle.
    /// - Returns: The parsed points, packed into a ``PackedPointCloud``.
    public static func load(
        url: URL,
        pointsPerBatch: Int = pointRasteriserThreadsPerGroup * 80,
        shuffleBatches: Bool = true
    ) throws -> PackedPointCloud {
        try parse(Data(contentsOf: url, options: [.mappedIfSafe]), pointsPerBatch: pointsPerBatch, shuffleBatches: shuffleBatches)
    }

    /// Parses and packs a PLY point cloud from in-memory `data`.
    ///
    /// - Parameters:
    ///   - data: Raw bytes of a `.ply` file.
    ///   - pointsPerBatch: Target number of points per packed batch. Defaults to
    ///     ``pointRasteriserThreadsPerGroup`` × 80.
    ///   - shuffleBatches: Forwarded to ``PackedPointCloudFixtures/pack(positions:colors:pointsPerBatch:lodLevels:coarseVoxelDivisions:shuffleBatches:)``;
    ///     `true` (the default) applies the inter-wave-contention batch shuffle.
    /// - Returns: The parsed points, packed into a ``PackedPointCloud``.
    public static func parse(
        _ data: Data,
        pointsPerBatch: Int = pointRasteriserThreadsPerGroup * 80,
        shuffleBatches: Bool = true
    ) throws -> PackedPointCloud {
        let (positions, colors) = try parseArrays(data)
        return PackedPointCloudFixtures.pack(positions: positions, colors: colors, pointsPerBatch: pointsPerBatch, shuffleBatches: shuffleBatches)
    }

    /// Parses a PLY into contiguous position/color arrays **without packing** —
    /// the fast path's front half, shared by the CPU pack (``parse(_:pointsPerBatch:shuffleBatches:)``)
    /// and the GPU pack. Colors are normalized to `[0, 1]`.
    public static func parseArrays(_ data: Data) throws -> (positions: [SIMD3<Float>], colors: [SIMD4<Float>]) {
        try data.withUnsafeBytes { rawBuffer in
            guard let base = rawBuffer.bindMemory(to: UInt8.self).baseAddress else {
                throw PLYPointCloudLoaderError.invalidHeader
            }
            let header = try parseHeader(base: base, count: rawBuffer.count)
            let layout = try PLYVertexLayout(properties: header.vertexProperties)

            switch header.format {
            case .ascii:
                return try parseASCII(base: base, count: rawBuffer.count, header: header, layout: layout)
            case .binaryLittleEndian:
                return try parseBinaryLittleEndian(base: base, count: rawBuffer.count, header: header, layout: layout)
            }
        }
    }

    /// Loads a PLY into contiguous position/color arrays without packing.
    public static func loadArrays(url: URL) throws -> (positions: [SIMD3<Float>], colors: [SIMD4<Float>]) {
        try parseArrays(Data(contentsOf: url, options: [.mappedIfSafe]))
    }
}

private enum PLYFormat {
    case ascii
    case binaryLittleEndian
}

private enum PLYScalarType: String {
    case int8
    case uint8
    case int16
    case uint16
    case int32
    case uint32
    case float32
    case float64

    init?(_ raw: String) {
        switch raw {
        case "char", "int8": self = .int8
        case "uchar", "uint8": self = .uint8
        case "short", "int16": self = .int16
        case "ushort", "uint16": self = .uint16
        case "int", "int32": self = .int32
        case "uint", "uint32": self = .uint32
        case "float", "float32": self = .float32
        case "double", "float64": self = .float64
        default: return nil
        }
    }

    var byteCount: Int {
        switch self {
        case .int8, .uint8: 1
        case .int16, .uint16: 2
        case .int32, .uint32, .float32: 4
        case .float64: 8
        }
    }

    var colorScale: Float {
        switch self {
        case .int8: 127.0
        case .uint8: 255.0
        case .int16: 32767.0
        case .uint16: 65535.0
        case .int32: 2_147_483_647.0
        case .uint32: 4_294_967_295.0
        case .float32, .float64: 1.0
        }
    }
}

private struct PLYProperty {
    var name: String
    var type: PLYScalarType
}

private struct PLYHeader {
    var format: PLYFormat
    var vertexCount: Int
    var vertexProperties: [PLYProperty]
    var bodyOffset: Int
}

private enum PLYSemantic {
    case x
    case y
    case z
    case red
    case green
    case blue
    case ignored
}

private struct PLYField {
    var semantic: PLYSemantic
    var name: String
    var type: PLYScalarType
    var offset: Int
}

private struct PLYVertexLayout {
    var fields: [PLYField]
    var stride: Int

    init(properties: [PLYProperty]) throws {
        var offset = 0
        var fields: [PLYField] = []
        fields.reserveCapacity(properties.count)

        for property in properties {
            fields.append(PLYField(semantic: semantic(for: property.name), name: property.name, type: property.type, offset: offset))
            offset += property.type.byteCount
        }

        guard fields.contains(where: { $0.semantic == .x }),
              fields.contains(where: { $0.semantic == .y }),
              fields.contains(where: { $0.semantic == .z }) else {
            throw PLYPointCloudLoaderError.missingProperty("x, y, z")
        }

        self.fields = fields
        stride = offset
    }
}

private func semantic(for name: String) -> PLYSemantic {
    switch name {
    case "x": return .x
    case "y": return .y
    case "z": return .z
    case "red", "r", "diffuse_red": return .red
    case "green", "g", "diffuse_green": return .green
    case "blue", "b", "diffuse_blue": return .blue
    default: return .ignored
    }
}

private func parseHeader(base: UnsafePointer<UInt8>, count: Int) throws -> PLYHeader {
    guard count >= 4 else { throw PLYPointCloudLoaderError.invalidHeader }
    let marker = Array("end_header".utf8)
    var markerEnd: Int?

    if count >= marker.count {
        for index in 0 ... (count - marker.count) {
            var matches = true
            for offset in 0 ..< marker.count where base[index + offset] != marker[offset] {
                matches = false
                break
            }
            if matches {
                var end = index + marker.count
                if end < count, base[end] == 13 { end += 1 }
                if end < count, base[end] == 10 { end += 1 }
                markerEnd = end
                break
            }
        }
    }

    guard let bodyOffset = markerEnd,
          let headerText = String(bytes: UnsafeBufferPointer(start: base, count: bodyOffset), encoding: .utf8) else {
        throw PLYPointCloudLoaderError.invalidHeader
    }

    var format: PLYFormat?
    var vertexCount: Int?
    var inVertexElement = false
    var vertexProperties: [PLYProperty] = []

    for line in headerText.split(whereSeparator: \.isNewline).map(String.init) {
        let parts = line.split(separator: " ").map(String.init)
        guard !parts.isEmpty else { continue }

        switch parts[0] {
        case "ply":
            continue
        case "format":
            guard parts.count >= 3 else { throw PLYPointCloudLoaderError.invalidHeader }
            switch parts[1] {
            case "ascii": format = .ascii
            case "binary_little_endian": format = .binaryLittleEndian
            default: throw PLYPointCloudLoaderError.unsupportedFormat(parts[1])
            }
        case "element":
            guard parts.count >= 3 else { throw PLYPointCloudLoaderError.invalidHeader }
            inVertexElement = parts[1] == "vertex"
            if inVertexElement {
                guard let count = Int(parts[2]) else { throw PLYPointCloudLoaderError.invalidHeader }
                vertexCount = count
            }
        case "property" where inVertexElement:
            guard parts.count >= 3 else { throw PLYPointCloudLoaderError.invalidHeader }
            if parts[1] == "list" {
                throw PLYPointCloudLoaderError.unsupportedProperty("list \(parts.dropFirst(2).joined(separator: " "))")
            }
            guard let type = PLYScalarType(parts[1]) else {
                throw PLYPointCloudLoaderError.unsupportedProperty(parts[1])
            }
            vertexProperties.append(PLYProperty(name: parts[2], type: type))
        default:
            continue
        }
    }

    guard let format else { throw PLYPointCloudLoaderError.invalidHeader }
    guard let vertexCount else { throw PLYPointCloudLoaderError.missingVertexElement }
    return PLYHeader(format: format, vertexCount: vertexCount, vertexProperties: vertexProperties, bodyOffset: bodyOffset)
}

private func parseASCII(
    base: UnsafePointer<UInt8>,
    count: Int,
    header: PLYHeader,
    layout: PLYVertexLayout
) throws -> (positions: [SIMD3<Float>], colors: [SIMD4<Float>]) {
    guard let body = String(bytes: UnsafeBufferPointer(start: base + header.bodyOffset, count: count - header.bodyOffset), encoding: .utf8) else {
        throw PLYPointCloudLoaderError.truncatedBody
    }

    var positions: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    positions.reserveCapacity(header.vertexCount)
    colors.reserveCapacity(header.vertexCount)

    let lines = body.split(whereSeparator: \.isNewline)
    guard lines.count >= header.vertexCount else { throw PLYPointCloudLoaderError.truncatedBody }

    for index in 0 ..< header.vertexCount {
        let values = lines[index].split(separator: " ")
        guard values.count >= layout.fields.count else { throw PLYPointCloudLoaderError.truncatedBody }
        var position = SIMD3<Float>.zero
        var color = SIMD4<Float>(1, 1, 1, 1)

        for (fieldIndex, field) in layout.fields.enumerated() {
            guard let value = Float(values[fieldIndex]) else {
                throw PLYPointCloudLoaderError.invalidASCIIValue(line: index + 1, property: field.name)
            }
            assign(value: value, field: field, position: &position, color: &color)
        }

        positions.append(position)
        colors.append(color)
    }

    return (positions, colors)
}

private func parseBinaryLittleEndian(
    base: UnsafePointer<UInt8>,
    count: Int,
    header: PLYHeader,
    layout: PLYVertexLayout
) throws -> (positions: [SIMD3<Float>], colors: [SIMD4<Float>]) {
    let byteCount = header.vertexCount * layout.stride
    guard header.bodyOffset + byteCount <= count else { throw PLYPointCloudLoaderError.truncatedBody }

    var positions: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    positions.reserveCapacity(header.vertexCount)
    colors.reserveCapacity(header.vertexCount)

    for index in 0 ..< header.vertexCount {
        let vertexBase = base + header.bodyOffset + index * layout.stride
        var position = SIMD3<Float>.zero
        var color = SIMD4<Float>(1, 1, 1, 1)

        for field in layout.fields {
            let value = readScalar(type: field.type, base: vertexBase + field.offset)
            assign(value: value, field: field, position: &position, color: &color)
        }

        positions.append(position)
        colors.append(color)
    }

    return (positions, colors)
}

private func assign(value: Float, field: PLYField, position: inout SIMD3<Float>, color: inout SIMD4<Float>) {
    switch field.semantic {
    case .x: position.x = value
    case .y: position.y = value
    case .z: position.z = value
    case .red: color.x = normalizedColor(value, field.type)
    case .green: color.y = normalizedColor(value, field.type)
    case .blue: color.z = normalizedColor(value, field.type)
    case .ignored: break
    }
}

private func normalizedColor(_ value: Float, _ type: PLYScalarType) -> Float {
    if type == .float32 || type == .float64 {
        return simd_clamp(value, 0, 1)
    }
    return simd_clamp(value / type.colorScale, 0, 1)
}

private func readScalar(type: PLYScalarType, base: UnsafePointer<UInt8>) -> Float {
    switch type {
    case .int8:
        return Float(Int8(bitPattern: base.pointee))
    case .uint8:
        return Float(base.pointee)
    case .int16:
        return Float(Int16(bitPattern: readUInt16LE(base)))
    case .uint16:
        return Float(readUInt16LE(base))
    case .int32:
        return Float(Int32(bitPattern: readUInt32LE(base)))
    case .uint32:
        return Float(readUInt32LE(base))
    case .float32:
        return Float(bitPattern: readUInt32LE(base))
    case .float64:
        return Float(Double(bitPattern: readUInt64LE(base)))
    }
}

private func readUInt16LE(_ base: UnsafePointer<UInt8>) -> UInt16 {
    UInt16(base[0]) | (UInt16(base[1]) << 8)
}

private func readUInt32LE(_ base: UnsafePointer<UInt8>) -> UInt32 {
    UInt32(base[0])
        | (UInt32(base[1]) << 8)
        | (UInt32(base[2]) << 16)
        | (UInt32(base[3]) << 24)
}

private func readUInt64LE(_ base: UnsafePointer<UInt8>) -> UInt64 {
    UInt64(readUInt32LE(base)) | (UInt64(readUInt32LE(base + 4)) << 32)
}

#if canImport(Metal)
import Metal
import Satin

public extension PLYPointCloudLoader {
    /// Fast path: parse a PLY into contiguous arrays, then pack on the **GPU**
    /// into a wholesale ``PointRasteriserPointCloud`` — replacing the ~2-minute
    /// CPU pack for large clouds. Requires a Satin `context`, a configured
    /// ``GPUPacker``, and a command `queue`. Callers without a device should use
    /// the CPU ``load(url:pointsPerBatch:shuffleBatches:)`` path.
    ///
    /// - Parameter shuffle: apply the whole-batch shuffle (default `true`,
    ///   matching the CPU path).
    static func loadCloudGPU(
        url: URL,
        context: Context,
        packer: GPUPacker,
        queue: MTLCommandQueue,
        pointsPerBatch: Int = pointRasteriserThreadsPerGroup * 80,
        lodCapacity: Int? = nil,
        shuffle: Bool = true,
        label: String = "PointRasteriserPointCloud"
    ) throws -> PointRasteriserPointCloud {
        let (positions, colors) = try loadArrays(url: url)
        return PointRasteriserPointCloud.gpuPacked(
            context: context, packer: packer, queue: queue,
            positions: positions, colors: colors,
            pointsPerBatch: pointsPerBatch, lodCapacity: lodCapacity, shuffle: shuffle, label: label
        )
    }
}
#endif
