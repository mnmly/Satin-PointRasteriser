// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Satin-PointRasteriser",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(
            name: "SatinPointRasteriser",
            targets: ["SatinPointRasteriser"]
        ),
        .library(
            name: "SatinPointRasteriserStreaming",
            targets: ["SatinPointRasteriserStreaming"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/Fabric-Project/Satin", exact: "20.0.0-Beta-1"),
        // Streaming adapter's residency pre-roll (submit(views:)/isResidencySettled/
        // residencyBudgetLimited) needs SwiftPDAL 1.26.0+.
        .package(url: "https://github.com/mnmly/SwiftPDAL", from: "1.26.0"),
    ],
    targets: [
        .target(
            name: "SatinPointRasteriser",
            dependencies: [
                .product(name: "Satin", package: "Satin"),
            ],
            resources: [
                .copy("Pipelines"),
            ]
        ),
        .testTarget(
            name: "SatinPointRasteriserTests",
            dependencies: ["SatinPointRasteriser"]
        ),
        // COPC/SwiftPDAL streaming adapter. Split from the core library so
        // non-streaming consumers never pay the C++ interop compile cost.
        .target(
            name: "SatinPointRasteriserStreaming",
            dependencies: [
                "SatinPointRasteriser",
                .product(name: "Satin", package: "Satin"),
                .product(name: "SwiftPDAL", package: "SwiftPDAL"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .testTarget(
            name: "SatinPointRasteriserStreamingTests",
            dependencies: [
                "SatinPointRasteriser",
                "SatinPointRasteriserStreaming",
                .product(name: "Satin", package: "Satin"),
                .product(name: "SwiftPDAL", package: "SwiftPDAL"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
    ]
)

// The example is a SwiftUI + AppKit app (Satin's SatinMetalView); only add it
// when the manifest is evaluated on macOS so non-macOS builds skip it entirely.
#if os(macOS)
package.products.append(
    .executable(name: "PointRasteriserExample", targets: ["PointRasteriserExample"])
)
package.targets.append(
    .executableTarget(
        name: "PointRasteriserExample",
        dependencies: [
            "SatinPointRasteriser",
            "SatinPointRasteriserStreaming",
            .product(name: "Satin", package: "Satin"),
            .product(name: "SwiftPDAL", package: "SwiftPDAL"),
        ],
        swiftSettings: [
            // Required to import SatinPointRasteriserStreaming (SwiftPDAL is
            // a C++ interop module). Streaming-specific code in this target
            // is still guarded with `#if canImport(SwiftPDAL)` so the app
            // degrades gracefully if the dependency is ever unavailable.
            .interoperabilityMode(.Cxx),
        ]
    )
)
package.products.append(
    .executable(name: "PointRasteriserBench", targets: ["PointRasteriserBench"])
)
package.targets.append(
    .executableTarget(
        name: "PointRasteriserBench",
        dependencies: [
            "SatinPointRasteriser",
            .product(name: "Satin", package: "Satin"),
        ]
    )
)
#endif
