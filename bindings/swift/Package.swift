// swift-tools-version:5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Transcode",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .tvOS(.v15)
    ],
    products: [
        .library(
            name: "Transcode",
            targets: ["Transcode"]
        ),
        .executable(
            name: "TranscodeExample",
            targets: ["TranscodeExample"]
        )
    ],
    targets: [
        // C library wrapper
        .systemLibrary(
            name: "CTranscode",
            pkgConfig: "transcode",
            providers: [
                .apt(["libtranscode-dev"]),
                .brew(["transcode"])
            ]
        ),

        // Swift wrapper
        .target(
            name: "Transcode",
            dependencies: ["CTranscode"],
            path: "Sources/Transcode"
        ),

        // Example executable
        .executableTarget(
            name: "TranscodeExample",
            dependencies: ["Transcode"],
            path: "Examples/Basic"
        ),

        // Tests
        .testTarget(
            name: "TranscodeTests",
            dependencies: ["Transcode"],
            path: "Tests/TranscodeTests"
        )
    ]
)
