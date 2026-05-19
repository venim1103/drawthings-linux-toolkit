// swift-tools-version:5.9
import Foundation
import PackageDescription

let isLinuxHost = FileManager.default.fileExists(atPath: "/proc/sys/kernel/ostype")

let nncTargetDependencies: [Target.Dependency] = {
  var deps: [Target.Dependency] = [
    .product(name: "nnc", package: "ccv"),
    .product(name: "sfmt", package: "ccv"),
  ]
  if !isLinuxHost {
    deps.append(.product(name: "lib_nnc_mps_compat", package: "ccv"))
  }
  deps.append(.product(name: "SQLite3", package: "swift-sqlite3-support"))
  deps.append(.product(name: "C_fpzip", package: "swift-fpzip-support"))
  deps.append("C_zlib")
  return deps
}()

let coreMLConversionDependencies: [Target.Dependency] = {
  var deps: [Target.Dependency] = [
    "NNC",
  ]
  if !isLinuxHost {
    deps.append(.product(name: "lib_nnc_mps_compat", package: "ccv"))
  }
  deps.append(.product(name: "sfmt", package: "ccv"))
  return deps
}()

let package = Package(
  name: "s4nnc",
  platforms: [
    .macOS(.v13),
    .iOS(.v16),
    .tvOS(.v16),
    .visionOS(.v1),
  ],
  products: [
    .library(
      name: "NNC",
      targets: ["NNC"]),
    .library(
      name: "NNCCoreMLConversion",
      targets: ["NNCCoreMLConversion"]),
    .library(
      name: "TensorBoard",
      targets: ["TensorBoard"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/liuliu/ccv.git", revision: "0b3bb15153c19f120a9e41f842c26e6ab6c09ece"
    ),
    .package(
      url: "https://github.com/weiyanlin117/swift-fpzip-support.git",
      revision: "0ec6d4668c9c83bc3da0f8b2d6dfc46da0b98609"),
    .package(
      url: "https://github.com/apple/swift-protobuf.git",
      revision: "d57a5aecf24a25b32ec4a74be2f5d0a995a47c4b"),
    .package(
      url: "https://github.com/apple/swift-system.git",
      revision: "fbd61a676d79cbde05cd4fda3cc46e94d6b8f0eb"),
    .package(
      url: "https://github.com/liuliu/swift-sqlite3-support.git",
      revision: "9d543d0af5da0b81ae65be3a61ea6920647779a7"),
  ],
  targets: [
    // C_zlib - System zlib wrapper
    .systemLibrary(
      name: "C_zlib",
      path: "nnc/C_zlib"
    ),

    // NNC - Main Swift library
    .target(
      name: "NNC",
      dependencies: nncTargetDependencies,
      path: "nnc",
      exclude: [
        "C_ccv",
        "C_nnc",
        "C_sfmt",
        "C_zlib",
        "C_sqlite3",
        "BUILD.bazel",
        "CoreMLConversion.swift",
        "PythonConversion.swift",
        "MuJoCoConversion.swift",
      ],
      sources: [
        "AnyModel.swift",
        "AutoGrad.swift",
        "DataFrame.swift",
        "DataFrameAddons.swift",
        "DataFrameCore.swift",
        "DynamicGraph.swift",
        "Functional.swift",
        "FunctionalAddons.swift",
        "GradScaler.swift",
        "Group.swift",
        "Hint.swift",
        "Loss.swift",
        "Model.swift",
        "ModelAddons.swift",
        "ModelBuilder.swift",
        "ModelCore.swift",
        "ModelIOAddons.swift",
        "Operators.swift",
        "Optimizer.swift",
        "OptimizerAddons.swift",
        "Store.swift",
        "StreamContext.swift",
        "Tensor.swift",
        "TensorGroup.swift",
        "Wrapped.swift",
      ]
    ),

    // NNCCoreMLConversion - CoreML conversion utilities
    .target(
      name: "NNCCoreMLConversion",
      dependencies: coreMLConversionDependencies,
      path: "nnc",
      exclude: [
        "C_ccv",
        "C_nnc",
        "C_sfmt",
        "C_zlib",
        "C_sqlite3",
        "BUILD.bazel",
        "PythonConversion.swift",
        "MuJoCoConversion.swift",
        "AnyModel.swift",
        "AutoGrad.swift",
        "DataFrame.swift",
        "DataFrameAddons.swift",
        "DataFrameCore.swift",
        "DynamicGraph.swift",
        "Functional.swift",
        "FunctionalAddons.swift",
        "GradScaler.swift",
        "Group.swift",
        "Hint.swift",
        "Loss.swift",
        "Model.swift",
        "ModelAddons.swift",
        "ModelBuilder.swift",
        "ModelCore.swift",
        "ModelIOAddons.swift",
        "Operators.swift",
        "Optimizer.swift",
        "OptimizerAddons.swift",
        "Store.swift",
        "StreamContext.swift",
        "Tensor.swift",
        "TensorGroup.swift",
        "Wrapped.swift",
      ],
      sources: ["CoreMLConversion.swift"]
    ),

    // TensorBoard - TensorBoard logging support
    .target(
      name: "TensorBoard",
      dependencies: [
        "NNC",
        .product(name: "SwiftProtobuf", package: "swift-protobuf"),
        .product(name: "SystemPackage", package: "swift-system"),
      ],
      path: "tensorboard",
      exclude: [
        "BUILD.bazel",
        "README.md",
        "compat",
      ]
    ),

    // Test targets
    .testTarget(
      name: "NNCTests",
      dependencies: ["NNC"],
      path: "test",
      exclude: [
        "coreml",
        "python",
        "BUILD.bazel",
      ],
      sources: [
        "dataframe.swift",
        "graph.swift",
        "loss.swift",
        "model.swift",
        "ops.swift",
        "optimizer.swift",
        "store.swift",
        "tensor.swift",
      ],
      resources: [
        .copy("scaled_data.csv"),
        .copy("some_variables.db"),
      ]
    ),

    .testTarget(
      name: "NNCCoreMLTests",
      dependencies: ["NNCCoreMLConversion"],
      path: "test/coreml",
      sources: [
        "mlshapedarray.swift"
      ]
    ),
  ]
)
