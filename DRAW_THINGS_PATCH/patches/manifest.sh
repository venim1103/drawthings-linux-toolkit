#!/usr/bin/env bash

# Shared patch manifest for Draw Things patch tooling.
# Keep this file as the single source of truth for files managed by:
# - tools/apply_drawthings_quant_patch.sh
# - tools/generate_drawthings_quant_patches.sh
# - tools/sync_drawthings_patch_bundle.sh

PATCH_ROOT_FILES=(
  "Package.swift"
  "Package.resolved"
  "Apps/ModelConverter/Converter.swift"
  "Libraries/ModelOp/Sources/ModelImporter.swift"
  "Libraries/ModelZoo/Sources/ModelZoo.swift"
  "Apps/ModelQuantizer/Quantizer.swift"
  "Libraries/SwiftDiffusion/Sources/Functional+SwishMul.swift"
  "Libraries/SwiftDiffusion/Sources/TextEncoder.swift"
  "Libraries/SwiftDiffusion/Sources/Archive/SafeTensors.swift"
  "Libraries/SwiftDiffusion/Sources/Models/HiDream.swift"
  "Libraries/SwiftDiffusion/Sources/Models/LTX2.swift"
  "Libraries/SwiftDiffusion/Sources/Extensions/TensorDescriptor.swift"
  "Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift"
  "Libraries/LocalImageGenerator/Sources/ImageConverter.swift"
  "Libraries/GRPC/Server/Sources/GRPCServerAdvertiser.swift"
  "Libraries/GRPC/Server/Sources/GRPCServiceBrowser.swift"
  "Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift"
)

PATCH_S4NNC_FILES=(
  "Package.swift"
  "nnc/Store.swift"
)

PATCH_CCV_FILES=(
  "Package.swift"
  "lib/nnc/ccv_cnnp_model.c"
  "lib/nnc/ccv_cnnp_model_addons.c"
  "lib/nnc/ccv_nnc_tensor.c"
  "lib/nnc/ccv_nnc_cmd.c"
  "lib/nnc/cmd/scaled_dot_product_attention/ccv_nnc_scaled_dot_product_attention.c"
)
