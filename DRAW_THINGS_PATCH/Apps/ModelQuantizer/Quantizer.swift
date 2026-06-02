import ArgumentParser
import Diffusion
import NNC

enum QuantizationTargetCodec: String, ExpressibleByArgument {
  case auto
  case q4p
  case q5p
  case q6p
  case q8p
  case i8x
}

@main
struct Quantizer: ParsableCommand {
  @Option(
    name: .shortAndLong,
    help: "The input file to be converted.")
  var inputFile: String

  @Option(
    name: .shortAndLong,
    help: """
      The model version of the input file. Available versions:
      v1, v2, kandinsky2.1, sdxl_base_v0.9, sdxl_refiner_v0.9, ssd_1b, svd_i2v,
      wurstchen_v3.0_stage_c, wurstchen_v3.0_stage_b, sd3, pixart, auraflow,
      flux1, sd3_large, hunyuan_video, wan_v2.1_1.3b, wan_v2.1_14b, hidream_i1,
      qwen_image, wan_v2.2_5b, z_image, ernie_image, flux2, flux2_9b, flux2_4b, cosmos2.5_2b, ltx2, ltx2.3,
      seedvr2_3b, seedvr2_7b
      """)
  var modelVersion: String

  @Option(name: .shortAndLong, help: "The output file after conversion")
  var outputFile: String

  @Option(
    name: .long,
    help:
      "Target quantization codec. Use auto for model-specific defaults, or force one of: q4p, q5p, q6p, q8p, i8x."
  )
  var targetCodec: QuantizationTargetCodec = .auto

  mutating func run() throws {
    // Convert string to ModelVersion enum
    guard let version = ModelVersion(rawValue: modelVersion) else {
      throw ValidationError("Invalid model version: \(modelVersion)")
    }
    // Now you can use 'version' as your ModelVersion enum
    print("Converting \(inputFile), model version: \(version)")

    let forcedCodec: DynamicGraph.Store.Codec?
    switch targetCodec {
    case .auto:
      forcedCodec = nil
    case .q4p:
      forcedCodec = .q4p
    case .q5p:
      forcedCodec = .q5p
    case .q6p:
      forcedCodec = .q6p
    case .q8p:
      forcedCodec = .q8p
    case .i8x:
      forcedCodec = .i8x
    }
    if let forcedCodec {
      print("Forcing quantization codec: \(targetCodec.rawValue) (\(forcedCodec))")
    }

    let graph = DynamicGraph()
    graph.openStore(
      inputFile, flags: .readOnly, externalStore: TensorData.externalStore(filePath: inputFile)
    ) { store in
      let keys = store.keys

      graph.openStore(outputFile) {
        for key in keys {
          guard
            let tensor = store.read(
              key,
              codec: [
                .jit, .q4p, .q5p, .q6p, .q7p, .q8p, .i8x, .ezm7, .externalData,
                .externalOnDemand,
              ])
          else {
            continue
          }

          // First convert the tensor to FP16, and then to q8p.
          let fp16 = Tensor<FloatType>(from: tensor)
          let shape = fp16.shape
          let squeezedDims = shape.reduce(0) { $1 > 1 ? 1 + $0 : $0 }

          if let forcedCodec {
            if version == .ltx2 || version == .ltx2_3 {
              // LTX models are sensitive to blanket forced quantization.
              // Preserve key modulation / projection tensors, and keep higher precision for ada_ln / conv.
              let isLTXTextFeaturePath =
                key.contains("text_feature_extractor") || key.contains("text_video_connector")
                || key.contains("text_audio_connector")
              if isLTXTextFeaturePath {
                // Keep clip-side feature extractor and connector weights in FP16.
                // These tensors are consumed by TextEncoder.encodeLTX2 via filePaths[1].
                $0.write(key, tensor: fp16)
                continue
              }
              if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-")
                || key.contains("scale_shift_table") || key.contains("caption_projection")
                || key.contains("patchify_proj") || key.contains("proj_out")
              {
                $0.write(key, tensor: fp16)
                continue
              }
              if squeezedDims > 1 {
                if key.contains("ada_ln") || key.contains("adaln") || shape.count == 4 {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  $0.write(key, tensor: fp16, codec: [forcedCodec, .ezm7])
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
              continue
            }

            // Keep common fragile projection/positional tensors in FP16 when forcing a codec.
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("visual_proj")
              || key.contains("encoder_hid_proj") || key.contains("register_tokens")
              || key.contains("refiner_")
            {
              $0.write(key, tensor: fp16)
              continue
            }
            if squeezedDims > 1 {
              $0.write(key, tensor: fp16, codec: forcedCodec)
            } else {
              $0.write(key, tensor: fp16, codec: .ezm7)
            }
            continue
          }

          switch version {
          case .v1, .v2, .ssd1b, .svdI2v, .sdxlBase, .sdxlRefiner, .wurstchenStageB, .kandinsky21:
            if key.contains("visual_proj") || key.contains("encoder_hid_proj") {
              $0.write(key, tensor: tensor)
              continue
            }
            if shape.count == 2 && squeezedDims > 1 {
              $0.write(key, tensor: fp16, codec: [.q6p, .ezm7])
            } else if shape.count == 4 && squeezedDims > 1 {
              $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
            } else {
              $0.write(key, tensor: fp16, codec: .ezm7)
            }
          case .wurstchenStageC:
            if key.contains("text_emb") || key.contains("effnet") || key.contains("previewer") {
              $0.write(key, tensor: fp16)
            } else {
              if shape.count == 2 && squeezedDims > 1 {
                $0.write(key, tensor: fp16, codec: [.q6p, .ezm7])
              } else if shape.count == 4 && squeezedDims > 1 {
                $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .pixart:
            if key.contains("embedder") || key.contains("shift_table") || key.contains("t_block") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .sd3, .sd3Large:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("ada_ln") {
              $0.write(key, tensor: fp16)
            } else if key.contains("norm") {
              $0.write(key, tensor: fp16, codec: [.ezm7])
            } else {
              if squeezedDims > 1 {
                $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .auraflow:
            if key.contains("embedder") || key.contains("pos_embed")
              || key.contains("register_tokens")
            {
              $0.write(key, tensor: fp16)
            } else if key.contains("norm") {
              $0.write(key, tensor: fp16, codec: [.ezm7])
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") || key.contains("-linear-") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  $0.write(key, tensor: fp16, codec: [.q5p, .ezm7])
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .flux1:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .hunyuanVideo:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-")
              || key.contains("refiner_")
            {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: [.q5p, .ezm7])
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .wan21_1_3b, .wan22_5b:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .wan21_14b:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: [.q6p, .ezm7])
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .hiDreamI1:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else if shape.count == 3 {  // MoE.
                    $0.write(key, tensor: fp16, codec: [.q5p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: [.q6p, .ezm7])
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .qwenImage, .ernieImage, .seedvr2_3b, .seedvr2_7b:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: [.q6p, .ezm7])
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .cosmos2_5_2b:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-")
              || key.contains("_linear_")
            {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if shape.count == 4 {  // Convolution.
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  $0.write(key, tensor: fp16, codec: [.q6p, .ezm7])
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .zImage:
            if key.contains("embedder") || key.contains("pos_embed")
              || key.contains("-linear_final-")
            {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: [.q6p, .ezm7])
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .ltx2, .ltx2_3:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: [.q6p, .ezm7])
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .flux2, .flux2_9b, .flux2_4b:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          }
        }
      }
    }
  }
}
