import ArgumentParser
import Diffusion
import Foundation
import ModelOp
import ModelZoo
import NNC
#if os(Linux)
import Glibc
#else
import Darwin
#endif

@main
struct Converter: ParsableCommand {
  @Option(
    name: .shortAndLong,
    help: "The model file that is either the safetensors or the PyTorch checkpoint.")
  var file: String
  @Option(name: .shortAndLong, help: "The name of the model.")
  var name: String
  @Option(help: "The autoencoder file.")
  var autoencoderFile: String? = nil
  @Option(name: .shortAndLong, help: "The directory to write the output files to.")
  var outputDirectory: String
  @Flag(help: "Whether to convert the text encoder(s).")
  var textEncoders = false
  @Option(help: "Override math backend thread count. Defaults to available CPU cores.")
  var mathThreads: Int? = nil
  @Flag(help: "Suppress progress updates on stderr.")
  var quiet = false

  private struct Specification: Codable {
    var name: String
    var file: String
    var version: ModelVersion
    var modifier: SamplerModifier
    var textEncoder: String?
    var autoencoder: String?
    var clipEncoder: String?
    var t5Encoder: String?
    var guidanceEmbed: Bool?
  }

  private func writeStderr(_ message: String) {
    FileHandle.standardError.write(Data(message.utf8))
  }

  private func formatTimestamp() -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    return formatter.string(from: Date())
  }

  private func formatBytes(_ bytes: UInt64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"]
    var value = Double(bytes)
    var unitIndex = 0
    while value >= 1024 && unitIndex < units.count - 1 {
      value /= 1024
      unitIndex += 1
    }
    return String(format: "%.2f %@", value, units[unitIndex])
  }

  private func logStage(_ message: String) {
    writeStderr("[\(formatTimestamp())] converter: \(message)\n")
  }

  private func configureMathThreadEnvironment() {
    let requestedThreads = mathThreads ?? ProcessInfo.processInfo.activeProcessorCount
    let threadCount = max(requestedThreads, 1)
    let threadCountString = String(threadCount)
    let env = ProcessInfo.processInfo.environment
    let mathThreadKeys = [
      "OPENBLAS_NUM_THREADS",
      "OMP_NUM_THREADS",
      "MKL_NUM_THREADS",
      "GOTO_NUM_THREADS",
      "BLIS_NUM_THREADS",
      "VECLIB_MAXIMUM_THREADS",
      "NUMEXPR_NUM_THREADS",
    ]

    var applied = [String]()
    for key in mathThreadKeys {
      if mathThreads == nil && env[key] != nil {
        continue
      }
      setenv(key, threadCountString, 1)
      applied.append("\(key)=\(threadCountString)")
    }

    if !quiet && !applied.isEmpty {
      writeStderr("configured math threads: \(applied.joined(separator: ", "))\n")
    }
  }

  mutating func run() throws {
    let startedAt = Date()
    configureMathThreadEnvironment()

    let fileManager = FileManager.default
    let inputURL = URL(fileURLWithPath: file).standardizedFileURL
    let outputURL = URL(fileURLWithPath: outputDirectory).standardizedFileURL
    let cleanedModelName = Importer.cleanup(filename: name)

    guard fileManager.fileExists(atPath: inputURL.path) else {
      logStage("input file missing: \(inputURL.path)")
      throw ValidationError("input file not found: \(inputURL.path)")
    }

    try fileManager.createDirectory(at: outputURL, withIntermediateDirectories: true)

    if let attrs = try? fileManager.attributesOfItem(atPath: inputURL.path),
      let fileSize = attrs[.size] as? NSNumber
    {
      logStage(
        "starting conversion model=\(name) cleanedModel=\(cleanedModelName) input=\(inputURL.path) size=\(formatBytes(fileSize.uint64Value)) outputDir=\(outputURL.path) textEncoders=\(textEncoders)"
      )
    } else {
      logStage(
        "starting conversion model=\(name) cleanedModel=\(cleanedModelName) input=\(inputURL.path) outputDir=\(outputURL.path) textEncoders=\(textEncoders)"
      )
    }

    ModelZoo.externalUrls = [outputURL]
    let fileName = cleanedModelName
    let importer = ModelImporter(
      filePath: file, modelName: fileName,
      isTextEncoderCustomized: textEncoders,
      autoencoderFilePath: autoencoderFile, textEncoderFilePath: nil, textEncoder2FilePath: nil)
    let shouldReportProgress = !quiet
    var lastPercent = -1
    let reportProgress: (Float) -> Void = { progress in
      if !shouldReportProgress {
        return
      }

      let clamped = max(0, min(progress, 1))
      let percent = Int((clamped * 100).rounded(.down))
      if percent <= lastPercent {
        return
      }
      lastPercent = percent

      let barWidth = 40
      let filled = min(barWidth, max(0, Int((clamped * Float(barWidth)).rounded(.down))))
      let bar = String(repeating: "#", count: filled)
        + String(repeating: "-", count: barWidth - filled)
      let paddedPercent = String(format: "%3d", percent)
      FileHandle.standardError.write(Data("convert progress \(paddedPercent)% [\(bar)]\n".utf8))
    }
    var phase = "import"
    do {
      logStage("phase=\(phase) started")
      let (filePaths, modelVersion, modifier, inspectionResult) = try importer.import { _ in
      } progress: { progress in
        reportProgress(progress)
      }
      if shouldReportProgress {
        reportProgress(1)
      }
      logStage(
        "phase=\(phase) finished generatedFiles=\(filePaths.count) version=\(modelVersion) modifier=\(modifier)"
      )

    let fileNames = filePaths.map { ($0 as NSString).lastPathComponent }
    var autoencoder = fileNames.first {
      $0.hasSuffix("_vae_f16.ckpt")
    }
    var clipEncoder: String? = nil
    let textEncoder: String?
    var t5Encoder: String? = nil
    switch modelVersion {
    case .v1:
      textEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
    case .v2:
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_h14_f16.ckpt")
      }
    case .sdxlBase, .ssd1b:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .sd3:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
      t5Encoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sd3_vae_f16.ckpt"
      }
    case .sd3Large:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
      t5Encoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sd3_vae_f16.ckpt"
      }
    case .pixart:
      textEncoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .sdxlRefiner:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
        } ?? "open_clip_vit_bigg14_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .svdI2v:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_open_clip_vit_h14_f16.ckpt")
        } ?? "open_clip_vit_h14_vision_model_f16.ckpt"
      if clipEncoder == nil {
        clipEncoder = "open_clip_vit_h14_visual_proj_f16.ckpt"
      }
    case .flux1:
      textEncoder = "t5_xxl_encoder_q6p.ckpt"
      clipEncoder = "clip_vit_l14_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "flux_1_vae_f16.ckpt"
      }
    case .hunyuanVideo:
      textEncoder = "llava_llama_3_8b_v1.1_q8p.ckpt"
      clipEncoder = "clip_vit_l14_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "hunyuan_video_vae_f16.ckpt"
      }
    case .wan21_1_3b, .wan21_14b, .wan22_5b:
      textEncoder = "umt5_xxl_encoder_q8p.ckpt"
      if modifier == .inpainting {
        clipEncoder = "open_clip_xlm_roberta_large_vit_h14_f16.ckpt"
      }
      clipEncoder = "clip_vit_l14_f16.ckpt"
      if autoencoder == nil {
        if modelVersion == .wan22_5b {
          autoencoder = "wan_v2.2_video_vae_f16.ckpt"
        } else {
          autoencoder = "wan_v2.1_video_vae_f16.ckpt"
        }
      }
    case .hiDreamI1:
      fatalError()
    case .qwenImage:
      fatalError()
    case .cosmos2_5_2b:
      textEncoder = "qwen_3_0.6b_f16.ckpt"
      clipEncoder = "\(fileName)_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "qwen_image_vae_f16.ckpt"
      }
    case .zImage:
      fatalError()
    case .ernieImage:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_ministral_3_3b_f16.ckpt")
        } ?? "ministral_3_3b_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "flux_2_vae_f16.ckpt"
      }
    case .seedvr2_3b, .seedvr2_7b:
      textEncoder = "\(fileName)_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "seedvr2_vae_f16.ckpt"
      }
    case .flux2, .flux2_9b, .flux2_4b:
      fatalError()
    case .ltx2, .ltx2_3:
      textEncoder = nil
    case .kandinsky21, .wurstchenStageC, .wurstchenStageB, .auraflow:
      fatalError()
    }
    var specification = Specification(
      name: name, file: "\(fileName)_f16.ckpt", version: modelVersion, modifier: modifier,
      textEncoder: textEncoder, autoencoder: autoencoder, clipEncoder: clipEncoder,
      t5Encoder: t5Encoder)
    if inspectionResult.hasGuidanceEmbed {
      specification.guidanceEmbed = true
    }
      phase = "encode specification"
      logStage("phase=\(phase) started")
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    let jsonData = try jsonEncoder.encode(specification)
    print(String(decoding: jsonData, as: UTF8.self))

      let elapsed = Int(Date().timeIntervalSince(startedAt))
      logStage(
        "completed successfully elapsed=\(elapsed)s primaryFile=\(fileName)_f16.ckpt outputDir=\(outputURL.path)"
      )
    } catch {
      let elapsed = Int(Date().timeIntervalSince(startedAt))
      logStage("failed phase=\(phase) elapsed=\(elapsed)s error=\(error)")
      if let importerError = error as? ModelImporter.Error {
        switch importerError {
        case .tensorWritesFailed:
          logStage("failure detail: tensorWritesFailed")
        case .noTextEncoder:
          logStage("failure detail: noTextEncoder")
        case .textEncoder(let nestedError):
          logStage("failure detail: textEncoder nestedError=\(nestedError)")
        case .autoencoder(let nestedError):
          logStage("failure detail: autoencoder nestedError=\(nestedError)")
        case .unsupportedSourceFormat(let message):
          logStage("failure detail: unsupportedSourceFormat=\(message)")
        }
      }
      if let unpickleError = error as? UnpickleError {
        logStage("failure detail: unpickleError=\(unpickleError)")
      }
      throw error
    }
  }
}
