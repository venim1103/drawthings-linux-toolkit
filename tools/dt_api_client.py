#!/usr/bin/env python3
import argparse
import base64
import binascii
import hashlib
import json
import sys
import time
from pathlib import Path

import grpc
from google.protobuf.json_format import MessageToDict

DEFAULT_PROTO = Path(
    "/workspaces/LTX2_3/draw-things-community/Libraries/GRPC/Models/Sources/imageService/imageService.proto"
)
DEFAULT_HOST = "127.0.0.1:7859"


def _detect_extension(payload: bytes, default_ext: str = ".bin") -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"RIFF") and b"WAVE" in payload[:16]:
        return ".wav"
    if len(payload) > 12 and payload[4:8] == b"ftyp":
        return ".mp4"
    if payload.startswith(b"OggS"):
        return ".ogg"
    return default_ext


def _write_payload(payload: bytes, out_dir: Path, stem: str, default_ext: str = ".bin") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = _detect_extension(payload, default_ext)
    out_path = out_dir / f"{stem}{ext}"
    out_path.write_bytes(payload)
    return out_path


def _device_value(pb2, name: str) -> int:
    n = name.strip().lower()
    if n == "phone":
        return pb2.PHONE
    if n == "tablet":
        return pb2.TABLET
    if n == "laptop":
        return pb2.LAPTOP
    raise ValueError("device must be one of: phone, tablet, laptop")


def _decode_b64_to_bytes(raw: str) -> bytes:
    txt = raw.strip()
    if not txt:
        raise ValueError("base64 config input is empty")

    if txt.startswith("data:") and "," in txt:
        txt = txt.split(",", 1)[1].strip()

    try:
        return base64.b64decode(txt, validate=True)
    except binascii.Error:
        # Fallback for URL-safe base64 or missing padding.
        normalized = txt.replace("-", "+").replace("_", "/")
        normalized += "=" * ((4 - (len(normalized) % 4)) % 4)
        return base64.b64decode(normalized)


def _resolve_config_bytes(args) -> tuple[bytes, str | None]:
    if args.config_bin:
        config_path = Path(args.config_bin)
        if not config_path.exists():
            raise FileNotFoundError(f"Config bytes file not found: {config_path}")
        return config_path.read_bytes(), None

    if args.config_b64:
        config_bytes = _decode_b64_to_bytes(args.config_b64)
    elif args.config_b64_file:
        config_b64_path = Path(args.config_b64_file)
        if not config_b64_path.exists():
            raise FileNotFoundError(f"Base64 config file not found: {config_b64_path}")
        config_bytes = _decode_b64_to_bytes(config_b64_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Provide one of --config-bin, --config-b64, --config-b64-file")

    materialized_dir = Path(args.cache_dir) / "materialized_config"
    materialized_dir.mkdir(parents=True, exist_ok=True)
    materialized_path = materialized_dir / f"config_{int(time.time())}.bin"
    materialized_path.write_bytes(config_bytes)
    return config_bytes, str(materialized_path)


def _compile_proto(proto_file: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_file.parent}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto_file),
    ]

    import subprocess

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to compile proto:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _load_stubs(proto_file: Path, cache_dir: Path):
    _compile_proto(proto_file, cache_dir)
    sys.path.insert(0, str(cache_dir))
    import imageService_pb2 as pb2  # type: ignore
    import imageService_pb2_grpc as pb2_grpc  # type: ignore

    return pb2, pb2_grpc


def _print_message(message) -> None:
    data = MessageToDict(message, preserving_proto_field_name=True)
    print(json.dumps(data, indent=2, sort_keys=True))


def cmd_echo(stub, pb2, args):
    req = pb2.EchoRequest(name=args.name)
    if args.shared_secret:
        req.sharedSecret = args.shared_secret
    _print_message(stub.Echo(req))


def cmd_hours(stub, pb2, _args):
    _print_message(stub.Hours(pb2.HoursRequest()))


def cmd_files_exist(stub, pb2, args):
    req = pb2.FileListRequest(files=args.files)
    if args.shared_secret:
        req.sharedSecret = args.shared_secret
    _print_message(stub.FilesExist(req))


def cmd_generate_raw(stub, pb2, args):
    config_bytes, materialized_path = _resolve_config_bytes(args)
    req = pb2.ImageGenerationRequest(
        scaleFactor=args.scale_factor,
        prompt=args.prompt,
        negativePrompt=args.negative_prompt,
        configuration=config_bytes,
        user=args.user,
        device=_device_value(pb2, args.device),
        chunked=args.chunked,
    )

    if args.shared_secret:
        req.sharedSecret = args.shared_secret

    if args.keyword:
        req.keywords.extend(args.keyword)

    content_hashes = []
    for file_path_str in args.content_file:
        p = Path(file_path_str)
        if not p.exists():
            raise FileNotFoundError(f"Content file not found: {p}")
        blob = p.read_bytes()
        digest = hashlib.sha256(blob).digest()
        content_hashes.append(digest)
        req.contents.append(blob)

    if args.image_content_index is not None:
        i = args.image_content_index
        if i < 0 or i >= len(content_hashes):
            raise IndexError("image-content-index is out of range for --content-file list")
        req.image = content_hashes[i]

    if args.mask_content_index is not None:
        i = args.mask_content_index
        if i < 0 or i >= len(content_hashes):
            raise IndexError("mask-content-index is out of range for --content-file list")
        req.mask = content_hashes[i]

    if args.image_hash_hex:
        req.image = bytes.fromhex(args.image_hash_hex)

    if args.mask_hash_hex:
        req.mask = bytes.fromhex(args.mask_hash_hex)

    out_dir = Path(args.output_dir)
    print("sending GenerateImage request", flush=True)
    print(f"host: {args.host}", flush=True)
    print(f"config bytes: {len(config_bytes)}", flush=True)
    if materialized_path:
        print(f"materialized config path: {materialized_path}", flush=True)
    print(f"content blobs: {len(req.contents)}", flush=True)
    if args.image_content_index is not None or args.image_hash_hex:
        print("image hash provided")
    if args.mask_content_index is not None or args.mask_hash_hex:
        print("mask hash provided")

    response_count = 0
    image_count = 0
    audio_count = 0
    preview_count = 0
    pending_image_chunk = b""
    pending_audio_chunk = b""

    try:
        for response_count, resp in enumerate(stub.GenerateImage(req), start=1):
            print(f"response #{response_count}")
            if resp.tags:
                print(f"tags: {list(resp.tags)}")
            if resp.currentSignpost is not None:
                signpost = resp.currentSignpost.WhichOneof("signpost")
                print(f"current signpost: {signpost}")
            if resp.downloadSize:
                print(f"download size: {resp.downloadSize}")
            if resp.remoteDownload is not None:
                rd = resp.remoteDownload
                print(
                    "remote download progress: "
                    f"{rd.bytesReceived}/{rd.bytesExpected} item {rd.item}/{rd.itemsExpected} {rd.tag}"
                )

            if resp.previewImage:
                preview_count += 1
                if args.save_preview:
                    path = _write_payload(resp.previewImage, out_dir, f"preview_{response_count:04d}")
                    print(f"wrote preview: {path}")

            for i, blob in enumerate(resp.generatedImages, start=1):
                # Chunked streams split payloads as MORE_CHUNKS + LAST_CHUNK.
                # Reassemble the first payload slot across responses.
                if i == 1 and pending_image_chunk:
                    blob = pending_image_chunk + blob
                    pending_image_chunk = b""

                if resp.chunkState == pb2.MORE_CHUNKS and i == 1:
                    pending_image_chunk = blob
                    continue

                image_count += 1
                path = _write_payload(blob, out_dir, f"image_r{response_count:04d}_{i:02d}")
                print(f"wrote image: {path}")

            for i, blob in enumerate(resp.generatedAudio, start=1):
                if i == 1 and pending_audio_chunk:
                    blob = pending_audio_chunk + blob
                    pending_audio_chunk = b""

                if resp.chunkState == pb2.MORE_CHUNKS and i == 1:
                    pending_audio_chunk = blob
                    continue

                audio_count += 1
                path = _write_payload(blob, out_dir, f"audio_r{response_count:04d}_{i:02d}")
                print(f"wrote audio: {path}")

            if args.max_responses > 0 and response_count >= args.max_responses:
                print(f"stopping early at max responses: {args.max_responses}")
                break

            if resp.chunkState == pb2.LAST_CHUNK:
                print("chunk state: LAST_CHUNK")
    except grpc.RpcError as exc:
        print(f"gRPC error: {exc.code().name}: {exc.details()}")
        return

    print("generation stream finished")
    print(f"responses: {response_count}")
    print(f"images written: {image_count}")
    print(f"audio written: {audio_count}")
    print(f"preview frames seen: {preview_count}")
    if pending_image_chunk:
        print(f"warning: leftover partial image chunk bytes: {len(pending_image_chunk)}")
    if pending_audio_chunk:
        print(f"warning: leftover partial audio chunk bytes: {len(pending_audio_chunk)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw Things gRPC API client")
    parser.add_argument("--host", default=DEFAULT_HOST, help="gRPC host:port")
    parser.add_argument(
        "--proto",
        default=str(DEFAULT_PROTO),
        help="Path to imageService.proto",
    )
    parser.add_argument(
        "--cache-dir",
        default="/workspaces/LTX2_3/.cache/dt_proto",
        help="Directory for generated Python stubs",
    )
    parser.add_argument(
        "--max-recv-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="gRPC max receive message size in bytes",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_echo = sub.add_parser("echo", help="Call Echo")
    p_echo.add_argument("--name", default="api-test")
    p_echo.add_argument("--shared-secret", default="")

    p_hours = sub.add_parser("hours", help="Call Hours")

    p_files = sub.add_parser("files-exist", help="Call FilesExist")
    p_files.add_argument("files", nargs="+")
    p_files.add_argument("--shared-secret", default="")

    p_generate = sub.add_parser("generate-raw", help="Call GenerateImage stream with raw config bytes")
    config_group = p_generate.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config-bin", help="Path to raw flatbuffer configuration bytes")
    config_group.add_argument("--config-b64", help="Inline base64 flatbuffer bytes")
    config_group.add_argument("--config-b64-file", help="Path to text file containing base64 flatbuffer bytes")
    p_generate.add_argument("--prompt", default="")
    p_generate.add_argument("--negative-prompt", default="")
    p_generate.add_argument("--scale-factor", type=int, default=1)
    p_generate.add_argument("--device", default="laptop", choices=["phone", "tablet", "laptop"])
    p_generate.add_argument("--user", default="python-cli")
    p_generate.add_argument("--keyword", action="append", default=[])
    p_generate.add_argument("--chunked", action="store_true")
    p_generate.add_argument("--shared-secret", default="")
    p_generate.add_argument("--content-file", action="append", default=[])
    p_generate.add_argument("--image-content-index", type=int)
    p_generate.add_argument("--mask-content-index", type=int)
    p_generate.add_argument("--image-hash-hex", default="", help="Optional SHA256 hex for image field")
    p_generate.add_argument("--mask-hash-hex", default="", help="Optional SHA256 hex for mask field")
    p_generate.add_argument("--output-dir", default="/workspaces/LTX2_3/output/dt_api")
    p_generate.add_argument("--save-preview", action="store_true")
    p_generate.add_argument("--max-responses", type=int, default=0, help="0 means no limit")

    args = parser.parse_args()

    proto_file = Path(args.proto)
    if not proto_file.exists():
        raise FileNotFoundError(f"Proto file not found: {proto_file}")

    cache_dir = Path(args.cache_dir)
    pb2, pb2_grpc = _load_stubs(proto_file, cache_dir)

    channel_options = [
        ("grpc.max_receive_message_length", args.max_recv_bytes),
        ("grpc.max_send_message_length", args.max_recv_bytes),
    ]

    with grpc.insecure_channel(args.host, options=channel_options) as channel:
        stub = pb2_grpc.ImageGenerationServiceStub(channel)
        if args.command == "echo":
            cmd_echo(stub, pb2, args)
        elif args.command == "hours":
            cmd_hours(stub, pb2, args)
        elif args.command == "files-exist":
            cmd_files_exist(stub, pb2, args)
        elif args.command == "generate-raw":
            cmd_generate_raw(stub, pb2, args)
        else:
            raise ValueError(f"Unsupported command: {args.command}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
