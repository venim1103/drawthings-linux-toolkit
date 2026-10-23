import argparse
import ctypes
import json
import os
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path


def check_certificates() -> dict[str, object]:
    bundle = Path("/etc/ssl/certs/ca-certificates.crt")
    for variable in (
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "PIP_CERT",
    ):
        if os.environ.get(variable) != str(bundle):
            raise RuntimeError(f"{variable} must use {bundle}")
    context = ssl.create_default_context()
    count = context.cert_store_stats()["x509_ca"]
    if not bundle.is_file() or count == 0:
        raise RuntimeError("The system CA trust store is empty or missing")
    return {"bundle": str(bundle), "trusted_ca_count": count}


def check_cuda() -> dict[str, object]:
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise RuntimeError(
            "Cannot load libcuda.so.1; on WSL add /usr/lib/wsl/lib to LD_LIBRARY_PATH"
        ) from error

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check /dev/dxg and NVIDIA GPU passthrough")
    if not torch.backends.cudnn.is_available():
        raise RuntimeError("PyTorch has no cuDNN support")
    with torch.no_grad():
        values = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
        actual = (values @ torch.eye(4, device="cuda")).cpu()
        torch.testing.assert_close(actual, torch.arange(16, dtype=torch.float32).reshape(4, 4))
        features = torch.ones((1, 1, 8, 8), device="cuda")
        kernel = torch.ones((1, 1, 3, 3), device="cuda")
        convolved = torch.nn.functional.conv2d(features, kernel).cpu()
        torch.testing.assert_close(convolved, torch.full((1, 1, 6, 6), 9.0))
        torch.cuda.synchronize()
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_check": "passed",
    }


def check_network() -> list[str]:
    import requests

    addresses = (
        "https://github.com",
        "https://huggingface.co/api/models/drawthingsai/MOSS-Image-Base",
        "https://pypi.org/simple/pip/",
    )
    for address in addresses:
        try:
            with urllib.request.urlopen(address, timeout=20) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTPS status {response.status}")
            response = requests.get(address, timeout=20)
            response.raise_for_status()
        except Exception as error:
            raise RuntimeError(f"Verified HTTPS failed for {address}: {error}") from error
    result = subprocess.run(
        ["git", "ls-remote", "https://github.com/drawthingsai/draw-things-community.git", "HEAD"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git HTTPS failed: {result.stderr.strip()}")
    return list(addresses)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check devcontainer CA trust and CUDA without models")
    parser.add_argument("--network", action="store_true", help="Also verify HTTPS and Git access")
    options = parser.parse_args()
    try:
        results: dict[str, object] = {"certificates": check_certificates(), "cuda": check_cuda()}
        if options.network:
            results["verified_https"] = check_network()
    except Exception as error:
        print(f"Container smoke check failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(results, indent=2))
    print("Container smoke checks passed; no pretrained models were downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())