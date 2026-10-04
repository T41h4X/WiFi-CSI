"""
Environment and Checkpoint Verification Script
Verifies Python dependencies, PyTorch CUDA/CPU device, and model checkpoint integrity.
"""

import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("       CSI-HAR Environment & Checkpoint Verification       ")
    print("=" * 60)

    # 1. Python version
    print(f"[*] Python Version: {sys.version.split()[0]}")

    # 2. PyTorch & Device
    try:
        import torch
        print(f"[+] PyTorch Version: {torch.__version__}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Compute Device : {device}")
        if device == "cuda":
            print(f"[*] GPU Name       : {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[-] Error: PyTorch is not installed.")
        sys.exit(1)

    # 3. Scientific Packages
    for pkg in ["numpy", "pandas", "sklearn", "scipy", "matplotlib"]:
        try:
            __import__(pkg)
            print(f"[+] Package '{pkg}' is available.")
        except ImportError:
            print(f"[-] Warning: Package '{pkg}' is NOT installed.")

    # 4. Check model checkpoints
    ckpt_v1 = Path("models/csi_har_cnn_best.pth")
    ckpt_v2 = Path("models/csi_har_cnn_v2_best.pth")

    print("-" * 60)
    for name, p in [("v1 Baseline Checkpoint", ckpt_v1), ("v2 Production Checkpoint", ckpt_v2)]:
        if p.exists():
            data = torch.load(p, map_location="cpu", weights_only=False)
            keys = list(data.keys())
            print(f"[+] Found {name}: {p} ({p.stat().st_size / 1024:.1f} KB, keys: {keys})")
        else:
            print(f"[-] Missing {name} at {p}")

    print("=" * 60)
    print("[+] Verification complete!")

if __name__ == "__main__":
    main()
