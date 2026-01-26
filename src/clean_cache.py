import os
import shutil
import gc
import torch


def nuke_hf_cache(base_cache_dir=None, aggressive=True):
    """
    Safely delete HuggingFace + ML caches from disk and free GPU memory

    aggressive=True will also delete:
      - ~/.cache/torch
      - ~/.cache/transformers
    """

    # Resolve HF base
    base = (
        base_cache_dir
        or os.environ.get("HF_HOME")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
        or os.path.expanduser("~/.cache/huggingface")
    )

    base = os.path.abspath(base)

    # Safety check
    if len(base) < 10 or base in ["/", "/root", "/home"]:
        raise RuntimeError(f"Refusing to nuke unsafe path: {base}")

    print(f"\n🧹 Nuking HuggingFace cache at: {base}")

    targets = ["hub", "xet", "assets", "tmp"]

    for name in targets:
        path = os.path.join(base, name)
        if os.path.exists(path):
            print(f"  → Deleting {path}")
            shutil.rmtree(path, ignore_errors=True)
        else:
            print(f"  → {path} not found (skip)")

    # Aggressive cleanup
    if aggressive:
        extra_paths = [
            os.path.expanduser("~/.cache/torch"),
            os.path.expanduser("~/.cache/transformers"),
        ]

        for p in extra_paths:
            p = os.path.abspath(p)
            if os.path.exists(p):
                print(f"🧹 Deleting {p}")
                shutil.rmtree(p, ignore_errors=True)

    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("✓ Cache wipe complete\n")
