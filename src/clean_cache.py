import os
import shutil
import gc
import torch
from config import MODEL_CACHE_PATH


def nuke_hf_cache(cache_dir=None):
    print("🔄 Resetting memory...")
    
    cache_dir = os.path.abspath(cache_dir)
    # Safety check
    if len(cache_dir) < 10 or cache_dir in ["/", "/root", "/home"]:
        raise RuntimeError(f"Refusing to delete unsafe path: {cache_dir}")

    print(f"\n💣 NUKING EVERYTHING in: {cache_dir}")

    for name in os.listdir(cache_dir):
        path = os.path.join(cache_dir, name)
        print(f"  → Deleting {path}")
        shutil.rmtree(path, ignore_errors=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("✓ hf_cache wiped completely\n")

if __name__ == "__main__":
    nuke_hf_cache(cache_dir=MODEL_CACHE_PATH)