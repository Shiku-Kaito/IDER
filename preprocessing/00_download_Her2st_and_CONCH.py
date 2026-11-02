"""Download the data needed for preprocessing: Her2st (Andersson et al., Zenodo record 3957257) raw data into supp_script/data/Her2st/org.

Resulting layout (expected by 01_align_directory.py):
  org/images/images/HE/...
  org/count-matrices/count-matrices/...
  org/spot-selections/*.tsv.gz
  org/meta/*_labeled_coordinates.tsv
Files that already exist are skipped, downloads are md5-verified.

The zip files are AES-encrypted with two different passwords (published in the README of the original
repository, https://github.com/almaan/her2st):
  count-matrices.zip, images.zip      -> --password_counts_images / HER2ST_ZIP_PASSWORD_COUNTS_IMAGES
  meta.zip, spot-selections.zip       -> --password_meta_spots    / HER2ST_ZIP_PASSWORD_META_SPOTS
Extraction uses `pyzipper` (pip install pyzipper), falling back to the `7z` command.
"""
import argparse
import subprocess
import hashlib
import os
import urllib.request

DATA_ROOT = os.environ.get("IDER_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
BASE_URL = "https://zenodo.org/records/3957257/files/%s.zip?download=1"
FILES = {  # name: (md5 from the Zenodo record, password group)
    "count-matrices": ("d0dd522b61892435d21ac437af289ae5", "counts_images"),
    "images": ("383e23ac398790dbdb408ec7b8b5d3b2", "counts_images"),
    "meta": ("57d358398d7f8cc1410398aa032ff029", "meta_spots"),
    "spot-selections": ("edd47d57c97d9563ca9f3ccc70767c7b", "meta_spots"),
}


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def progress(name):
    last = [-1]

    def hook(block, block_size, total):
        pct = min(100, block * block_size * 100 // total) if total > 0 else 0
        if pct // 10 != last[0]:
            last[0] = pct // 10
            print(f"  {name}: {pct}%", flush=True)
    return hook


def extract(zip_path, out_dir, password):
    try:
        import pyzipper
    except ImportError:
        subprocess.run(["7z", "x", "-y", "-p" + password, "-o" + out_dir, zip_path], check=True)
        return
    with pyzipper.AESZipFile(zip_path) as z:
        z.extractall(out_dir, pwd=password.encode())


def main(passwords):
    zip_dir = os.path.join(DATA_ROOT, "Her2st", "zip")
    org_dir = os.path.join(DATA_ROOT, "Her2st", "org")
    os.makedirs(zip_dir, exist_ok=True)
    for name, (md5, group) in FILES.items():
        zip_path = os.path.join(zip_dir, name + ".zip")
        out_dir = os.path.join(org_dir, name)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            print("skip (already extracted):", out_dir)
            continue
        if not (os.path.exists(zip_path) and md5sum(zip_path) == md5):
            print("downloading", name)
            urllib.request.urlretrieve(BASE_URL % name, zip_path, reporthook=progress(name))
            if md5sum(zip_path) != md5:
                raise RuntimeError("md5 mismatch: " + zip_path)
        os.makedirs(out_dir, exist_ok=True)
        extract(zip_path, out_dir, passwords[group])
        print("extracted:", out_dir)


def download_conch(token):
    """CONCH weights (gated model): accept the terms at https://huggingface.co/MahmoodLab/CONCH
    and pass a Hugging Face token with --hf_token or HF_TOKEN."""
    from huggingface_hub import hf_hub_download
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "conch")
    os.makedirs(out_dir, exist_ok=True)
    path = hf_hub_download(repo_id="MahmoodLab/CONCH", filename="pytorch_model.bin", local_dir=out_dir, token=token)
    print("saved:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--password_counts_images", default=os.environ.get("HER2ST_ZIP_PASSWORD_COUNTS_IMAGES"))
    parser.add_argument("--password_meta_spots", default=os.environ.get("HER2ST_ZIP_PASSWORD_META_SPOTS"))
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--only", choices=["data", "conch"], default=None)
    args = parser.parse_args()
    if args.only != "conch":
        if not (args.password_counts_images and args.password_meta_spots):
            parser.error("both zip passwords are required (see the docstring)")
        main({"counts_images": args.password_counts_images, "meta_spots": args.password_meta_spots})
    if args.only != "data":
        download_conch(args.hf_token)
