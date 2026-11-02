import os
import shutil
from glob import glob

DATA_ROOT = os.environ.get("IDER_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))  # supp_script/data

# =========================
# input directories
# =========================
image_dir = os.path.join(DATA_ROOT, "Her2st/org/images/images/HE")
expr_dir = os.path.join(DATA_ROOT, "Her2st/org/count-matrices/count-matrices")
spot_dir = os.path.join(DATA_ROOT, "Her2st/org/spot-selections")
ann_dir = os.path.join(DATA_ROOT, "Her2st/org/meta")

# output directory
out_root = os.path.join(DATA_ROOT, "Her2st/preprocessed")

os.makedirs(out_root, exist_ok=True)

# =========================
# get slide ids from the image file names
# =========================
image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
slide_ids = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

print(f"Found {len(slide_ids)} slides")

# =========================
# organize files per slide
# =========================
for slide_id in slide_ids:
    slide_out_dir = os.path.join(out_root, slide_id)
    os.makedirs(slide_out_dir, exist_ok=True)

    # source files
    img_src = os.path.join(image_dir, f"{slide_id}.jpg")
    expr_src = os.path.join(expr_dir, f"{slide_id}.tsv.gz")
    spot_src = os.path.join(spot_dir, f"{slide_id}_selection.tsv.gz")
    ann_src = os.path.join(ann_dir, f"{slide_id}_labeled_coordinates.tsv")

    # destinations
    img_dst = os.path.join(slide_out_dir, f"{slide_id}.jpg")
    expr_dst = os.path.join(slide_out_dir, f"{slide_id}.tsv.gz")
    spot_dst = os.path.join(slide_out_dir, f"{slide_id}_selection.tsv.gz")
    ann_dst = os.path.join(slide_out_dir, f"{slide_id}_labeled_coordinates.tsv")

    # the image is required
    if os.path.exists(img_src):
        shutil.copy2(img_src, img_dst)
    else:
        print(f"[Skip image] {img_src} not found")
        continue

    # expression
    if os.path.exists(expr_src):
        shutil.copy2(expr_src, expr_dst)
    else:
        print(f"[Warning] expression file not found for {slide_id}: {expr_src}")

    # spot coordinates
    if os.path.exists(spot_src):
        shutil.copy2(spot_src, spot_dst)
    else:
        print(f"[Warning] spot-selection file not found for {slide_id}: {spot_src}")

    # annotation only if it exists
    if os.path.exists(ann_src):
        shutil.copy2(ann_src, ann_dst)
    else:
        print(f"[Info] annotation not found for {slide_id}, skipped")

print("Done.")