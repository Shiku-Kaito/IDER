# Run from anywhere. Data is written to supp_script/data (override with IDER_DATA_ROOT).
# Requirements: pip install git+https://github.com/mahmoodlab/CONCH.git pyzipper huggingface_hub
set -e
cd "$(dirname "$0")"
HER2ST_ZIP_PASSWORD_COUNTS_IMAGES=${HER2ST_ZIP_PASSWORD_COUNTS_IMAGES:?see 00_download_Her2st_and_CONCH.py} \
HER2ST_ZIP_PASSWORD_META_SPOTS=${HER2ST_ZIP_PASSWORD_META_SPOTS:?see 00_download_Her2st_and_CONCH.py} \
HF_TOKEN=${HF_TOKEN:?set HF_TOKEN (CONCH is gated)} python 00_download_Her2st_and_CONCH.py
python 01_align_directory.py
python 02_crop_patches.py
python 03_extract_features.py
python 04_common_genes.py
python 05_split_cv.py
