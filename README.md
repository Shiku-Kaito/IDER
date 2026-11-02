# IDER: DEG-rank loss for spatial transcriptomics prediction

Code for **IDER**, which trains an ST-Net regressor (an MLP on [CONCH](https://github.com/mahmoodlab/CONCH) patch features) with a
**DEG-rank loss** (a differentiable Mann-Whitney U-score between a cluster and the other clusters) plus a **PCC loss**.
The baselines (ST-Net with a single MSE or PCC loss) are included.

Experiments use the [Her2st](https://zenodo.org/records/3957257) dataset (Andersson et al.) with 4-fold patient-rotation cross-validation.
Test metrics: DEG Spearman and nDCG@50 / 100 / 200 (binary relevance) between the predicted and the measured DEG U-scores of the pathologist-annotated clusters.

## Directory structure

```
supp_script/
├── preprocessing/   download Her2st + CONCH weights, crop patches, extract CONCH features, make CV splits
├── script/          training / evaluation code (IDER_script/, baseline_script/, main.py, run.sh, ...)
├── tutorial/        notebooks: how to use the metrics and the losses on toy data
├── data/            input data written by preprocessing/   (created by the code, ~16 GB)
├── result/          training / evaluation outputs           (created by the code)
└── requirements.txt
```

All paths are resolved relative to `script/paths.py`, never to the current directory.
Set `IDER_DATA_ROOT` / `IDER_RESULT_ROOT` to place `data/` and `result/` elsewhere.

## 1. Environment

```bash
conda create -n IDER python=3.9 -y
conda activate IDER
pip install -r requirements.txt
pip install git+https://github.com/mahmoodlab/CONCH.git    # CONCH is not bundled; install the official package
```

A CUDA GPU is required for feature extraction and recommended for training (about 14 s / epoch on one GPU).

## 2. Data preparation

### What you need
| | where to get it |
|---|---|
| Her2st zip passwords (2 different ones) | published in the README of the original repository, <https://github.com/almaan/her2st> |
| Hugging Face token | CONCH weights are gated. Request access at <https://huggingface.co/MahmoodLab/CONCH>, then create a token at <https://huggingface.co/settings/tokens> |

The passwords unlock `count-matrices.zip` / `images.zip` (password 1) and `meta.zip` / `spot-selections.zip` (password 2).

### Run
```bash
export HER2ST_ZIP_PASSWORD_COUNTS_IMAGES=...   # password 1
export HER2ST_ZIP_PASSWORD_META_SPOTS=...      # password 2
export HF_TOKEN=...                            # Hugging Face token
bash preprocessing/run_preprocessing.sh
```
Pass the secrets through environment variables only; do not write them into files.

| step | script | output (under `data/Her2st/`) |
|---|---|---|
| 0 | `00_download_Her2st_and_CONCH.py` | `zip/`, `org/` (md5-verified downloads, extracted) and `preprocessing/checkpoints/conch/` |
| 1 | `01_align_directory.py` | `preprocessed/<slide>/`: image, expression, spot selection and label files of each slide |
| 2 | `02_crop_patches.py` | `preprocessed/<slide>/*_preprcessed.h5` (patches, expression, pathologist labels) |
| 3 | `03_extract_features.py` | CONCH features added to the `.h5` files |
| 4 | `04_common_genes.py` | `*_preprcessed_common_gene.h5` (genes shared by all slides) |
| 5 | `05_split_cv.py` | `cross_validation/cv_splits.json` (4 folds over patients A-H) |

Each step can also be run on its own (`python preprocessing/0X_*.py`). Existing downloads are skipped.
The file name `_preprcessed.h5` (sic) is kept as it is in the code.

## 3. Training and evaluation

Run from `script/`:

```bash
cd script
# IDER (DEG-rank loss + PCC loss); --is_evaluation 0: train, 1: evaluate
python main.py --module IDER --batch_size 1024 --loss pcc --is_evaluation 0
python main.py --module IDER --batch_size 1024 --loss pcc --is_evaluation 1

# baselines: ST-Net with a single loss (mse or pcc)
python main.py --module baseline --batch_size 1024 --loss mse --is_evaluation 0
python main.py --module baseline --batch_size 1024 --loss mse --is_evaluation 1
```
`bash run.sh` runs all of them (IDER, baseline-mse, baseline-pcc). Training runs the 4 folds one after another, 1000 epochs each.

Main options (see `main.py`):

| option | default | meaning |
|---|---|---|
| `--module` | `IDER` | `IDER` or `baseline` |
| `--loss` | `pcc` | loss of the baseline (`mse` / `pcc`); IDER uses `pcc` for its PCC term |
| `--PCCLoss_lambda` | 1.0 | weight of the PCC loss in IDER: `loss = loss_DEG + lambda * loss_PCC` |
| `--temperature` | 0.01 | temperature of the differentiable U-score |
| `--folds` | `0 1 2 3` | folds to run, e.g. `--folds 0` |
| `--ndcg_topk` | `50 100 200` | k of nDCG@k; the largest k selects the best model (use the same value for training and evaluation) |
| `--clustering_k` | 10 | number of k-means clusters used for training |
| `--num_epochs`, `--lr`, `--batch_size`, `--seed` | 1000, 3e-4, 1024, 0 | |

The best epoch is selected by the validation nDCG@k (largest k) and saved as `best_model_cluster.pkl`; evaluation loads only this model.

### Outputs
```
result/Her2st/eval_top_genes=<G>_TrainCluster_k=10/w_pretrain_lr=3e-4/<mode>_loss=<loss>/
├── model/          best_model_cluster.pkl   (per fold)
├── log_dict/       training logs
├── loss_graph/, acc_graph/    curves
└── test_metrics/result.txt    4-fold mean ± std of DEG Spearman and nDCG@k (written by --is_evaluation 1)
```
`<mode>` is `IDER_PCCLoss_lambda=1.00` for IDER and `ST_net_DEGrank_stop` for the baselines.

## 4. Tutorials

`tutorial/` contains notebooks that run on toy data (no dataset, no GPU):

1. `01_tutorial_metrics.ipynb`: `DEG_ranking` (Mann-Whitney U-scores) and `DEG_metric_func` (DEG Spearman, nDCG@k)
2. `02_tutorial_losses.ipynb`: `Diff_Uscore`, `DEGrank_loss`, the PCC loss and the IDER loss

Open them from the `tutorial/` directory (VS Code with the Jupyter extension, or `jupyter lab`) and select the `IDER` environment as the kernel.

## 5. Code map

| file | content |
|---|---|
| `script/main.py` | argument parsing, training / evaluation entry point |
| `script/get_module.py` | builds the model, optimizer, loss, data loaders for `IDER` / `baseline` |
| `script/IDER_script/train.py`, `script/baseline_script/train.py` | training loops |
| `script/losses.py` | `PearsonCorrCoef_loss`, `MSE_loss`, `Diff_Uscore`, `DEGrank_loss` |
| `script/culc_DEG_metrics.py` | `DEG_ranking` (GPU Mann-Whitney U), `DEG_metric_func` |
| `script/eval.py` | validation / test evaluation |
| `script/dataloader.py`, `script/model.py`, `script/utils.py` | data loading, ST-Net, helpers |

## Notes

* Her2st and CONCH have their own licenses / terms of use; follow them when using the data and the weights.
* In IDER training, 2 clusters are sampled per batch and the DEG-rank loss uses the last of the two.

## Citation

TBD
