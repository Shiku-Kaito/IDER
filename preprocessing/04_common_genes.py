import os
import h5py
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

DATA_ROOT = os.environ.get("IDER_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))  # supp_script/data


# =========================
# settings
# =========================
data_path = os.path.join(DATA_ROOT, "Her2st/preprocessed")

output_h5ad_path = os.path.join(data_path, "all_slides_common_genes_preprocessed.h5ad")
common_gene_txt_path = os.path.join(data_path, "common_genes_all_slides.txt")


# =========================
# helper
# =========================
def decode_to_str_list(arr):
    arr = np.asarray(arr)
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


# =========================
# find all slides
# =========================
dataset_labels = sorted([
    d for d in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, d))
])

print("All slides:")
print(dataset_labels)


# =========================
# read each slide and make AnnData
# =========================
adatas = []
gene_sets = []

for dataset_label in dataset_labels:
    h5_path = os.path.join(
        data_path,
        dataset_label,
        f"{dataset_label}_preprcessed.h5"
    )

    if not os.path.exists(h5_path):
        print(f"skip {dataset_label}: h5 not found")
        continue

    with h5py.File(h5_path, "r") as f:
        if "expression" not in f:
            print(f"skip {dataset_label}: expression not found")
            continue
        if "gene_name" not in f:
            print(f"skip {dataset_label}: gene_name not found")
            continue

        expr = f["expression"][:]  # raw count, shape: (n_spots, n_genes)
        gene_names = decode_to_str_list(f["gene_name"][:][0])

        if "cancer_label" in f:
            cancer_label = decode_to_str_list(f["cancer_label"][:])
        else:
            cancer_label = ["unknown"] * expr.shape[0]

        if "spot_name" in f:
            spot_name = decode_to_str_list(f["spot_name"][:])
        else:
            spot_name = [f"{dataset_label}_{i}" for i in range(expr.shape[0])]
            
        feats = f["feats"][:]
        coord = f['coord'][:]

    patient_id = dataset_label[0]  # A1 -> A

    print(f"\n{dataset_label}")
    print("  expr.shape       :", expr.shape)
    print("  len(gene_names)  :", len(gene_names))
    print("  len(cancer_label):", len(cancer_label))
    print("  len(spot_name)   :", len(spot_name))
    print("  len(feats)   :", len(feats))

    assert expr.shape[1] == len(gene_names), f"{dataset_label}: gene number mismatch"
    assert expr.shape[0] == len(cancer_label), f"{dataset_label}: cancer label number mismatch"
    assert expr.shape[0] == len(spot_name), f"{dataset_label}: spot name number mismatch"

    obs = pd.DataFrame({
        "slide_id": [dataset_label] * expr.shape[0],
        "patient_id": [patient_id] * expr.shape[0],
        "cancer_label": cancer_label,
        "spot_name": spot_name,
    }, index=[f"{dataset_label}_{i}" for i in range(expr.shape[0])])

    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=expr, obs=obs, var=var)
    adata.obsm["feats"] = feats
    adata.obsm["coord"] = coord
    adatas.append(adata)
    gene_sets.append(set(gene_names))


if len(adatas) == 0:
    raise RuntimeError("No valid h5 files were loaded.")


# =========================
# common genes
# =========================
common_genes = sorted(set.intersection(*gene_sets))
print("\nNumber of common genes:", len(common_genes))

if len(common_genes) == 0:
    raise RuntimeError("No common genes found across slides.")

with open(common_gene_txt_path, "w") as f:
    for g in common_genes:
        f.write(g + "\n")
print(f"saved common genes: {common_gene_txt_path}")


# =========================
# align genes and concat
# =========================
adatas = [adata[:, common_genes].copy() for adata in adatas]
adata_all = ad.concat(adatas, axis=0, join="inner", merge="same")

print("\nConcatenated AnnData shape (before preprocessing):", adata_all.shape)


# =========================
# preprocessing (without scVI)
# =========================
# mitochondrial genes, "MT-" for human
adata_all.var["mt"] = adata_all.var_names.str.startswith("MT-")
# ribosomal genes
adata_all.var["ribo"] = adata_all.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata_all.var["hb"] = adata_all.var_names.str.contains("^HB[^(P)]", regex=True)

# filter cells / genes
sc.pp.filter_cells(adata_all, min_genes=200)
sc.pp.filter_genes(adata_all, min_cells=3)

print("AnnData shape after filtering:", adata_all.shape)

# save raw counts
adata_all.layers["counts"] = adata_all.X.copy()

# cast and log1p
adata_all.X = adata_all.X.astype("float32")
sc.pp.log1p(adata_all)

# save whole AnnData
adata_all.write_h5ad(output_h5ad_path)
print(f"saved h5ad: {output_h5ad_path}")


# =========================
# save each slide as new h5
# =========================
for slide_id in sorted(adata_all.obs["slide_id"].unique()):
    print(f"Saving {slide_id}...")

    adata_slide = adata_all[adata_all.obs["slide_id"] == slide_id].copy()

    save_path = os.path.join(
        data_path,
        slide_id,
        f"{slide_id}_preprcessed_common_gene.h5"
    )

    with h5py.File(save_path, "w") as f:
        # expression after preprocessing (log1p)
        f.create_dataset(
            "expression",
            data=np.asarray(adata_slide.X, dtype=np.float32),
        )

        # raw counts
        f.create_dataset(
            "counts",
            data=np.asarray(adata_slide.layers["counts"], dtype=np.float32),
        )

        # gene names
        gene_names = adata_slide.var_names.tolist()
        f.create_dataset(
            "gene_name",
            data=np.array(gene_names, dtype="S"),
        )

        # spot names
        f.create_dataset(
            "spot_name",
            data=np.array(adata_slide.obs["spot_name"].values, dtype="S"),
        )

        # cancer labels
        f.create_dataset(
            "cancer_label",
            data=np.array(adata_slide.obs["cancer_label"].values, dtype="S"),
        )

        # patient id
        f.create_dataset(
            "patient_id",
            data=np.array(adata_slide.obs["patient_id"].values, dtype="S"),
        )

        # slide id
        f.create_dataset(
            "slide_id",
            data=np.array(adata_slide.obs["slide_id"].values, dtype="S"),
        )

        # feats
        f.create_dataset(
            "feats",
            data=np.array(adata_slide.obsm["feats"], dtype="S"),
        )

        # feats
        f.create_dataset(
            "coord",
            data=np.array(adata_slide.obsm["coord"], dtype="S"),
        )
        
        # qc flags for genes
        f.create_dataset(
            "mt",
            data=np.array(adata_slide.var["mt"].values, dtype=np.bool_),
        )
        f.create_dataset(
            "ribo",
            data=np.array(adata_slide.var["ribo"].values, dtype=np.bool_),
        )
        f.create_dataset(
            "hb",
            data=np.array(adata_slide.var["hb"].values, dtype=np.bool_),
        )

    print(f"saved: {save_path}")

print("\nDone.")