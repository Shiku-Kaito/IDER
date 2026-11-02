from PIL import Image
from scipy.spatial import cKDTree
import os
import anndata
import pandas as pd
import numpy as np
import h5py

DATA_ROOT = os.environ.get("IDER_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))  # supp_script/data

def extract_patch(image, center, patch_size):
    half_size = int(patch_size // 2)
    x_center, y_center = center
    x_start = max(0, x_center - half_size)
    x_end = min(image.shape[1], x_center + half_size)
    y_start = max(0, y_center - half_size)
    y_end = min(image.shape[0], y_center + half_size)

    patch = image[y_start:y_end, x_start:x_end]
    return Image.fromarray(patch)


def load_annotation_labels(data_path, dataset_label):
    """
    Assign a label to each spot by matching the annotation (x, y) to the
    nearest spot-selection (x, y).
    Returns None if the slide has no annotation.
    """
    ann_path = f"{data_path}/{dataset_label}/{dataset_label}_labeled_coordinates.tsv"
    spot_path = f"{data_path}/{dataset_label}/{dataset_label}_selection.tsv.gz"

    if not os.path.exists(ann_path):
        return None

    ann = pd.read_csv(ann_path, sep="\t")
    spot = pd.read_csv(spot_path, sep="\t")
    spot = spot[spot["selected"] == 1].copy()
    
    # convert to numeric and drop NaN
    ann["x"] = pd.to_numeric(ann["x"], errors="coerce")
    ann["y"] = pd.to_numeric(ann["y"], errors="coerce")
    spot["x"] = pd.to_numeric(spot["x"], errors="coerce")
    spot["y"] = pd.to_numeric(spot["y"], errors="coerce")

    ann = ann.dropna(subset=["x", "y"])
    spot = spot.dropna(subset=["x", "y"])

    # build spot names
    spot["spot"] = spot["x"].astype(str) + "x" + spot["y"].astype(str)

    # nearest-neighbour match: annotation (x, y) -> spot-selection (x, y)
    ann_xy = ann[["x", "y"]].to_numpy()
    spot_xy = spot[["x", "y"]].to_numpy()

    tree = cKDTree(spot_xy)
    dist, idx = tree.query(ann_xy, k=1)

    ann = ann.copy()
    ann["matched_spot"] = spot.iloc[idx]["spot"].to_numpy()
    ann["match_dist"] = dist

    # several annotations may match the same spot; keep only the closest one
    ann = ann.sort_values("match_dist").drop_duplicates("matched_spot", keep="first")

    spot_to_label = ann.set_index("matched_spot")["label"].to_dict()
    return spot_to_label


def create_adata(data_path, dataset_label, labeled):
    expression_matrix = pd.read_csv(
        f"{data_path}/{dataset_label}/{dataset_label}.tsv.gz",
        sep="\t",
        index_col=0,
    )
    spot_coordinates = pd.read_csv(
        f"{data_path}/{dataset_label}/{dataset_label}_selection.tsv.gz",
        sep="\t",
    )
    spot_coordinates["spot"] = (
        spot_coordinates["x"].astype(str) + "x" + spot_coordinates["y"].astype(str)
    )

    common_spots = expression_matrix.index.intersection(spot_coordinates["spot"])
    expression_matrix = expression_matrix.loc[common_spots]
    spot_coordinates = (
        spot_coordinates.set_index("spot").loc[common_spots].reset_index()
    )

    coordinates = spot_coordinates[["pixel_x", "pixel_y"]].values
    tree = cKDTree(coordinates)
    distances, _ = tree.query(coordinates, k=5)
    min_distances = distances[:, 1:].min(axis=1)
    spot_coordinates["neighbor_distance"] = min_distances

    image_path = f"{data_path}/{dataset_label}/{dataset_label}.jpg"
    image = Image.open(image_path)
    image_array = np.array(image)

    image_data = {
        dataset_label: {
            "images": {
                "hires": image_array,
                "lowres": image_array,
            },
            "scalefactors": {
                "spot_diameter_fullres": np.min(min_distances),
                "tissue_hires_scalef": 1,
                "tissue_lowres_scalef": 1,
            },
            "metadata": {
                "chemistry_description": "raw data not provided",
                "software_version": "raw data not provided",
            },
        }
    }

    adata = anndata.AnnData(X=expression_matrix)
    adata.obsm["spatial"] = spot_coordinates[["pixel_x", "pixel_y"]].values
    adata.obs["neighbor_distance"] = spot_coordinates["neighbor_distance"].values
    adata.uns["spatial"] = image_data

    # add annotation labels (annotated slides only)
    if labeled:
        spot_to_label = load_annotation_labels(data_path, dataset_label)
        if spot_to_label is None:
            adata.obs["cancer_label"] = "unlabeled"
        else:
            adata.obs["cancer_label"] = [
                spot_to_label.get(spot_name, "unlabeled") for spot_name in adata.obs.index
            ]

    return adata, image_array


def process_slide(data_path, dataset_label, labeled):
    """Crop one patch per spot and save patches, coordinates, expression (and labels) to <slide>_preprcessed.h5."""
    os.makedirs(f"{data_path}/{dataset_label}/patches", exist_ok=True)

    adata, image_array = create_adata(data_path, dataset_label, labeled)
    adata.var_names_make_unique()

    spot_size = adata.uns["spatial"][dataset_label]["scalefactors"]["spot_diameter_fullres"]
    patch_size = int(round(spot_size))
    spot_names = adata.obs.index
    coordinates = adata.obsm["spatial"]

    # collect patches, coordinates, labels and expression
    patch_imgs = []
    coords_list = []
    cancer_labels = []
    spot_name_list = []
    expression_list = []
    gene_name_list = []

    for spot_name, coord in zip(spot_names, coordinates):
        patch_image = extract_patch(image_array, (int(coord[0]), int(coord[1])), patch_size)

        # PIL Image -> numpy array
        patch_imgs.append(np.array(patch_image))
        coords_list.append([float(coord[0]), float(coord[1])])
        if labeled:
            cancer_labels.append(str(adata.obs.loc[spot_name, "cancer_label"]))
        spot_name_list.append(str(spot_name))

        expression_list.append(adata[spot_name].X)
        gene_name_list.append(list(adata[spot_name].var_names.astype(str)))

    # list -> numpy
    str_dtype = h5py.string_dtype(encoding="utf-8")
    patch_imgs = np.stack(patch_imgs, axis=0)          # (N, H, W, C)
    coords_list = np.array(coords_list, dtype=np.float32)
    spot_name_list = np.array(spot_name_list, dtype=str_dtype)
    expression_array = np.vstack(expression_list)
    gene_name_list = np.array(np.vstack(gene_name_list), dtype=str_dtype)

    h5_save_path = f"{data_path}/{dataset_label}/{dataset_label}_preprcessed.h5"
    with h5py.File(h5_save_path, "w") as f:
        f.create_dataset("img", data=patch_imgs)
        f.create_dataset("coord", data=coords_list)
        if labeled:
            f.create_dataset("cancer_label", data=np.array(cancer_labels, dtype=str_dtype))
        f.create_dataset("spot_name", data=spot_name_list)
        f.create_dataset("expression", data=expression_array)
        f.create_dataset("gene_name", data=gene_name_list)

    print(f"Saved: {h5_save_path}")


if __name__ == "__main__":
    data_path = os.path.join(DATA_ROOT, "Her2st/preprocessed")
    # annotated slides (with cancer_label) and the remaining slides
    labeled_slides = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]
    unlabeled_slides = ["A2", "A3", "A4", "A5", "A6",
                        "B2", "B3", "B4", "B5", "B6",
                        "C2", "C3", "C4", "C5", "C6",
                        "D2", "D3", "D4", "D5", "D6",
                        "E2", "E3",
                        "F2", "F3",
                        "G1", "G3",
                        "H2", "H3"]

    for dataset_label in labeled_slides + unlabeled_slides:
        process_slide(data_path, dataset_label, labeled=dataset_label in labeled_slides)
