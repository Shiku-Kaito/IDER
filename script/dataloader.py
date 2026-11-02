import os
import json
from collections import defaultdict

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import umap
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

from culc_DEG_metrics import DEG_ranking

# Pathologist annotation -> integer label. The annotation defines the clusters of the test slides.
class_to_idx = {
    'adipose tissue': 0,
    'cancer in situ': 1,
    'connective tissue': 2,
    'immune infiltrate': 3,
    'invasive cancer': 4,
    'breast glands': 5,
    'undetermined': 6
}

# Arrays loaded from <slide>_preprcessed_common_gene.h5 and concatenated along the spot axis.
SPOT_KEYS = ["cancer_label", "expression", "feats"]


def get_patient_id(slide_name: str) -> str:
    """Example: C1 -> C"""
    return slide_name[0]


def apply_fixed_label_mapping(cancer_label, class_to_idx):
    # bytes -> string -> integer label
    labels = [x.decode("utf-8") for x in cancer_label]
    return np.array([class_to_idx[x] for x in labels], dtype=np.int64)


def check_gene_name(gene_names, message):
    """All slides must share the same gene_name. Returns that gene_name."""
    first = np.asarray(gene_names[0])
    for gene_name in gene_names[1:]:
        if not np.array_equal(first, np.asarray(gene_name)):
            raise ValueError(message)
    return first


def concat_patient_data(slide_dict):
    """Concatenate the slides of each patient (e.g. C1, C2, ... -> C) along the spot axis."""
    grouped = defaultdict(list)
    for slide_key, data in slide_dict.items():
        grouped[get_patient_id(slide_key)].append((slide_key, data))

    patient_data = {}
    for patient_id, items in grouped.items():
        items = sorted(items, key=lambda x: x[0])   # C1, C2, ...

        merged = {}
        merged["gene_name"] = check_gene_name([data["gene_name"] for _, data in items], f"gene_name mismatch in patient {patient_id}")
        for key in SPOT_KEYS:
            merged[key] = np.concatenate([np.asarray(data[key]) for _, data in items], axis=0)

        # patient_id of every spot
        merged["patient_id"] = np.array([patient_id] * merged["expression"].shape[0])

        patient_data[patient_id] = merged

    return patient_data


def concat_all_patients(patient_data):
    patient_ids = sorted(patient_data.keys())

    merged = {}
    merged["gene_name"] = check_gene_name([patient_data[pid]["gene_name"] for pid in patient_ids], "gene_name mismatch")
    for key in SPOT_KEYS + ["patient_id", "cluster_ids"]:
        merged[key] = np.concatenate([np.asarray(patient_data[pid][key]) for pid in patient_ids], axis=0)

    return merged


def load_slides(slide_list, base_dir):
    slide_data = {}
    for slide_id in slide_list:
        file_path = os.path.join(base_dir, slide_id, f"{slide_id}_preprcessed_common_gene.h5")

        if not os.path.exists(file_path):
            print(f"[WARNING] Not found: {file_path}")
            continue

        with h5py.File(file_path, "r") as f:
            slide_data[slide_id] = {key: f[key][:] for key in SPOT_KEYS + ["gene_name"]}
        print(f"Loaded: {slide_id}")

    return slide_data


def clustering(args, feat_list, n_clusters, save_path):
    """K-means clustering of the features; saves a UMAP plot colored by cluster."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.random_state)
    labels = kmeans.fit_predict(feat_list)

    reducer = umap.UMAP(random_state=args.random_state)
    embedding = reducer.fit_transform(feat_list)

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=30)

    # dummy scatter for the legend
    for k in range(n_clusters):
        plt.scatter([], [], color=cmap(k), label=f"Cluster {k}")

    plt.title("UMAP of Features Colored by K-means Clusters")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="Cluster ID", loc="best", markerscale=1.2, frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig("%sk=%d_clustering.png" % (save_path, n_clusters), dpi=400, bbox_inches="tight")
    plt.close()
    return labels


class RandomPatientPatchDataset(Dataset):
    """
    Training set. Each item is a batch of `n_patches` spots sampled from one randomly chosen patient,
    so that the DEG ranking of that patient can be computed within the batch.

    patient_data: {patient_id: {"feats", "expression", "cluster_ids", ...}}
    n_patches: number of spots sampled from one patient per __getitem__
    steps_per_epoch: number of steps per epoch (returned by __len__)
    """
    def __init__(self, patient_data, n_patches=128, steps_per_epoch=1000, seed=None):
        self.patient_ids = sorted(list(patient_data.keys()))
        self.n_patches = n_patches
        self.steps_per_epoch = steps_per_epoch
        self.rng = np.random.default_rng(seed)

        self.processed = {}
        for pid in self.patient_ids:
            pdata = patient_data[pid]
            self.processed[pid] = {
                "feats": np.asarray(pdata["feats"]).astype(np.float32),   # stored as strings in the h5 file
                "expression": np.asarray(pdata["expression"]).astype(np.float32),
                "cluster_ids": np.asarray(pdata["cluster_ids"]).astype(np.int64),
            }

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        # pick one patient at random
        patient_id = self.rng.choice(self.patient_ids)
        pdata = self.processed[patient_id]

        n_total = pdata["feats"].shape[0]
        replace = n_total < self.n_patches
        patch_indices = self.rng.choice(n_total, size=self.n_patches, replace=replace)

        return {
            "patch": torch.from_numpy(pdata["feats"][patch_indices]),    # [n_patches, 512]
            "patient_id": [patient_id] * self.n_patches,
            "st": torch.from_numpy(pdata["expression"][patch_indices]),
            "cluster_id": torch.from_numpy(pdata["cluster_ids"][patch_indices]),
        }


class EvalDataset(Dataset):
    """Validation / test set: all spots of all patients."""
    def __init__(self, args, data_dict):
        np.random.seed(args.seed)
        self.patch = data_dict['feats']
        self.patient_id = np.asarray(data_dict["patient_id"])
        self.cluster_id = np.asarray(data_dict["cluster_ids"]).astype(np.int64)
        self.st = np.asarray(data_dict["expression"]).astype(np.float32)
        self.len = len(self.patch)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            "patch": torch.from_numpy(self.patch[idx].astype(np.float32)),   # [512]
            "patient_id": self.patient_id[idx],
            "st": torch.tensor(self.st[idx]),
            "cluster_id": self.cluster_id[idx],
        }


def collate_fn_custom_eval(batch):
    return {
        "patch": torch.stack([b["patch"] for b in batch], dim=0),
        "st": torch.stack([b["st"] for b in batch], dim=0),
        "patient_id": [b["patient_id"] for b in batch],
        "cluster_id": torch.tensor([b["cluster_id"] for b in batch]),
    }


def compute_DEG(args, patient_data):
    """Mann-Whitney U based DEG ranking of every patient, using its cluster_ids."""
    DEGRank_dict, Uscore_dict = {}, {}
    for patient_id in patient_data.keys():
        DEGRank_dict[patient_id], Uscore_dict[patient_id] = DEG_ranking(
            X=torch.tensor(patient_data[patient_id]['expression']),
            cluster_id=patient_data[patient_id]["cluster_ids"],
            gene_chunk=50,
            device=args.device,
        )
    return DEGRank_dict, Uscore_dict


def load_dataset(args):
    ######### load data #######
    with open('%s/data/Her2st/cross_validation/cv_splits.json' % args.input_path, "r") as f:
        cv_splits = json.load(f)

    train_patients, val_patients, test_patients = cv_splits[args.fold]['train_patients'], cv_splits[args.fold]['val_patients'], cv_splits[args.fold]['test_patients']

    # get all slides
    data_path = "%s/data/Her2st/preprocessed" % args.input_path
    all_slides = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])

    # train / val: all slides of the patients, test: only the annotated slides
    train_slides = sorted([slide for slide in all_slides if get_patient_id(slide) in train_patients])
    val_slides = sorted([slide for slide in all_slides if get_patient_id(slide) in val_patients])
    test_slides = sorted([slide for slide in args.labeled_slides if get_patient_id(slide) in test_patients])

    print("train_slides:", train_slides)
    print("val_slides  :", val_slides)
    print("test_slides :", test_slides)

    train_patient_data = concat_patient_data(load_slides(train_slides, base_dir=data_path))
    val_patient_data = concat_patient_data(load_slides(val_slides, base_dir=data_path))
    test_patient_data = concat_patient_data(load_slides(test_slides, base_dir=data_path))

    first_patient = list(train_patient_data.keys())[0]
    all_gene_num = train_patient_data[first_patient]['gene_name'].shape[0]
    eval_gene_name = np.array([x.decode("utf-8") for x in train_patient_data[first_patient]['gene_name']])

    # clusters: k-means on the features (train / val), pathologist annotation (test)
    for patient_data in (train_patient_data, val_patient_data):
        for patient_id in patient_data.keys():
            save_path = '%s/%s/eval_top_genes=%d_TrainCluster_k=%d/Train_patient=%s_' % (args.output_path, args.dataset, all_gene_num, args.clustering_k, patient_id)
            patient_data[patient_id]["cluster_ids"] = clustering(args, patient_data[patient_id]['feats'], args.clustering_k, save_path)
    for patient_id in test_patient_data.keys():
        test_patient_data[patient_id]["cluster_ids"] = apply_fixed_label_mapping(test_patient_data[patient_id]['cancer_label'], class_to_idx)

    # Obtain DEGs
    train_DEGRank_dict, train_DEGscore_dict = compute_DEG(args, train_patient_data)
    val_DEGRank_dict, val_DEGscore_dict = compute_DEG(args, val_patient_data)
    test_DEGRank_dict, test_DEGscore_dict = compute_DEG(args, test_patient_data)

    train_dataset = RandomPatientPatchDataset(
        patient_data=train_patient_data,
        n_patches=args.batch_size,
        steps_per_epoch=20,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,   # 1 step = one batch sampled from one patient
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )

    val_dataset = EvalDataset(args=args, data_dict=concat_all_patients(val_patient_data))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_custom_eval)
    test_dataset = EvalDataset(args=args, data_dict=concat_all_patients(test_patient_data))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_custom_eval)

    return train_loader, val_loader, test_loader, {"eval": eval_gene_name}, all_gene_num, {"train": train_DEGRank_dict, "val": val_DEGRank_dict, "test": test_DEGRank_dict}, {"train": train_DEGscore_dict, "val": val_DEGscore_dict, "test": test_DEGscore_dict}
