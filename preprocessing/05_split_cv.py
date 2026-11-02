import os
from collections import defaultdict
import json

DATA_ROOT = os.environ.get("IDER_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))  # supp_script/data

def get_patient_id(slide_name: str) -> str:
    """
    Example:
        A1 -> A
        G2 -> G
    """
    return slide_name[0]


def make_4fold_patient_rotation(data_path, labeled_slide):
    """
    8 patients are split into 4 fixed folds:
        [A, B], [C, D], [E, F], [G, H]

    Rotation rule:
        test = fold i
        val  = fold (i-1) mod 4
        train = remaining 2 folds

    Only labeled_slide are used for val/test.
    All slides from train patients are used for training.

    Returns:
        splits: list of dict
    """
    all_slides = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])

    labeled_slide = sorted(labeled_slide)

    # group all slides by patient
    slides_by_patient = defaultdict(list)
    for slide in all_slides:
        pid = get_patient_id(slide)
        slides_by_patient[pid].append(slide)

    # fixed patient folds
    patient_folds = [
        ["A", "B"],
        ["C", "D"],
        ["E", "F"],
        ["G", "H"],
    ]

    splits = []

    for test_idx in range(4):
        val_idx = (test_idx - 1) % 4
        train_idx = [i for i in range(4) if i not in [test_idx, val_idx]]

        test_patients = patient_folds[test_idx]
        val_patients = patient_folds[val_idx]
        train_patients = patient_folds[train_idx[0]] + patient_folds[train_idx[1]]

        # train: use all slides from train patients
        train_slides = sorted([
            slide
            for p in train_patients
            for slide in slides_by_patient[p]
        ])

        # val/test: use only labeled slides
        val_slides = sorted([
            slide for slide in labeled_slide
            if get_patient_id(slide) in val_patients
        ])

        test_slides = sorted([
            slide for slide in labeled_slide
            if get_patient_id(slide) in test_patients
        ])

        splits.append({
            "fold_id": test_idx + 1,
            "train_patients": train_patients,
            "val_patients": val_patients,
            "test_patients": test_patients,
            "train_slides": train_slides,
            "val_slides": val_slides,
            "test_slides": test_slides,
        })

    return splits


if __name__ == "__main__":
    data_path = os.path.join(DATA_ROOT, "Her2st/preprocessed")
    labeled_slide = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]
    
    output_path = os.path.join(DATA_ROOT, "Her2st/cross_validation/cv_splits.json")

    splits = make_4fold_patient_rotation(data_path, labeled_slide)

    for split in splits:
        print(f"===== Fold {split['fold_id']} =====")
        print("train_patients:", split["train_patients"])
        print("val_patients  :", split["val_patients"])
        print("test_patients :", split["test_patients"])
        print("train_slides  :", split["train_slides"])
        print("val_slides    :", split["val_slides"])
        print("test_slides   :", split["test_slides"])
        print()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"saved: {output_path}")
