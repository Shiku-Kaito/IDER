import argparse
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

from conch.open_clip_custom import create_model_from_pretrained

DATA_ROOT = os.environ.get("IDER_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))  # supp_script/data

ALL_SLIDES = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1",
              "A2", "A3", "A4", "A5", "A6",
              "B2", "B3", "B4", "B5", "B6",
              "C2", "C3", "C4", "C5", "C6",
              "D2", "D3", "D4", "D5", "D6",
              "E2", "E3",
              "F2", "F3",
              "G1", "G3",
              "H2", "H3"]


def image_transform(image_size=512):
    return Compose([
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),
    ])


class PatchH5Dataset(Dataset):
    def __init__(self, h5_path, transform):
        self.h5_path = h5_path
        self.transform = transform

        with h5py.File(self.h5_path, "r") as f:
            self.length = len(f["img"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            patch = f["img"][idx]  # (H, W, C), expected uint8

        patch = Image.fromarray(patch)
        patch = self.transform(patch)
        return {"patch": patch, "idx": idx}


def extract_and_save_features_for_slide(args, model, preprocess, slide_id):
    h5_path = os.path.join(
        args.data_path,
        slide_id,
        f"{slide_id}_preprcessed.h5"
    )

    if not os.path.exists(h5_path):
        print(f"skip {slide_id}: not found -> {h5_path}")
        return

    dataset = PatchH5Dataset(h5_path=h5_path, transform=preprocess)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if "cuda" in args.device else False,
    )

    model.eval()
    feat_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"extract {slide_id}"):
            patch = batch["patch"].to(args.device, non_blocking=True)

            feat = model.encode_image(
                patch,
                proj_contrast=False,
                normalize=False
            )
            feat_list.append(feat.cpu().numpy())

    feat_array = np.concatenate(feat_list, axis=0).astype(np.float32)

    with h5py.File(h5_path, "a") as f:
        if "feats" in f:
            del f["feats"]
        f.create_dataset("feats", data=feat_array)

    print(f"saved feats to: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        print(f"keys: {list(f.keys())}")


def main(args):
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        device=args.device,
        force_image_size=512,
        checkpoint_path=args.checkpoint_path
    )

    if args.slide_ids is None or len(args.slide_ids) == 0:
        slide_ids = sorted([
            d for d in os.listdir(args.data_path)
            if os.path.isdir(os.path.join(args.data_path, d))
        ])
    else:
        slide_ids = args.slide_ids

    print("target slides:", slide_ids)

    for slide_id in slide_ids:
        extract_and_save_features_for_slide(
            args=args,
            model=model,
            preprocess=preprocess,
            slide_id=slide_id
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(DATA_ROOT, "Her2st/preprocessed"),
        help="Root dir containing A1/, B1/, ..."
    )
    parser.add_argument(
        "--slide_ids",
        nargs="*",
        default=ALL_SLIDES,
        help="Target slide IDs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/conch/pytorch_model.bin")
    )

    args = parser.parse_args()
    main(args)