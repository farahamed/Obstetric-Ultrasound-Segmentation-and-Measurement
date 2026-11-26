"""PyTorch Dataset, synchronized ultrasound augmentations, and DataLoader builders."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import COL_FILENAME, COL_HC_MM, PROCESSED_DATA_DIR, RANDOM_SEED


class UltrasoundAugmenter:
    """Synchronized geometric & photometric augmentations for 2D fetal ultrasound segmentation.

    Designed with OpenCV/NumPy so training works cleanly without external augmentation packages.
    Applied exclusively to the training split to improve generalization and mitigate
    trimester/size distribution imbalance.
    """

    def __init__(
        self,
        hflip_p: float = 0.5,
        vflip_p: float = 0.2,
        rotate_limit: float = 15.0,
        scale_limit: float = 0.08,
        shift_limit: float = 0.05,
        brightness_limit: float = 0.15,
        contrast_limit: float = 0.15,
        noise_std: float = 0.02,
        geom_p: float = 0.7,
        photo_p: float = 0.6,
    ) -> None:
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate_limit = rotate_limit
        self.scale_limit = scale_limit
        self.shift_limit = shift_limit
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.noise_std = noise_std
        self.geom_p = geom_p
        self.photo_p = photo_p

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment float32 image in [0, 1] and binary float32 mask in {0, 1}."""
        img = image.copy()
        msk = mask.copy()

        # Horizontal flip
        if rng.random() < self.hflip_p:
            img = np.ascontiguousarray(img[:, ::-1])
            msk = np.ascontiguousarray(msk[:, ::-1])

        # Mild vertical flip
        if rng.random() < self.vflip_p:
            img = np.ascontiguousarray(img[::-1, :])
            msk = np.ascontiguousarray(msk[::-1, :])

        # Combined affine: small rotation, mild scale, small translation
        if rng.random() < self.geom_p:
            h, w = img.shape[:2]
            angle = float(rng.uniform(-self.rotate_limit, self.rotate_limit))
            scale = float(rng.uniform(1.0 - self.scale_limit, 1.0 + self.scale_limit))
            tx = float(rng.uniform(-self.shift_limit, self.shift_limit) * w)
            ty = float(rng.uniform(-self.shift_limit, self.shift_limit) * h)

            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty

            img = cv2.warpAffine(
                img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
            )
            msk = cv2.warpAffine(
                msk, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
            )

        # Photometric jitter (ultrasound gain/contrast variation + subtle sensor noise)
        if rng.random() < self.photo_p:
            active = img > 0.0
            alpha = float(rng.uniform(1.0 - self.contrast_limit, 1.0 + self.contrast_limit))
            beta = float(rng.uniform(-self.brightness_limit, self.brightness_limit))
            img = np.clip(img * alpha + beta, 0.0, 1.0)
            if self.noise_std > 0:
                noise = rng.normal(0.0, self.noise_std, size=img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 1.0)
            img[~active] = 0.0

        msk = (msk > 0.5).astype(np.float32)
        return img.astype(np.float32), msk


class HC18SegmentationDataset(Dataset):
    """PyTorch Dataset loading standardized HC18 ultrasound images and solid binary masks."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        processed_dir: Path = PROCESSED_DATA_DIR,
        image_arm: str = "raw_norm",
        input_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
        seed: int = RANDOM_SEED,
        preload: bool = True,
    ) -> None:
        if image_arm not in ("raw_norm", "despeckled"):
            raise ValueError("image_arm must be 'raw_norm' or 'despeckled'.")

        self.df = metadata_df.reset_index(drop=True).copy()
        self.processed_dir = Path(processed_dir)
        self.image_col = "raw_norm_relpath" if image_arm == "raw_norm" else "despeckled_relpath"
        self.input_size = input_size  # optional (H, W) downscale for fast baseline training
        self.augmenter = UltrasoundAugmenter() if augment else None
        self.rng = np.random.default_rng(seed)
        self.preload = preload

        self._cache: Optional[list[Tuple[np.ndarray, np.ndarray]]] = None
        if preload:
            self._cache = [self._read_pair(i) for i in range(len(self.df))]

    def _read_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        row = self.df.iloc[idx]
        img_path = self.processed_dir / row[self.image_col]
        msk_path = self.processed_dir / row["mask_relpath"]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise IOError(f"Missing processed image/mask pair for {row[COL_FILENAME]}")

        if self.input_size is not None and img.shape[:2] != tuple(self.input_size):
            th, tw = self.input_size
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (tw, th), interpolation=cv2.INTER_NEAREST)

        img_f = (img.astype(np.float32) / 255.0)
        msk_f = (msk > 127).astype(np.float32)
        return img_f, msk_f

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        img_f, msk_f = self._cache[idx] if self._cache is not None else self._read_pair(idx)

        if self.augmenter is not None:
            img_f, msk_f = self.augmenter(img_f, msk_f, self.rng)

        img_t = torch.from_numpy(img_f).unsqueeze(0)  # (1, H, W)
        msk_t = torch.from_numpy(msk_f).unsqueeze(0)  # (1, H, W)

        row = self.df.iloc[idx]
        return {
            "image": img_t,
            "mask": msk_t,
            "filename": str(row[COL_FILENAME]),
            "hc_mm": float(row[COL_HC_MM]),
            "pixel_size_x_mm": float(row["pixel_size_x_mm"]),
            "pixel_size_y_mm": float(row["pixel_size_y_mm"]),
        }


def compute_train_pos_weight(
    train_df: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> float:
    """Compute background-to-foreground pixel ratio strictly from the training split."""
    fg_total = 0
    px_total = 0
    for rel in train_df["mask_relpath"]:
        msk = cv2.imread(str(processed_dir / rel), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise IOError(f"Missing mask: {processed_dir / rel}")
        fg_total += int((msk > 127).sum())
        px_total += int(msk.size)
    bg_total = px_total - fg_total
    return float(bg_total / max(fg_total, 1))


def _build_stratum_sampler(
    train_df: pd.DataFrame,
    power: float = 0.5,
    seed: int = RANDOM_SEED,
) -> WeightedRandomSampler:
    """Build a tempered inverse-frequency sampler across HC size strata.

    Uses ``weight ~ (1 / count)^power`` (default sqrt tempering, power=0.5) so
    under-represented 1st and 3rd trimester scans are seen more frequently without
    overfitting the smallest bins.
    """
    counts = train_df["hc_bin"].value_counts().to_dict()
    sample_weights = [1.0 / (counts[b] ** power) for b in train_df["hc_bin"]]
    gen = torch.Generator()
    gen.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_df),
        replacement=True,
        generator=gen,
    )


def build_dataloaders(
    image_arm: str = "raw_norm",
    batch_size: int = 16,
    input_size: Optional[Tuple[int, int]] = None,
    use_stratum_sampler: bool = True,
    processed_dir: Path = PROCESSED_DATA_DIR,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], Dict[str, pd.DataFrame], float]:
    """Load processed metadata and return Train/Val/Test DataLoaders and training pos_weight."""
    meta_path = Path(processed_dir) / "processed_metadata.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Processed metadata not found at {meta_path}. Run preprocessing first."
        )

    full_df = pd.read_csv(meta_path)
    split_dfs = {
        s: full_df[full_df["split"] == s].reset_index(drop=True)
        for s in ("train", "val", "test")
    }

    pos_weight = compute_train_pos_weight(split_dfs["train"], processed_dir=processed_dir)

    train_ds = HC18SegmentationDataset(
        split_dfs["train"],
        processed_dir=processed_dir,
        image_arm=image_arm,
        input_size=input_size,
        augment=True,
        seed=seed,
        preload=True,
    )
    val_ds = HC18SegmentationDataset(
        split_dfs["val"],
        processed_dir=processed_dir,
        image_arm=image_arm,
        input_size=input_size,
        augment=False,
        seed=seed + 1,
        preload=True,
    )
    test_ds = HC18SegmentationDataset(
        split_dfs["test"],
        processed_dir=processed_dir,
        image_arm=image_arm,
        input_size=input_size,
        augment=False,
        seed=seed + 2,
        preload=True,
    )

    if use_stratum_sampler:
        train_sampler = _build_stratum_sampler(split_dfs["train"], power=0.5, seed=seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            drop_last=False,
        )
    else:
        gen = torch.Generator()
        gen.manual_seed(seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=gen,
            num_workers=num_workers,
            drop_last=False,
        )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    return loaders, split_dfs, pos_weight
