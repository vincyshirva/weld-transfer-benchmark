#!/usr/bin/env python3
"""
YOLO-format detection dataset for the torchvision family.

Faster R-CNN is the one family ultralytics cannot train, so it needs its own
loader. This one reads the exact same tree the other three families read, so
the splits, images and labels are byte identical across the whole grid. Only
the model and optimizer differ, which is the point of a third family.

Two things here are load bearing and easy to get wrong.

  Letterbox, not stretch. Images are resized preserving aspect ratio and
  padded, matching what ultralytics does. LoHi images are about 4.8:1, so a
  stretch would distort them into a different domain from what the YOLO
  families see, and the transfer numbers would not be comparable.

  Background images are kept. twoclass has 58 images with no boxes and
  lohi_weld has 54. They are real negatives and dropping them would inflate
  every false alarm metric. torchvision accepts an empty (0, 4) box tensor.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def letterbox(image: Image.Image, size: int) -> tuple[Image.Image, float, int, int]:
    """Resize preserving aspect ratio, pad to square. Returns scale and pads."""
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (px, py))
    return canvas, scale, px, py


class YoloDetection(Dataset):
    """One split of one dataset, in torchvision detection format.

    Labels come out 1 based, because torchvision reserves class 0 for
    background. evaluate_cell.py subtracts the one again when it maps
    predictions back onto the source's own class list.
    """

    def __init__(self, split_dir: Path, imgsz: int, train: bool):
        self.dir = Path(split_dir)
        self.imgsz = imgsz
        self.train = train
        self.images = sorted(p for p in (self.dir / "images").iterdir()
                             if p.suffix.lower() in IMG_EXT)
        if not self.images:
            raise FileNotFoundError(f"no images under {self.dir / 'images'}")

    def __len__(self) -> int:
        return len(self.images)

    def _labels(self, stem: str) -> np.ndarray:
        path = self.dir / "labels" / f"{stem}.txt"
        if not path.is_file():
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(v) for v in parts[:5]])
        return (np.array(rows, dtype=np.float32) if rows
                else np.zeros((0, 5), dtype=np.float32))

    def __getitem__(self, i: int):
        path = self.images[i]
        with Image.open(path) as im:
            image = im.convert("RGB")
            w0, h0 = image.size
        label = self._labels(path.stem)

        image, scale, px, py = letterbox(image, self.imgsz)

        # normalized cx cy w h  ->  absolute xyxy in the letterboxed frame
        if len(label):
            cx, cy, bw, bh = (label[:, 1] * w0, label[:, 2] * h0,
                              label[:, 3] * w0, label[:, 4] * h0)
            x1 = (cx - bw / 2) * scale + px
            y1 = (cy - bh / 2) * scale + py
            x2 = (cx + bw / 2) * scale + px
            y2 = (cy + bh / 2) * scale + py
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            classes = label[:, 0].astype(np.int64) + 1   # 0 is background
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int64)

        if self.train and random.random() < 0.5:      # matches fliplr 0.5
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes):
                boxes[:, [0, 2]] = self.imgsz - boxes[:, [2, 0]]

        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        if self.train:
            # Rough stand-in for the HSV jitter in the ultralytics recipe.
            # Exact parity across families is impossible and is stated in the
            # paper rather than pretended away.
            tensor = (tensor * random.uniform(0.6, 1.4)).clamp_(0, 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = torch.as_tensor(classes, dtype=torch.int64)

        # Faster R-CNN raises on degenerate boxes, and letterbox rounding can
        # collapse a very thin box. Drop them rather than crash 20 hours in.
        if len(boxes):
            keep = (boxes[:, 2] > boxes[:, 0] + 1e-3) & (boxes[:, 3] > boxes[:, 1] + 1e-3)
            boxes, classes = boxes[keep], classes[keep]

        target = {
            "boxes": boxes,
            "labels": classes,
            "image_id": torch.tensor([i]),
            "orig_size": torch.tensor([h0, w0]),
            "letterbox": torch.tensor([scale, px, py], dtype=torch.float32),
        }
        return tensor, target


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
