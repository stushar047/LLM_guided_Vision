from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class LiveTripletDataset(Dataset):
    """JSONL manifest dataset with image, instruction, answer, and negatives."""

    def __init__(self, manifest: str | Path, image_transform=None) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.image_transform = image_transform
        with self.manifest.open("r", encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        for index, row in enumerate(self.rows):
            missing = {"image", "instruction", "answer"} - row.keys()
            if missing:
                raise ValueError(f"Row {index} is missing required fields: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError("Install Pillow to load images: pip install pillow") from error
        image_path = (self.root / row["image"]).resolve()
        image = Image.open(image_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        return {
            "image": image,
            "instruction": str(row["instruction"]),
            "answer": str(row["answer"]),
            "negatives": [str(text) for text in row.get("negatives", [])],
        }


def collate_triplets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate rows while preserving variable-length negative text lists."""
    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch is required to collate batches") from error
    return {
        "images": torch.stack([row["image"] for row in rows]),
        "instructions": [row["instruction"] for row in rows],
        "answers": [row["answer"] for row in rows],
        "negatives": [row["negatives"] for row in rows],
    }
