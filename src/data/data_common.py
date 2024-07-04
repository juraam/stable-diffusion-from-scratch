from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Any

@dataclass
class DataPack:
    train_dataset: Dataset
    train_loader: DataLoader
    val_dataset: Dataset
    val_loader: DataLoader
    transform_to_tensor: Any
    transform_to_pil: Any
    in_channels: int
    out_channels: int
    num_classes: int

