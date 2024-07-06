import torchvision
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from dataclasses import dataclass
from src.data.data_common import DataPack
import numpy as np

class CifarTransformation:    
    def __call__(self, tensor: torch.Tensor):
        return (tensor * 127.5 + 127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy()

def get_cifar10_loader_and_transform(
    path_to_dataset: str = "./datasets",
    batch_size: int = 128,
    num_workers: int = 2
) -> DataPack:
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    ])
    transform_to_pil = CifarTransformation()
    train_dataset = torchvision.datasets.CIFAR10(root=path_to_dataset, download=True, transform=transform_to_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = torchvision.datasets.CIFAR10(root=path_to_dataset, download=True, transform=transform_to_tensor, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return DataPack(
        train_dataset=train_dataset,
        train_loader=train_dataloader,
        val_dataset=val_dataset,
        val_loader=val_dataloader,
        transform_to_tensor=transform_to_tensor,
        transform_to_pil=transform_to_pil,
        in_channels=3,
        out_channels=3,
        num_classes=10,
        recommended_steps=(1,2,2,2),
        recommended_attn_step_indexes=[1,2]
    )

