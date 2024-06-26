import torchvision
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from dataclasses import dataclass
from src.data.data_common import DataPack

class MNISTTransformation:    
    def __call__(self, tensor):
        return tensor * -1 + 1

def get_mnist_loader_and_transform(
    path_to_dataset: str = "./datasets",
    batch_size: int = 128,
    num_workers: int = 2
) -> DataPack:
    transform_to_tensor = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root=path_to_dataset, download=True, transform=transform_to_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = torchvision.datasets.MNIST(root=path_to_dataset, download=True, transform=transform_to_tensor, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return DataPack(
        train_dataset=train_dataset,
        train_loader=train_dataloader,
        val_dataset=val_dataset,
        val_loader=val_dataloader,
        transform_to_tensor=transform_to_tensor,
        transform_to_pil=MNISTTransformation()
    )

