import torch
from src.ddpm.ddpm import DDPM
import tqdm
from torch.utils.data.dataloader import DataLoader

def train(
    model: DDPM,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader
):
    training_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train(True)
        training_loss = 0
        val_loss = 0
        pbar = tqdm.tqdm(train_dataloader)
        for index, (imgs, labels) in enumerate(pbar):
            optimizer.zero_grad()
            
            imgs = imgs.to(device)
    
            loss = model(imgs)
    
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            pbar.set_description(f"loss for epoch {epoch}: {training_loss / (index + 1):.4f}")
        model.eval()
        with torch.no_grad():
            for (imgs, labels) in val_dataloader:
                imgs = imgs.to(device)
                
                loss = model(imgs)
        
                val_loss += loss.item()
        training_losses.append(training_loss / len(val_dataloader))
        val_losses.append(val_loss / len(val_dataloader))
    return training_losses, val_losses