import torch
import torchvision

from load_data import Data
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("Saving checkpoint...")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_dataset = Data(
        image_directory=train_dir,
        mask_directory=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_dataset = Data(
        image_directory=val_dir,
        mask_directory=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))

            #ako pixel prediction e pogolem od 0.5 -> 1, ako e pomal -> 0
            #ova e za binarno klasificiranje

            preds = (preds > 0.5).float()
            num_correct += preds.eq(y).sum()
            num_pixels += torch.numel(preds)


    print(f"Got {num_correct} correct out of {num_pixels} pixels with accuracy {num_correct / num_pixels*100:.2f}")

    model.train()

def save_predictions_as_images(loader, model, folder="saved_images/", device="cuda"):

    model.eval()

    for idx, (image, mask) in enumerate(loader):
        image = image.to(device=device)

        with torch.no_grad():
            preds = torch.sigmoid(model(image))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, f"{folder}pred_{idx}.png")
        torchvision.utils.save_image(mask.unsqueeze(1), f"{folder}mask_{idx}.png")

    model.train()

