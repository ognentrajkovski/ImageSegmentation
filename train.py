import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_images,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()  # Use AMP only if CUDA available
BATCH_SIZE = 8
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # original 1280 / 8
IMAGE_WIDTH = 240  # original 1918 / 8
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMAGE_DIRECTORY = "data/train_images/"
TRAIN_MASK_DIRECTORY = "data/train_masks/"
VAL_IMAGE_DIRECTORY = "data/validation_images/"
VAL_MASK_DIRECTORY = "data/validation_masks/"

def train_function(loader, model, optimizer, loss_function, scaler=None):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)  # for binary cross-entropy

        # forward
        if USE_AMP:
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_function(predictions, targets)
        else:
            predictions = model(data)
            loss = loss_function(predictions, targets)

        # backward
        optimizer.zero_grad()
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loop.set_postfix(loss=loss.item())

def main():
    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_ch=3, out_ch=1).to(DEVICE)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMAGE_DIRECTORY,
        TRAIN_MASK_DIRECTORY,
        VAL_IMAGE_DIRECTORY,
        VAL_MASK_DIRECTORY,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint.pth.tar"), model)

    for epoch in range(NUM_EPOCHS):
        train_function(train_loader, model, optimizer, loss_function, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)

        save_predictions_as_images(val_loader, model, folder="saved_images/", device=DEVICE)


if __name__ == "__main__":
    main()
