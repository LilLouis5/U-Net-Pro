import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model_pro import Unet
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_img,
    plot_loss_acc,
)
import numpy as np
import random

# Hyperparameters
LEARNING_RATE = 1e-8
BATCH_SIZE = 32
NUM_EPOCH = 2
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR = "./dataset/train_img_dir/"
TRAIN_MASK_DIR = "./dataset/train_mask_dir"
VAL_IMG_DIR = "./dataset/val_img_dir/"
VAL_MASK_DIR = "./dataset/val_mask_dir/"

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

train_losses = []
val_acc = []
val_dice = []

# 设置随机种子
seed = random.randint(1, 100)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果你使用多个GPU
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_fn(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)
    total_loss = 0.0
    for index, (data, target) in enumerate(loop):
        data = data.to(device=DEVICE)
        target = target.unsqueeze(1).float().to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predicts = model(data)
            loss = loss_fn(predicts, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],)

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],)

    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
                                           train_transform, val_transform,
                                           BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

    model = Unet(in_channel=3, out_channel=1).to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL is True:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()

    for index in range(NUM_EPOCH):
        print("index:", index)
        train_loss = train_fn(train_loader, model, loss_fn, optimizer, scaler)
        train_losses.append(train_loss)

        accuracy, dice = check_accuracy(val_loader, model, DEVICE)
        val_acc.append(accuracy)
        val_dice.append(dice)

        #chekpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        #save_checkpoint(chekpoint)

        if index % 10 == 0:
            save_predictions_as_img(val_loader, model, epoch_index=index, folder="saved_images/", DEVICE=DEVICE)

    plot_loss_acc(train_losses, val_acc, val_dice, NUM_EPOCH)

    np.save("Unet_pro_train_losses", train_losses)
    np.save("Unet_pro_val_acc", val_acc)
    np.save("Unet_pro_val_dice", val_dice)


if __name__ == "__main__":
    main()
