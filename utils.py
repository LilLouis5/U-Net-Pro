import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                train_transform, val_transform,
                batch_size, num_workers, pin_memory=True):

    train_ds = CarvanaDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)

    val_ds = CarvanaDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader


def check_accuracy(loader, model, DEVICE="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=DEVICE)
            y = y.unsqueeze(1).to(device=DEVICE)
            predictions = torch.sigmoid(model(x))
            predictions = (predictions > 0.5).float()
            num_correct += (predictions == y).sum()
            num_pixels += torch.numel(predictions)
            dice_score += (2 * (predictions * y).sum()) / (2 * (predictions * y).sum()+((predictions*y)<1).sum())

    accuracy = round(float(num_correct/num_pixels), 4)
    dice = round(float(dice_score/len(loader)), 4)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels * 100 :.2f}")
    print(f"Dice Score:{dice_score/len(loader)}")

    model.train()

    return accuracy, dice


def save_predictions_as_img(loader, model, epoch_index, folder="saved_images/", DEVICE="cuda"):
    print("=> Saving Predictions")
    model.eval()
    for index, (x, y) in enumerate(loader):
        x = x.to(device=DEVICE)
        with torch.no_grad():
            predictions = torch.sigmoid(model(x))
            predictions = (predictions > 0.5).float()
        torchvision.utils.save_image(predictions, f"{folder}/pred_{epoch_index}_{index}.png")
        #torchvision.utils.save_image(y.unsqeeze(1), f"{folder}/{index}.png")
    model.train()


def plot_loss_acc(train_losses, val_acc, val_dice, num_epoch):
    plt.figure(figsize=(15, 5), dpi=400)

    plt.subplot(1, 3, 1)
    plt.plot(range(num_epoch), train_losses, label="Train Loss", color="blue")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(num_epoch), val_acc, label="Val Acc", color="green")
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title('Acc Over Epochs In Val')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(num_epoch), val_dice, label="Val Dice", color="red")
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.title('Dice Over Epochs In Val')
    plt.legend()

    plt.tight_layout()
    plt.savefig('train_acc_dice_plot.png')
    print("Save Success")









