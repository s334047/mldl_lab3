import torch
from torch import nn
from torch.utils.data import DataLoader

import wandb

from dataset.my_dataset import get_tiny_imagenet_datasets
from models.customnet import CustomNet
from utils.train_loop import train
from utils.validate_loop import validate

def main():

    # WANDB SETUP
    wandb.init(project="mldl_lab3")
    config = wandb.config
    config.learning_rate = 0.001

    # DATASET SETUP
    train_set, val_set = get_tiny_imagenet_datasets()
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)

    #MODEL SETUP: COPIA E INCOLLA DA LAB 2
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_acc = 0

    # Run the training process for {num_epochs} epochs
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion)

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)

        # LOGGING ON WANDB
        wandb.log({
            "epoch": epoch,
            "val_accuracy": val_accuracy
        })

    print(f'Best validation accuracy: {best_acc:.2f}%')

    #SALVIAMO IL MODELLO
    save_path = "checkpoints/best_model.pth"
    torch.save(model.state_dict(), save_path)
    print("Saved model to checkpoints/best_model.pth")

if __name__ == "__main__":
    main()
