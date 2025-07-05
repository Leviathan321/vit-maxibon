import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import LightningModule

from multisim.dataset_utils import load_archive_into_dataset, DrivingDataset
from udacity_gym.extras.model.lane_keeping.vit.vit_model import ViT


# Custom callback to append validation loss to CSV after each validation epoch
class ValLossCSVLogger(pl.Callback):
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        self.csv_path = os.path.join(save_dir, "val_loss.csv")
        # Write CSV header if file does not exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "val_loss"])

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            epoch = trainer.current_epoch
            val_loss_val = val_loss.cpu().item()
            # Append the epoch and loss to the CSV file
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{val_loss_val:.3f}"])

    def plot_val_loss(self):
        epochs = []
        losses = []
        with open(self.csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                losses.append(float(row["val_loss"]))

        plt.figure()
        plt.plot(epochs, losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.save_dir, "val_loss_plot.png")
        plt.savefig(plot_path)
        plt.close()


if __name__ == "__main__":
    archive_path = "/home/lev/Downloads/training_datasets/"
    archive_names = [
        #"beamng-2022_05_31_14_34_55-archive-agent-autopilot-seed-0-episodes-50.npz",
        #"udacity-2022_05_31_12_17_56-archive-agent-autopilot-seed-0-episodes-50.npz",
        "donkey-2022_05_31_12_45_57-archive-agent-autopilot-seed-0-episodes-50.npz"
    ]
    env_name = "udacity"
    # Load dataset splits
    X_train, X_test, y_train, y_test = load_archive_into_dataset(
        archive_path=archive_path,
        archive_names=archive_names,
        seed=0,
        test_split=0.2,
        predict_throttle=False,
        env_name=None
    )

    # Create PyTorch datasets
    train_ds = DrivingDataset(X_train, y_train, is_training=True, env_name=env_name)
    val_ds = DrivingDataset(X_test, y_test, is_training=False, env_name=env_name)

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=32,
                            shuffle=True, num_workers=8, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, 
                            num_workers=8,
                            prefetch_factor=2)

    checkpoint_dir = "./multisim/checkpoints/lane_keeping/vit"

    filename = "vit_{}".format(env_name)
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename + "_{loss:.3f}",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        verbose=True,
    )
    earlystopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=20)
    val_loss_logger = ValLossCSVLogger(save_dir=checkpoint_dir)

    # Trainer setup
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=[0],
        max_epochs=2000,
        callbacks=[checkpoint_callback, earlystopping_callback, val_loss_logger],
    )

    # Model init
    model = ViT()

    print(type(model))
    print(isinstance(model, LightningModule))

    # Train
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
