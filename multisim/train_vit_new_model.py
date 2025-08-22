import argparse
import json
import os
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import LightningModule
from datetime import datetime
from multisim.dataset_lazy import DrivingDatasetLazy, split_data
from multisim.dataset_utils import load_archive_into_dataset, DrivingDataset
from udacity_gym.extras.model.lane_keeping.vit.vit_model import ViT
from multisim.plot import plot_steering_distribution

# Custom callback to append validation loss to CSV after each validation epoch
class ValLossCSVLogger(pl.Callback):
    def __init__(self, save_dir: str, env_name: str):
        super().__init__()
        self.save_dir = save_dir
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        self.csv_path = os.path.join(save_dir, f"val_loss_{env_name}.csv")
        # Write CSV header if file does not exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "val_loss"])
        self.env_name = env_name

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            epoch = trainer.current_epoch
            val_loss_val = val_loss.cpu().item()
            # Append the epoch and loss to the CSV file
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{val_loss_val:.6f}"])

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
        plot_path = os.path.join(self.save_dir, f"val_loss_{self.env_name}_plot.png")
        plt.savefig(plot_path)
        plt.close()

def get_distribution(dataset):
    from collections import Counter

    # Assuming your dataset has a pandas DataFrame self.metadata with 'image_name' column
    env_counts = Counter()

    for idx in range(len(dataset)):
        image_name = str(dataset.metadata.iloc[idx]["image_name"]).lower()
        if "beamng" in image_name:
            env_counts["beamng"] += 1
        elif "donkey" in image_name:
            env_counts["donkey"] += 1
        elif "udacity" in image_name:
            env_counts["udacity"] += 1
        else:
            env_counts["unknown"] += 1

    print("Counts via __getitem__:")
    for env, count in env_counts.items():
        print(f"{env}: {count}")
    
    return env_counts

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train ViT model for lane keeping.")
    parser.add_argument(
        "--lr_rate_tune",
        dest='lr_rate_tune',
        type=float,
        required=False,
        default=0.0002,
        help="Learning rate to use for fine-tuning the model."
    )
    
    parser.add_argument(
        "--suffix_folder_name",
        dest="suffix_folder_name",
        type=str,
        required=False,
        default=None,
        help="Optional suffix to append to the output folder name."
    )
    args = parser.parse_args()

    lr_rate_tune = args.lr_rate_tune
    suffix_folder_name = args.suffix_folder_name
    print(f"[INFO] Using learning rate for tuning: {lr_rate_tune}")

    archive_path = rf"C:\Users\levia\Downloads\raw"
    
    resume_checkpoint_path = None #rf"..\vit-maxibon\multisim\checkpoints_16-08-2025_14-57_lr-1.5e-05\lane_keeping\vit\vit_mixed.ckpt"

    #for lr_rate_tune in [0.0001,0.00005,0.00002]:
    #lr_rate_tune = 0.00005 # original is 0.0002

    archive_names = [
        "beamng-2022_05_31_14_34_55-archive-agent-autopilot-seed-0-episodes-50.npz",
        "udacity-2022_05_31_12_17_56-archive-agent-autopilot-seed-0-episodes-50.npz",
        "donkey-2022_05_31_12_45_57-archive-agent-autopilot-seed-0-episodes-50.npz"
    ]
    # Extract simulator names (first token before the first '-')
    sim_names = {name.split("-")[0].lower() for name in archive_names}

    # Determine environment name
    if len(sim_names) == 1:
        env_name = sim_names.pop()
    elif len(sim_names) > 1:
        env_name = "mixed"
    else:
        raise ValueError("No valid (uncommented) archive names found.")

    print("env_name:", env_name)
    folder_paths = [
        archive_path,
    # maxibon - seed2000 - 50
        # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000_50/beamng_2025-08-08_19-23-23",
        # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000_50/donkey_2025-08-08_18-36-14",
        # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000_50/udacity_2025-08-08_19-00-00",

        # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/beamng_2025-07-31_22-59-29",
        # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/donkey_2025-07-31_22-47-17",
        # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/udacity_2025-08-02_01-55-41",

        # maxibon - seed2000 - 25
        #"/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000/beamng_2025-07-30_14-17-01",
        #"/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000/donkey_2025-07-30_14-04-44",
        #"/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000/udacity_2025-07-30_18-13-59",

        # maxibon - seed3000 - 25
        #"/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/beamng_2025-07-31_22-59-29/",
        #"/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/donkey_2025-07-31_22-47-17/",
        #"/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/udacity_2025-08-02_01-55-41"

        #"/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/20-07-2025",
        #"/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/18-07-2025/",
        # "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/20-07-2025_2000/",
        # "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/21-07-2025_2000/",
        # "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/23-07-2025_2000", # udacity
        # "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/24-07-2025_2000", # udacity
        # "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/bng_recording_25-07-25_2000/25-07-2025_2000" # udacity
        
        # finetuning beamng
        # rf"C://Users//levia//Documents//testing//Multi-Simulation///vit-recordings-maxi/3000_tracks25/beamng_2025-07-31_22-59-29"]
        # finetuning donkey
        # rf"C:\Users\levia\Documents\testing\Multi-Simulation\vit-recordings-maxi\3000_tracks25\donkey_2025-07-31_22-47-17"
        rf"..\vit-recordings-maxi\2000_tracks50_samefps\beamng_2000_50_2025-08-21_00-36-03",
        rf"..\vit-recordings-maxi\2000_tracks50_samefps\donkey_2000_50_2025-08-21_00-02-08",
        rf"..\vit-recordings-maxi\2000_tracks50_samefps\udacity_2000_50_2025-08-21_00-19-49",

        rf"..\vit-recordings-maxi\3000_tracks50_samefps\beamng_3000_50_2025-08-22_00-06-00",
        rf"..\vit-recordings-maxi\3000_tracks50_samefps\donkey_3000_50_2025-08-21_23-32-56",
        rf"..\vit-recordings-maxi\3000_tracks50_samefps\udacity_3000_50_2025-08-21_23-49-59"
        ]

    # evaluate distribution
    plot_steering_distribution(folder_paths, normalize=False)

    percentage = [ 1,
                      1,
                      1,
                      1, # maxibon based
                        1,
                        1,
                        1
                    # 1,
                    # 1,
                    # 1,

                    # 1,
                    # 1,
                    # 1,
                    #     1,       # initial
                    #   0.4,     # extra donkey
                    #   0.4,     # extra donkey
                    #   0.4,     # extra udacity
                    #   0.4,
                    #   0.4       # beamng
                ]
    use_every_kth = [1,
                    1,
                     1,
                     1,
                    1,
                     1,
                     1
                    #  2, # every second beamng
                    #  1,
                    #  1,
                    
                    #  2, # every second beamng
                    #  1,
                    #  1,
                    ]
    # Create PyTorch datasets
    dataset = DrivingDatasetLazy(folder_paths=folder_paths,
                                    predict_throttle=False,
                                    preprocess_images=True,
                                    is_training=True,
                                    percentage = percentage,
                                    use_every_kth=use_every_kth)   
    get_distribution(dataset)
    train_ds, val_ds = split_data(dataset)
    
    print(f"Dataset contains {len(dataset)} images.")
    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=32,
                            shuffle=True, num_workers=8, prefetch_factor=1, pin_memory = True,
                            persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, 
                            num_workers=8,
                            prefetch_factor=1,
                            pin_memory = True,
                            persistent_workers=True)

    print("Data lodaded")

    current_date = datetime.now().strftime("%d-%m-%Y_%H-%M")
    if resume_checkpoint_path is not None:
        if suffix_folder_name is not None:
            checkpoint_dir = f"./multisim/checkpoints_{current_date}_lr-{lr_rate_tune}_{suffix_folder_name}/lane_keeping/vit/"
        else:
            checkpoint_dir = f"./multisim/checkpoints_{current_date}_lr-{lr_rate_tune}/lane_keeping/vit/"
    else:
        if suffix_folder_name is not None:
            checkpoint_dir = f"./multisim/checkpoints_{current_date}_{suffix_folder_name}/lane_keeping/vit/"
        else:
            checkpoint_dir = f"./multisim/checkpoints_{current_date}/lane_keeping/vit/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # write data paths for tracking
    distro = get_distribution(dataset)
    output_json_path = Path(checkpoint_dir + os.sep + "data.json")

    with open(output_json_path, 'w') as f:
        json.dump({"folders_path" : [folder_paths],
                "distribution" : distro,
                "percentage_per_dataset" : percentage,
                "use_kth_image_per_dataset" : use_every_kth,
                "resume_ckpt_path" : resume_checkpoint_path,
                "learning_rate_tune" : lr_rate_tune
                }, f, indent=4)
    
    filename = "vit_{}".format(env_name)
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename,
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        verbose=True,
    )
    earlystopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=5)
    val_loss_logger = ValLossCSVLogger(save_dir=checkpoint_dir, 
                                    env_name = env_name)

    # Trainer setup
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=[0],
        max_epochs=2000,
        precision="32-true",  
        callbacks=[checkpoint_callback, earlystopping_callback, val_loss_logger],
    )

    if resume_checkpoint_path is not None:
        # Load the model from checkpoint
        model = ViT.load_from_checkpoint(resume_checkpoint_path, learning_rate = lr_rate_tune)
    else:
        # Model init
        model = ViT()

    # Train
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Print checkpoint directory so it can be used in bash commands
    print(os.path.abspath(checkpoint_dir))