import csv
import os
import numpy as np
from PIL import Image

archive_path = "/home/lev/Downloads/training_datasets/"
archive_names = [
    "beamng-2022_05_31_14_34_55-archive-agent-autopilot-seed-0-episodes-50.npz",
    "udacity-2022_05_31_12_17_56-archive-agent-autopilot-seed-0-episodes-50.npz",
    "donkey-2022_05_31_12_45_57-archive-agent-autopilot-seed-0-episodes-50.npz"
]
sims = ["beamng","udacity","donkey"]

save_dir = archive_path + "./raw"

os.makedirs(save_dir, exist_ok=True)

for sim, archive_file in zip(sims, archive_names):
    with np.load(archive_path + archive_file, allow_pickle=False) as archive:
        observations = archive["observations"]  # shape e.g. (N, H, W, C) or (N, C, H, W)
        actions = archive["actions"]

        # Check shape and channel order for correct image saving
        print("Observations shape:", observations.shape)
        
        # Save actions as CSV
        csv_path = os.path.join(save_dir, "actions.csv")
        
        names = []

        # Iterate over all images
        for i, img_array in enumerate(observations):
            # Convert from CHW to HWC if needed (common for torchvision)
            if img_array.shape[0] in [1, 3, 4]:  # channels first
                img_array = np.transpose(img_array, (1, 2, 0))

            # Convert to uint8 if float in [0,1]
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                img_array = (img_array * 255).astype(np.uint8)

            name = f"{sim}_{i:06d}"
            names.append(name)
            # Convert to PIL Image and save
            img = Image.fromarray(img_array)
            img.save(os.path.join(save_dir, f"{name}.png"))

            actions = archive["actions"].astype(np.float32)  # shape (N, ...) depending on your data

    # If actions is 2D or higher dimensional, flatten per row (e.g., actions[i] -> one row)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        for name, action in zip(names,actions):
            writer.writerow([name, action.flatten()])