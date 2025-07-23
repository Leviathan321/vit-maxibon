from functools import cache
import os
import pathlib
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
from multisim.dataset_utils import augment, preprocess
import torchvision
import torch 
from torch.utils.data import Subset, DataLoader
from PIL import Image
import pandas as pd

class DrivingDatasetLazy:
    def __init__(self, 
                 is_training: bool,
                 folder_paths,  # str | Path | list[str | Path]
                 predict_throttle: bool = False,
                 preprocess_images: bool = True,
                 percentage=0.3):  # float | list[float] | dict[str | Path, float]

        # Normalize folder paths
        if isinstance(folder_paths, (str, pathlib.Path)):
            folder_paths = [folder_paths]
        self.folder_paths = [pathlib.Path(fp) for fp in folder_paths]

        metadata_list = []

        if isinstance(percentage, (float, int)):
            # Global sampling: concatenate, then take first N rows
            for idx, folder_path in enumerate(self.folder_paths):
                df = pd.read_csv(folder_path / 'actions.csv')
                df['folder_idx'] = idx
                metadata_list.append(df)

            full_metadata = pd.concat(metadata_list, ignore_index=True)

            n_total = len(full_metadata)
            n_sample = int(n_total * float(percentage))
            self.metadata = full_metadata.iloc[:n_sample].reset_index(drop=True)

        elif isinstance(percentage, list):
            if len(percentage) != len(self.folder_paths):
                raise ValueError("Length of percentage list must match number of folder_paths")

            for idx, (folder_path, perc) in enumerate(zip(self.folder_paths, percentage)):
                df = pd.read_csv(folder_path / 'actions.csv')
                df['folder_idx'] = idx
                n_total = len(df)
                n_sample = int(n_total * float(perc))
                df = df.iloc[:n_sample].reset_index(drop=True)
                metadata_list.append(df)

            self.metadata = pd.concat(metadata_list, ignore_index=True)

        self.counter = 0
        self.index_map = []  # List of (file_path, key, local_index)
        self.predict_throttle = predict_throttle
        self.preprocess_images = preprocess_images
        self.is_training = is_training

        # for path in self.file_paths:
        #     with np.load(path, allow_pickle=False) as archive:
        #         arr = archive["observations"]
        #         num_samples = len(arr)
        #         print(f"File {path}, samples: {num_samples}")
        #         for local_idx in range(num_samples):
        #             self.index_map.append((path,local_idx))
    
    def __len__(self):
        return len(self.metadata)
    
    # @cache
    def get_image(self, idx):
        base_filename = self.metadata.iloc[idx, 0]
   
        folder_idx = self.metadata.iloc[idx]['folder_idx']
        folder_path = self.folder_paths[folder_idx]

        assert folder_idx < len(self.folder_paths)

        possible_extensions = [".png", ".jpg", ".jpeg"]
        
        for ext in possible_extensions:
            file_path = folder_path.joinpath(base_filename + ext)
            if file_path.exists():
                with Image.open(file_path) as img:
                    return np.array(img)
                
    def _get_env_name(self, path):
        if "beamng" in path:
            return "beamng"
        elif "donkey" in path:
            return "donkey"
        elif "udacity" in path:
            return "udacity"
        else:
            return ValueError("Not known env_name")
        
    def get_steering(self, value):
        arr_str = value.strip()  # e.g., '[0.0473874 0.506135]'
        
        # Remove brackets
        arr_str = arr_str.strip('[]')
        
        # Split by whitespace
        arr_values = arr_str.split()
        
        # Convert to float numpy array
        arr = np.array(arr_values, dtype=np.float32)
        return arr
        
    def __getitem__(self, idx):
        img = self.get_image(idx)
        steering_csv = self.metadata.iloc[idx, 1]
        label = self.get_steering(steering_csv)

        # print("steering read:", label)
        # print("image type read:", type(img))
        # input()
        # with np.load(path, allow_pickle=False) as archive:
        #     img = archive["observations"][local_idx]
        #     label = archive["actions"][local_idx].astype(np.float32)
        self.counter = self.counter + 1

        if not self.predict_throttle:
            label = label[:1]

        # if self.is_training:
        #     img, label = augment(img, label, fake_images=False)
        
        env_name = self._get_env_name(self.metadata.iloc[idx, 0])

        if self.preprocess_images:
            img = preprocess(img, env_name, fake_images=False)
        
        image = Image.fromarray(img)
        #image.save(f"./image_vit_training_{self.counter}.png")
        #image.save(f"./image_vit_training.png")

        img = torchvision.transforms.ToTensor()(img)  # shape (C, H, W), float in [0, 1]
        
        label = torch.from_numpy(label).float()

        return img, label

def split_data(dataset, ratio=0.2, seed=42, percentage=1.0):
    # Validate percentage
    assert 0 < percentage <= 1.0, "Percentage must be in (0, 1]."

    n_total = len(dataset)
    n_used = int(n_total * percentage)

    # Shuffle all indices reproducibly
    rng = np.random.default_rng(seed=seed)
    all_indices = rng.permutation(n_total)

    # Select a random subset of the data
    used_indices = all_indices[:n_used]

    # Split into train and val
    n_val = int(n_used * ratio)
    n_train = n_used - n_val

    train_indices = used_indices[:n_train]
    val_indices = used_indices[n_train:]

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    return train_ds, val_ds


if __name__ == "__main__":
    folder = "/home/lev/Downloads/training_datasets"
    index = 1
    dataset = DrivingDatasetLazy(folder_paths=folder,
                                 predict_throttle=False,
                                 preprocess_images=True,
                                 is_training=True)
    print(f"Dataset contains {len(dataset)} images.")

    train_ds, val_ds = split_data(dataset, percentage=0.6)

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=32,
                            shuffle=True, 
                            num_workers=8, 
                            prefetch_factor=2, 
                            pin_memory = True)
    val_loader = DataLoader(val_ds, 
                            batch_size=32, 
                            shuffle=False, 
                            num_workers=8,
                            prefetch_factor=2,
                            pin_memory = True)
    
    print("accessing image:", train_ds[0][0].shape)
    # if len(dataset) == 0:
    #     print("No data found.")
    # else:
    #     image, action = dataset[index]

    #     action = action if dataset.predict_throttle else action[0]

    #     print(f"Loaded sample at index {index}: action = {action}")
    #     print(f"Image has shape: ", image.shape)

    #     plt.figure(figsize=(6,6))
    #     plt.imshow(image)   
    #     plt.title(f"Sample index {index} - action {action}")
    #     plt.axis('off')
    #     plt.show()