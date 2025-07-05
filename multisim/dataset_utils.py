# prepare_data.py
import os
from typing import List, Tuple, Union

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Replace these with your actual values
IMAGE_HEIGHT, IMAGE_WIDTH = 66, 200  # example
SIMULATOR_NAMES = ["beamng", "udacity", "donkey"]  # example names
BEAMNG_SIM_NAME = "beamng"
UDACITY_SIM_NAME = "udacity"
DONKEY_SIM_NAME = "donkey"

# =============================================================================
# Data loading functions (same as original)
# =============================================================================
def _load_numpy_archive(archive_path: str, archive_name: str) -> dict:
    fp = os.path.join(archive_path, archive_name)
    assert os.path.exists(fp), f"Archive file {fp} does not exist"
    return np.load(fp, allow_pickle=True)

def load_archive(archive_path: str, archive_name: str) -> dict:
    return _load_numpy_archive(archive_path, archive_name)

def load_all_into_dataset(  archive_path: str,
    archive_names: List[str],
    predict_throttle: bool = False,
    env_name: str = None,
    max_num: int = None)  -> np.ndarray:

    obs = []
    actions = []
    for i in range(len(archive_names)):
        numpy_dict = load_archive(archive_path=archive_path, archive_name=archive_names[i])
        obs_i = numpy_dict["observations"]
        actions_i = numpy_dict["actions"]
        obs.append(obs_i)
        actions.append(actions_i)

    # print(f"shape before: {obs[0].shape}")
    obs = np.concatenate(obs)
    # print(f"shape after: {obs[0].shape}")
    actions = np.concatenate(actions)

    if len(actions.shape) > 2:
        actions = actions.squeeze(axis=1)

    if not predict_throttle:
        y = actions[:, 0]
    else:
        y = actions
    
    return obs, actions

def load_archive_into_dataset(
    archive_path: str,
    archive_names: List[str],
    seed: int,
    test_split: float = 0.2,
    predict_throttle: bool = False,
    env_name: str = None,
    num: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = []
    actions = []
    for name in archive_names:
        d = load_archive(archive_path, name)
        cnt = len(d["observations"]) if num is None else min(len(d["observations"]), num)
        obs.append(d["observations"][:cnt])
        actions.append(d["actions"][:cnt])

    X = np.concatenate(obs)
    actions = np.concatenate(actions)
    if actions.ndim > 2:
        actions = actions.squeeze(axis=1)
    y = actions if predict_throttle else actions[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed
    )
    print(f"Train: {X_train.shape}, {y_train.shape}; Test: {X_test.shape}, {y_test.shape}")
    return X_train, X_test, y_train, y_test

# =============================================================================
# Augmentation & Preprocessing
# =============================================================================
def crop(image: np.ndarray, env_name: str) -> np.ndarray:
    if env_name == BEAMNG_SIM_NAME:
        return image[80:-1]
    if env_name == UDACITY_SIM_NAME:
        return image[60:-25]
    if env_name == DONKEY_SIM_NAME:
        return image[60:]
    raise RuntimeError(f"Unknown env: {env_name}")

def resize(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def bgr2yuv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

def preprocess(image: np.ndarray, env_name: str, fake_images: bool = False) -> np.ndarray:
    if not fake_images:
        image = crop(image, env_name)
    image = resize(image)
    image = bgr2yuv(image)
    return image

def random_flip(image: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        y[0] = -y[0]
    return image, y

def random_translate(image: np.ndarray, y: np.ndarray, range_x=100, range_y=10):
    if np.random.rand() < 0.5:
        tx = range_x * (np.random.rand() - 0.5)
        ty = range_y * (np.random.rand() - 0.5)
        y[0] += tx * 0.002
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        h, w = image.shape[:2]
        image = cv2.warpAffine(image, M, (w, h))
    return image, y

def random_brightness(image: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = hsv[..., 2] * ratio
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image

def augment(image: np.ndarray, y: np.ndarray,
            range_x=100, range_y=10, fake_images=False) -> Tuple[np.ndarray, np.ndarray]:
    image, y = random_flip(image, y)
    image, y = random_translate(image, y, range_x, range_y)
    if not fake_images:
        image = random_brightness(image)
    return image, y

# =============================================================================
# PyTorch Dataset
# =============================================================================
class DrivingDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_training: bool,
        env_name: str,
        predict_throttle: bool = False,
        preprocess_images: bool = True,
        fake_images: bool = False
    ):
        self.X = X
        self.y = y
        self.is_training = is_training
        self.env_name = env_name
        self.predict_throttle = predict_throttle
        self.preprocess_images = preprocess_images
        self.fake_images = fake_images

        assert env_name in SIMULATOR_NAMES, f"Unknown env: {env_name}"
        self.indexes = np.arange(len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx].astype(np.float32)
        if label.ndim == 0:
            label = np.array([label], dtype=np.float32)

        if self.is_training:
            img, label = augment(img, label, fake_images=self.fake_images)

        if self.preprocess_images:
            img = preprocess(img, self.env_name, fake_images=self.fake_images)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        label_t = torch.from_numpy(label).float()
        return img_t, label_t

# =============================================================================
# Demo: main() usage
# =============================================================================
if __name__ == "__main__":
    archive_path = "./"
    archive_names = [
        "/home/lev/Downloads/training_datasets/udacity-2022_05_31_12_17_56-archive-agent-autopilot-seed-0-episodes-50.npz"]  # update as needed
    X_train, X_test, y_train, y_test = load_archive_into_dataset(
        archive_path=archive_path,
        archive_names=archive_names,
        seed=0, test_split=0.2,
        predict_throttle=False,
        env_name="udacity"
    )

    train_ds = DrivingDataset(X_train, y_train, is_training=True, env_name="udacity")
    test_ds = DrivingDataset(X_test, y_test, is_training=False, env_name="udacity")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # quick loop check
    for imgs, labs in train_loader:
        print("Batch imgs:", imgs.shape, "Batch labels:", labs.shape)
        break