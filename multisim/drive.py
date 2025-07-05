import os
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import matplotlib.pyplot as plt
from udacity_gym.extras.model.lane_keeping.vit.vit_model import ViT

# Paths
model_path = "./multisim/checkpoints/lane_keeping/vit/vit_udacity_loss=0.000-v1.ckpt"
image_folder = "./multisim/data/2"

# Extract identifiers for output filename
model_name = os.path.splitext(os.path.basename(model_path))[0]
dataset_name = os.path.basename(os.path.normpath(image_folder))
output_dir = os.path.dirname(os.path.normpath(image_folder))  # parent of data folder
plot_filename = f"steering_plot_{model_name}_{dataset_name}.png"
plot_path = os.path.join(output_dir, plot_filename)

# Load model
model = ViT.load_from_checkpoint(model_path)
model.eval()
model.to("cuda")

# Image preprocessing
resize = T.Resize((160, 160))

steering_angles = []
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")])

for idx, file_name in enumerate(image_files, 1):
    image_path = os.path.join(image_folder, file_name)

    # Load and preprocess image
    input_image = read_image(image_path).float() / 255.0  # Normalize to [0,1]
    input_image = resize(input_image).unsqueeze(0).to("cuda")  # (1,3,160,160)

    # Predict steering angle
    with torch.no_grad():
        steering_angle = model(input_image).item()
    steering_angles.append(steering_angle)

    print(f"{idx}: {file_name} => steering angle = {steering_angle:.3f}")

# Plot steering angles over image count
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(steering_angles) + 1), steering_angles, marker='o')
plt.xlabel("Image index")
plt.ylabel("Predicted Steering Angle")
plt.title(f"Predicted Steering Angles - {model_name} on {dataset_name}")
plt.grid(True)

# Save plot
plt.savefig(plot_path)
plt.close()
print(f"Plot saved to {plot_path}")
