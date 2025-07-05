import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    print(f"Number of available GPUs: {num_gpus}")
    print("GPU device names:")
    for i, name in enumerate(device_names):
        print(f"  GPU {i}: {name}")
else:
    print("CUDA GPUs are not available. Using CPU.")