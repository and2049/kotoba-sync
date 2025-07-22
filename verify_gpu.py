import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"PyTorch was built with CUDA version: {torch.version.cuda}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch cannot find a compatible GPU and is using the CPU.")
    print("ACTION: Please check that your NVIDIA drivers are installed and up-to-date.")