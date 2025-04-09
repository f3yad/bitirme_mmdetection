import torch
import platform
print("PyTorch version:", torch.__version__)
print("Is CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)
print("Cuda Dev Count:", torch.cuda.device_count())
print("Current Cuda Dev:", torch.cuda.current_device())
print("Cuda Dev 0:", torch.cuda.device(0))
print("Cuda Dev 0 Name:", torch.cuda.get_device_name(0))
print("cuDNN version:", torch.backends.cudnn.version())
print("Platform Arch:", platform.architecture())