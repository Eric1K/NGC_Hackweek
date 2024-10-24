# use nvidia gpu: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# pip install torchvision --index-url https://download.pytorch.org/whl/cu118

# torch, torchvision, pytorch

# import torch
# torch.cuda.is_available()
# print(torch.cuda.get_device_properties(0).name)

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())


import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())  # This should return True if CUDA is available
print(torch.cuda.get_device_properties(0).name)

