import torch
print(torch.__version__)          # Kiểm tra phiên bản torch
print(torch.version.cuda)         # Kiểm tra CUDA version 
print(torch.cuda.is_available())  # Phải in ra True
print(torch.cuda.get_device_name(0))