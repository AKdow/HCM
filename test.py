import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.get_device_name(0))