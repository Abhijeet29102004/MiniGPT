import platform
import os
import psutil
import torch

print("=" * 50)
print("MINIGPT HARDWARE DIAGNOSTICS")
print("=" * 50)

# System information
print("\nSYSTEM INFORMATION")

print("OS:", platform.system())
print("OS Version:", platform.version())
print("Processor:", platform.processor())
print("CPU Cores:", psutil.cpu_count(logical=False))
print("Logical Cores:", psutil.cpu_count(logical=True))

# RAM
memory = psutil.virtual_memory()

print("\nMEMORY INFORMATION")

print("Total RAM:", round(memory.total / (1024**3), 2), "GB")
print("Available RAM:", round(memory.available / (1024**3), 2), "GB")

# PyTorch
print("\nPYTORCH INFORMATION")

print("PyTorch Version:", torch.__version__)
print("CPU Threads:", torch.get_num_threads())

# GPU
print("\nGPU INFORMATION")

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if hasattr(torch.backends, "mps"):
    print("Apple MPS Available:", torch.backends.mps.is_available())

if hasattr(torch, "xpu"):
    print("Intel XPU Available:", torch.xpu.is_available())

# Tensor operation test
print("\nTENSOR TEST")

x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

z = torch.matmul(x, y)

print("Matrix multiplication successful:", z.shape)

print("\nDIAGNOSTICS COMPLETED")