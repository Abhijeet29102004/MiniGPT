import torch
import time

print("PyTorch:", torch.__version__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple GPU acceleration is available!")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Create tensors directly on MPS
x = torch.randn(2048, 2048, device=device)
y = torch.randn(2048, 2048, device=device)

start = time.time()

for _ in range(20):
    z = torch.matmul(x, y)

# Synchronize GPU before measuring
if device.type == "mps":
    torch.mps.synchronize()

elapsed = time.time() - start

print("Device:", device)
print("Output shape:", z.shape)
print("Execution time:", elapsed)
print("Test successful!")