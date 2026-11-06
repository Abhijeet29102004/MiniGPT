import torch
import torch.nn as nn
import torch.optim as optim

# Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Training device:", device)

# Reproducibility
torch.manual_seed(42)

# Generate synthetic data
X = torch.randn(1000, 10)

# Create meaningful labels
y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).long()

# Move data to device
X = X.to(device)
y = y.to(device)


# Define neural network
class SimpleNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


# Initialize model
model = SimpleNetwork().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training loop
epochs = 100

for epoch in range(epochs):

    # Forward propagation
    predictions = model(X)

    # Calculate loss
    loss = criterion(predictions, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    if (epoch + 1) % 10 == 0:

        predicted_classes = predictions.argmax(dim=1)

        accuracy = (
            predicted_classes == y
        ).float().mean()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.4f} "
            f"Accuracy: {accuracy.item()*100:.2f}%"
        )