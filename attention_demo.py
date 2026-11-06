import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Configuration
batch_size = 2
sequence_length = 8
embedding_dim = 32
head_size = 16


class SelfAttentionHead(nn.Module):

    def __init__(self, embedding_dim, head_size):
        super().__init__()

        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        # Causal mask
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(sequence_length, sequence_length))
        )

    def forward(self, x):

        B, T, C = x.shape

        # Generate query, key and value vectors
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Calculate attention scores
        scores = queries @ keys.transpose(-2, -1)

        scores = scores * (keys.shape[-1] ** -0.5)

        # Prevent looking at future positions
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Normalize attention scores
        weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = weights @ values

        return output, weights


# Test the attention mechanism

x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim,
    device=device
)

attention = SelfAttentionHead(
    embedding_dim,
    head_size
).to(device)

output, weights = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)

print("\nAttention matrix for first example:")
print(weights[0].detach().cpu())

print("\nAttention test completed!")