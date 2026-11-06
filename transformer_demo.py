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
num_heads = 4
head_size = embedding_dim // num_heads


# ==========================================
# 1. SINGLE CAUSAL ATTENTION HEAD
# ==========================================

class Head(nn.Module):

    def __init__(self, embedding_dim, head_size, block_size):
        super().__init__()

        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        scores = query @ key.transpose(-2, -1)

        scores = scores * (key.shape[-1] ** -0.5)

        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)

        output = weights @ value

        return output


# ==========================================
# 2. MULTI-HEAD ATTENTION
# ==========================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        num_heads,
        embedding_dim,
        head_size,
        block_size
    ):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(embedding_dim, head_size, block_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            num_heads * head_size,
            embedding_dim
        )

    def forward(self, x):

        # Run all attention heads independently
        outputs = [head(x) for head in self.heads]

        # Concatenate their outputs
        output = torch.cat(outputs, dim=-1)

        # Project back to embedding dimension
        output = self.projection(output)

        return output


# ==========================================
# 3. FEED-FORWARD NETWORK
# ==========================================

class FeedForward(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 4. TRANSFORMER BLOCK
# ==========================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        block_size
    ):
        super().__init__()

        head_size = embedding_dim // num_heads

        self.attention = MultiHeadAttention(
            num_heads,
            embedding_dim,
            head_size,
            block_size
        )

        self.feed_forward = FeedForward(embedding_dim)

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(self.ln1(x))

        # Feed-forward + residual connection
        x = x + self.feed_forward(self.ln2(x))

        return x


# ==========================================
# 5. TEST EVERYTHING
# ==========================================

x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
).to(device)

block = TransformerBlock(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    block_size=sequence_length
).to(device)

output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

print(
    "Transformer block parameters:",
    sum(p.numel() for p in block.parameters())
)

print("Output contains NaN:", torch.isnan(output).any().item())

print("\nTransformer block test successful!")