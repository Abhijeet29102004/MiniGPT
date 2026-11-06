import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. CONFIGURATION
# ==========================================

torch.manual_seed(42)

batch_size = 32
block_size = 128

embedding_dim = 128
num_heads = 4
num_layers = 4
dropout = 0.1

max_iters = 5000
eval_interval = 500
eval_iters = 50

learning_rate = 3e-4

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Training device:", device)


# ==========================================
# 2. LOAD DATASET
# ==========================================

dataset = torch.load("data/processed/dataset.pt")

train_data = dataset["train_data"]
val_data = dataset["val_data"]

vocab_size = dataset["vocab_size"]
stoi = dataset["stoi"]
itos = dataset["itos"]

print("Vocabulary size:", vocab_size)


# ==========================================
# 3. BATCH GENERATOR
# ==========================================

def get_batch(split):

    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ])

    return x.to(device), y.to(device)


# ==========================================
# 4. CAUSAL SELF-ATTENTION HEAD
# ==========================================

class Head(nn.Module):

    def __init__(self, head_size):

        super().__init__()

        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

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

        weights = self.dropout(weights)

        output = weights @ value

        return output


# ==========================================
# 5. MULTI-HEAD ATTENTION
# ==========================================

class MultiHeadAttention(nn.Module):

    def __init__(self):

        super().__init__()

        head_size = embedding_dim // num_heads

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        outputs = [
            head(x) for head in self.heads
        ]

        output = torch.cat(outputs, dim=-1)

        output = self.projection(output)

        output = self.dropout(output)

        return output


# ==========================================
# 6. FEED-FORWARD NETWORK
# ==========================================

class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(embedding_dim, 4 * embedding_dim),

            nn.ReLU(),

            nn.Linear(4 * embedding_dim, embedding_dim),

            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.network(x)


# ==========================================
# 7. TRANSFORMER BLOCK
# ==========================================

class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.attention = MultiHeadAttention()

        self.feed_forward = FeedForward()

        self.ln1 = nn.LayerNorm(embedding_dim)

        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):

        # Attention with residual connection
        x = x + self.attention(self.ln1(x))

        # Feed-forward with residual connection
        x = x + self.feed_forward(self.ln2(x))

        return x


# ==========================================
# 8. COMPLETE GPT MODEL
# ==========================================

class MiniGPT(nn.Module):

    def __init__(self):

        super().__init__()

        # Character embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock()
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        # Predict next character
        self.language_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # Token embeddings
        token_emb = self.token_embedding(idx)

        # Position embeddings
        positions = torch.arange(
            T,
            device=idx.device
        )

        position_emb = self.position_embedding(positions)

        # Combine both embeddings
        x = token_emb + position_emb

        # Transformer layers
        x = self.blocks(x)

        # Final normalization
        x = self.final_norm(x)

        # Vocabulary logits
        logits = self.language_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.reshape(B * T, C)

            targets = targets.reshape(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        self.eval()

        for _ in range(max_new_tokens):

            # Keep the most recent context
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            # Last time-step predictions
            logits = logits[:, -1, :]

            probabilities = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx


# ==========================================
# 9. INITIALIZE MODEL
# ==========================================

model = MiniGPT().to(device)

total_params = sum(
    p.numel() for p in model.parameters()
)

print("Total model parameters:", total_params)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.1
)


# ==========================================
# 10. EVALUATION
# ==========================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = get_batch(split)

            logits, loss = model(X, Y)

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# ==========================================
# 11. TRAINING LOOP
# ==========================================

print("\nStarting MiniGPT pretraining...\n")

for iteration in range(max_iters):

    if iteration % eval_interval == 0:

        losses = estimate_loss()

        print(
            f"Step {iteration}: "
            f"Train loss {losses['train']:.4f}, "
            f"Val loss {losses['val']:.4f}"
        )

    X, Y = get_batch("train")

    logits, loss = model(X, Y)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()


# ==========================================
# PROMPT-BASED GENERATION
# ==========================================

model.eval()

prompt = "ROMEO:\n"

context = torch.tensor(
    [[stoi[ch] for ch in prompt]],
    dtype=torch.long,
    device=device
)

generated = model.generate(
    context,
    max_new_tokens=500
)

generated_text = "".join(
    itos[int(i)] for i in generated[0].tolist()
)

print("\nPrompt:", prompt)
print("\nGenerated continuation:\n")
print(generated_text)