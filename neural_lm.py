import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. CONFIGURATION
# ==========================================

torch.manual_seed(42)

batch_size = 256
block_size = 8

embedding_dim = 32
hidden_dim = 128

max_iters = 5000
eval_interval = 500
eval_iters = 100

learning_rate = 0.001

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
        data[i + block_size] for i in ix
    ])

    return x.to(device), y.to(device)


# ==========================================
# 4. NEURAL LANGUAGE MODEL
# ==========================================

class NeuralLanguageModel(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        # Character embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Hidden neural network layer
        self.fc1 = nn.Linear(
            block_size * embedding_dim,
            hidden_dim
        )

        # Output layer
        self.fc2 = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, idx, targets=None):

        # Shape: (B, block_size, embedding_dim)
        embeddings = self.embedding(idx)

        # Flatten context embeddings
        # Shape: (B, block_size * embedding_dim)
        embeddings = embeddings.reshape(
            embeddings.shape[0], -1
        )

        # Hidden layer
        hidden = torch.tanh(self.fc1(embeddings))

        # Output logits
        logits = self.fc2(hidden)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            # Keep only the last block_size characters
            idx_context = idx[:, -block_size:]

            logits, loss = self(idx_context)

            # Probability distribution
            probabilities = F.softmax(logits, dim=-1)

            # Sample next character
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat((idx, next_token), dim=1)

        return idx


# ==========================================
# 5. INITIALIZE MODEL
# ==========================================

model = NeuralLanguageModel(vocab_size).to(device)

total_params = sum(
    p.numel() for p in model.parameters()
)

print("Model parameters:", total_params)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ==========================================
# 6. EVALUATION
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
# 7. TRAINING
# ==========================================

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

    optimizer.step()


# ==========================================
# 8. GENERATE TEXT
# ==========================================

print("\nTraining completed!")

# Start with 8 newline characters as context
context = torch.zeros(
    (1, block_size),
    dtype=torch.long,
    device=device
)

generated = model.generate(
    context,
    max_new_tokens=500
)

generated_text = "".join(
    itos[int(i)] for i in generated[0, block_size:].tolist()
)

print("\nGenerated text:\n")
print(generated_text)


# ==========================================
# 9. SAVE MODEL
# ==========================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "block_size": block_size,
        "stoi": dataset["stoi"],
        "itos": itos
    },
    "checkpoints/neural_lm.pt"
)

print("\nModel saved successfully!")