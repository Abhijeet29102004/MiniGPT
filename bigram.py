import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1. Configuration
# -----------------------------

torch.manual_seed(42)

batch_size = 32
block_size = 64
max_iters = 3000
eval_interval = 500
learning_rate = 0.01
eval_iters = 100

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Training device:", device)

# -----------------------------
# 2. Load processed dataset
# -----------------------------

dataset = torch.load("data/processed/dataset.pt")

train_data = dataset["train_data"]
val_data = dataset["val_data"]

vocab_size = dataset["vocab_size"]
itos = dataset["itos"]

print("Vocabulary size:", vocab_size)

# -----------------------------
# 3. Batch generator
# -----------------------------

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


# -----------------------------
# 4. Bigram Language Model
# -----------------------------

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):

        # idx shape: (batch_size, block_size)
        logits = self.token_embedding_table(idx)

        # logits shape: (batch_size, block_size, vocab_size)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, loss = self(idx)

            # Take predictions from the last position
            logits = logits[:, -1, :]

            # Convert scores into probabilities
            probabilities = F.softmax(logits, dim=-1)

            # Sample the next character
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append the predicted character
            idx = torch.cat((idx, next_token), dim=1)

        return idx


# -----------------------------
# 5. Initialize model
# -----------------------------

model = BigramLanguageModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print("Model parameters:", sum(
    p.numel() for p in model.parameters()
))


# -----------------------------
# 6. Evaluate model
# -----------------------------

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


# -----------------------------
# 7. Training loop
# -----------------------------

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


# -----------------------------
# 8. Generate text
# -----------------------------

print("\nTraining completed!")

context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated = model.generate(
    context,
    max_new_tokens=500
)

generated_text = "".join(
    itos[int(i)] for i in generated[0].tolist()
)

print("\nGenerated text:\n")
print(generated_text)


# -----------------------------
# 9. Save model
# -----------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "stoi": dataset["stoi"],
        "itos": itos
    },
    "checkpoints/bigram_model.pt"
)

print("\nModel saved successfully!")