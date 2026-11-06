import torch

# 1. Load the dataset
with open("data/raw/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Total characters:", len(text))

# 2. Build character vocabulary
chars = sorted(list(set(text)))

vocab_size = len(chars)

print("Vocabulary size:", vocab_size)
print("Characters:", chars)

# 3. Create character-to-integer mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 4. Encoder and decoder
def encode(text):
    return [stoi[ch] for ch in text]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

# 5. Convert entire dataset into token IDs
data = torch.tensor(encode(text), dtype=torch.long)

print("Encoded data shape:", data.shape)
print("First 100 tokens:", data[:100].tolist())
print("Decoded text:", decode(data[:100].tolist()))

# 6. Train-validation split
split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]

print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))

# 7. Save processed data
torch.save({
    "train_data": train_data,
    "val_data": val_data,
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size
}, "data/processed/dataset.pt")

print("Dataset preparation completed!")