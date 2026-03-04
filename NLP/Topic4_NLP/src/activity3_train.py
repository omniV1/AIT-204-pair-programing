"""
============================================================================
AIT-204 Deep Learning | Topic 4: Natural Language Processing
ACTIVITY 3: Training Pipeline — Train, Evaluate, Save (IMDB Dataset)
============================================================================

PURPOSE:
    Train the sentiment classifier on the IMDB movie review dataset (~25,000 reviews),
    evaluate it, plot training curves, and save the trained model + vocabulary to disk.

RUN THIS FILE: python activity3_train.py
    Produces: saved_model/model.pt, saved_model/vocab.json, plots/
============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import YOUR modules from Activities 1 and 2
from activity1_preprocessing import Vocabulary, preprocess_dataset, clean_text, tokenize
from activity2_model import SentimentClassifier, save_model

torch.manual_seed(42)
np.random.seed(42)


# =========================================================================
# LOAD IMDB DATASET
# =========================================================================
print("Loading IMDB dataset (this may take a moment on first run)...")

try:
    from datasets import load_dataset
    
    # Load IMDB dataset from Hugging Face
    dataset = load_dataset("imdb", trust_remote_code=True)
    
    # Convert to our format: list of (text, label) tuples
    train_data = [(item['text'], item['label']) for item in dataset['train']]
    test_data = [(item['text'], item['label']) for item in dataset['test']]
    
    print(f"Loaded {len(train_data)} training reviews and {len(test_data)} test reviews")
    
except ImportError:
    print("ERROR: 'datasets' library not installed. Run: pip install datasets")
    exit(1)


# =========================================================================
# HYPERPARAMETERS (tuned for IMDB)
# =========================================================================
MAX_LENGTH = 200      # IMDB reviews are longer - capture more context
EMBED_DIM = 128       # Larger embeddings for bigger vocabulary
HIDDEN_DIM = 64       # More capacity for complex patterns
BATCH_SIZE = 64       # Larger batches for faster training
LEARNING_RATE = 0.001
NUM_EPOCHS = 5        # IMDB is large - fewer epochs needed
DROPOUT = 0.5         # Higher dropout to prevent overfitting
MIN_FREQ = 5          # Only include words appearing 5+ times (reduces vocab size)
MAX_TRAIN_SAMPLES = 20000  # Use subset for faster training (set to None for full dataset)
MAX_VAL_SAMPLES = 5000


# =========================================================================
# PREPARE DATA
# =========================================================================
print("\nPreparing data...")

# Optionally use a subset for faster training
if MAX_TRAIN_SAMPLES and len(train_data) > MAX_TRAIN_SAMPLES:
    np.random.shuffle(train_data)
    train_data = train_data[:MAX_TRAIN_SAMPLES]

if MAX_VAL_SAMPLES and len(test_data) > MAX_VAL_SAMPLES:
    np.random.shuffle(test_data)
    test_data = test_data[:MAX_VAL_SAMPLES]

print(f"Using {len(train_data)} training and {len(test_data)} validation reviews")

# Build vocabulary on training data only
vocab = Vocabulary(min_freq=MIN_FREQ)

# Tokenize all training data first
print("Tokenizing training data...")
train_tokens = []
train_labels = []
for text, label in tqdm(train_data, desc="Tokenizing"):
    tokens = tokenize(clean_text(text))
    train_tokens.append(tokens)
    train_labels.append(label)

# Build vocabulary
print("Building vocabulary...")
vocab.build(train_tokens)
print(f"Vocabulary size: {len(vocab)} words (min_freq={MIN_FREQ})")

# Encode and pad training data
print("Encoding training data...")
train_encoded = [vocab.encode(tokens) for tokens in tqdm(train_tokens, desc="Encoding")]
from activity1_preprocessing import pad_sequence
train_padded = [pad_sequence(seq, MAX_LENGTH) for seq in train_encoded]
train_X = torch.tensor(train_padded, dtype=torch.long)
train_y = torch.tensor(train_labels, dtype=torch.long)

# Process validation data
print("Processing validation data...")
val_tokens = [tokenize(clean_text(text)) for text, _ in tqdm(test_data, desc="Tokenizing val")]
val_labels = [label for _, label in test_data]
val_encoded = [vocab.encode(tokens) for tokens in val_tokens]
val_padded = [pad_sequence(seq, MAX_LENGTH) for seq in val_encoded]
val_X = torch.tensor(val_padded, dtype=torch.long)
val_y = torch.tensor(val_labels, dtype=torch.long)

print(f"\nTraining:   {len(train_data)} reviews -> tensor {train_X.shape}")
print(f"Validation: {len(test_data)} reviews -> tensor {val_X.shape}")
print(f"Vocabulary: {len(vocab)} words")

# DataLoaders
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=BATCH_SIZE)


# =========================================================================
# CREATE MODEL
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = SentimentClassifier(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    pad_idx=vocab.pad_idx,
    dropout=DROPOUT,
).to(device)

print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =========================================================================
# TRAINING LOOP
# =========================================================================
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print(f"{'Epoch':>6} | {'Train Loss':>11} {'Train Acc':>10} | {'Val Loss':>11} {'Val Acc':>10}")
print("-" * 62)

for epoch in range(NUM_EPOCHS):
    # ---- TRAINING PHASE ----
    model.train()
    epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
    for batch_X, batch_y in progress_bar:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_X).squeeze(1)
        loss = criterion(predictions, batch_y.float())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch_y)
        epoch_correct += ((predictions >= 0.5).float() == batch_y.float()).sum().item()
        epoch_total += len(batch_y)
        
        progress_bar.set_postfix(loss=loss.item(), acc=epoch_correct/epoch_total)

    avg_train_loss = epoch_loss / epoch_total
    avg_train_acc = epoch_correct / epoch_total
    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)

    # ---- VALIDATION PHASE ----
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X).squeeze(1)
            loss = criterion(preds, batch_y.float())
            v_loss += loss.item() * len(batch_y)
            v_correct += ((preds >= 0.5).float() == batch_y.float()).sum().item()
            v_total += len(batch_y)

    avg_val_loss = v_loss / v_total
    avg_val_acc = v_correct / v_total
    val_losses.append(avg_val_loss)
    val_accs.append(avg_val_acc)

    print(f"{epoch+1:>6} | {avg_train_loss:>11.4f} {avg_train_acc:>9.1%} | "
          f"{avg_val_loss:>11.4f} {avg_val_acc:>9.1%}")

print("-" * 62)
print(f"{'FINAL':>6} | {train_losses[-1]:>11.4f} {train_accs[-1]:>9.1%} | "
      f"{val_losses[-1]:>11.4f} {val_accs[-1]:>9.1%}")


# =========================================================================
# SAVE MODEL AND VOCABULARY
# =========================================================================
# Move model back to CPU for saving
model = model.to("cpu")

os.makedirs("saved_model", exist_ok=True)
save_model(model, "saved_model/model.pt")
vocab.save("saved_model/vocab.json")

import json
with open("saved_model/config.json", "w") as f:
    json.dump({"max_length": MAX_LENGTH}, f)

print(f"\nSaved to saved_model/:")
print(f"  model.pt   — trained neural network ({sum(p.numel() for p in model.parameters()):,} params)")
print(f"  vocab.json — vocabulary ({len(vocab)} words)")
print(f"  config.json — inference config (max_length={MAX_LENGTH})")


# =========================================================================
# PLOTS
# =========================================================================
os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
actual_epochs = len(train_losses)

axes[0].plot(range(1, actual_epochs+1), train_losses, label='Train', linewidth=2)
axes[0].plot(range(1, actual_epochs+1), val_losses, label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (BCE)')
axes[0].set_title('Training and Validation Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, actual_epochs+1), train_accs, label='Train', linewidth=2)
axes[1].plot(range(1, actual_epochs+1), val_accs, label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('plots/training_curves.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: plots/training_curves.png")


# =========================================================================
# INFERENCE TEST
# =========================================================================
from activity1_preprocessing import preprocess_for_model

test_texts = [
    "I absolutely loved this movie it was fantastic and brilliantly made",
    "What a terrible waste of time avoid this garbage at all costs",
    "A great film with wonderful performances that moved me to tears",
    "The worst movie I have ever seen in my life do not watch",
    "Pretty good movie overall enjoyable experience worth watching",
    "Boring dull and completely uninspired throughout the entire film",
    "This movie was beautifully made with stunning cinematography",
    "I could not finish watching this it was so painfully bad",
]

print(f"\nInference test on unseen reviews:")
print("-" * 70)
model.eval()
for text in test_texts:
    tensor = preprocess_for_model(text, vocab, MAX_LENGTH)
    with torch.no_grad():
        prob = model(tensor).item()
    label = "POS" if prob >= 0.5 else "NEG"
    conf = abs(prob - 0.5) * 2  # Convert to confidence
    bar = "#" * int(prob * 20) + "." * (20 - int(prob * 20))
    print(f"  [{bar}] {prob:.2f} {label}  \"{text[:50]}...\"" if len(text) > 50 else f"  [{bar}] {prob:.2f} {label}  \"{text}\"")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE - Model trained on IMDB dataset")
print(f"  Vocabulary: {len(vocab):,} words | Accuracy: {val_accs[-1]:.1%}")
print("  Model saved to saved_model/")
print("=" * 70)
