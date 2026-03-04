"""
text_generator.py - Core PyTorch LSTM Text Generation Module
=============================================================
This module implements the TextGenerator class which handles:
1. Text preprocessing (cleaning and normalisation)
2. Tokenisation (building word vocabulary via WordTokenizer)
3. Sequence creation (sliding window approach)
4. LSTM model construction (Embedding -> LSTM -> LSTM -> Linear)
5. Training with manual loop, scheduler, early stopping
6. Text generation (auto-regressive sampling with temperature)
7. Model persistence (save/load weights + tokenizer)

THEORY -> CODE GUIDE:
Each method is annotated with the corresponding mathematical formulation.
"""

import os
import json
import pickle
import re
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ─── Default Hyperparameters ────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "sequence_length": 50,      # Context window (number of words model sees)
    "embedding_dim":   100,     # Dimensionality of word embedding vectors
    "lstm_units":      150,     # Hidden units per LSTM layer
    "dropout_rate":    0.2,     # Dropout fraction applied between layers
    "batch_size":      128,     # Samples per gradient update
    "epochs":          100,     # Maximum training epochs
    "learning_rate":   0.001,   # Adam learning rate
    "validation_split": 0.1,   # Fraction of training data held for validation
}


# ─── Vocabulary ─────────────────────────────────────────────────────────────

class WordTokenizer:
    """
    Word-level tokenizer that mirrors the Keras Tokenizer API.

    Builds a frequency-sorted vocabulary from training text.
    Index 0 is reserved for padding (never assigned to a word).
    Index 1 is reserved for the <OOV> (out-of-vocabulary) token.
    Real words start at index 2.
    """

    OOV_TOKEN = '<OOV>'
    PAD_IDX   = 0
    OOV_IDX   = 1

    def __init__(self):
        # word -> integer
        self.word_index  = {self.OOV_TOKEN: self.OOV_IDX}
        # integer -> word  (0 maps to '' for padding, 1 maps to OOV token)
        self.index_word  = {self.PAD_IDX: '', self.OOV_IDX: self.OOV_TOKEN}
        # word -> count (set after fit_on_texts)
        self.word_counts = {}

    def fit_on_texts(self, texts: list) -> None:
        """Build vocabulary from a list of strings."""
        all_words = ' '.join(texts).split()
        counter   = Counter(all_words)
        self.word_counts = dict(counter)

        # Assign IDs in frequency order: most common word gets the lowest ID (2)
        for idx, (word, _) in enumerate(counter.most_common(), start=2):
            self.word_index[word] = idx
            self.index_word[idx]  = word

    def texts_to_sequences(self, texts: list) -> list:
        """Convert each string in texts to a list of integer token IDs."""
        return [
            [self.word_index.get(w, self.OOV_IDX) for w in text.split()]
            for text in texts
        ]


# ─── PyTorch Model ───────────────────────────────────────────────────────────

class LSTMTextModel(nn.Module):
    """
    Stacked two-layer LSTM for next-word prediction.

    THEORY -> CODE MAPPING:
    ─────────────────────────────────────────────────────────────────────
    Embedding layer:
        Words are discrete symbols. Each word ID is mapped to a
        trainable vector in R^embedding_dim via a lookup table.
        Similar words develop similar vectors during training.
        Input: (batch, seq_len) int  →  Output: (batch, seq_len, embed_dim) float

    LSTM layer 1  (returns all hidden states):
        h_t = LSTM(h_{t-1}, x_t)  at every timestep t
        Passing output from all timesteps to LSTM 2 gives the second layer
        full sequential context — equivalent to return_sequences=True in Keras.
        Input: (batch, seq_len, embed_dim)  →  Output: (batch, seq_len, lstm_units)

    Dropout:
        Randomly zeroes activations during training to prevent overfitting.

    LSTM layer 2  (returns only final hidden state):
        Processes the full sequence passed by layer 1.
        We take only the final hidden state h_T, which encodes the entire context.
        Input: (batch, seq_len, lstm_units)  →  h: (1, batch, lstm_units)

    Output (Linear):
        Projects h_T to a vocabulary-sized vector of raw logits.
        Softmax is NOT applied here — nn.CrossEntropyLoss expects raw logits.
        Input: (batch, lstm_units)  →  Output: (batch, vocab_size)
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 lstm_units: int, dropout_rate: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1     = nn.LSTM(embed_dim,   lstm_units, batch_first=True)
        self.dropout1  = nn.Dropout(dropout_rate)
        self.lstm2     = nn.LSTM(lstm_units,  lstm_units, batch_first=True)
        self.dropout2  = nn.Dropout(dropout_rate)
        self.fc        = nn.Linear(lstm_units, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)  –– integer word IDs
        emb  = self.embedding(x)             # (batch, seq_len, embed_dim)
        out, _     = self.lstm1(emb)         # (batch, seq_len, lstm_units)
        out  = self.dropout1(out)
        out, (h, _) = self.lstm2(out)        # h: (1, batch, lstm_units)
        out  = self.dropout2(h.squeeze(0))   # (batch, lstm_units)  final hidden state
        return self.fc(out)                  # (batch, vocab_size)  raw logits


# ─── TextGenerator ───────────────────────────────────────────────────────────

class TextGenerator:
    """
    PyTorch LSTM-based text generation system.

    THEORY BACKGROUND:
    -----------------
    This class implements a language model using LSTM networks.
    A language model learns P(w_t | w_{t-1}, ..., w_{t-n}) — the probability
    of the next word given the preceding n words.

    During training:
      - Input X: sequences of n words (as integer IDs)
      - Target y: the word immediately following each sequence (class index)
      - Loss: nn.CrossEntropyLoss = -log P(correct_class)

    During generation:
      - Seed text is tokenised and fed to the model
      - Model outputs raw logits; softmax converts them to probabilities
      - A word is sampled from this distribution (temperature-controlled)
      - The sampled word is appended to the context, and the loop repeats
    """

    def __init__(self, config: dict = None):
        self.config   = {**DEFAULT_CONFIG, **(config or {})}

        self.tokenizer        = None
        self.model            = None
        self.vocab_size       = 0
        self.training_history = None

        self.sequence_length = self.config["sequence_length"]
        self.embedding_dim   = self.config["embedding_dim"]
        self.lstm_units      = self.config["lstm_units"]
        self.dropout_rate    = self.config["dropout_rate"]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    # ─── PHASE 1: DATA PREPROCESSING ─────────────────────────────────────────

    def preprocess_text(self, raw_text: str) -> str:
        """
        Clean and normalise raw text for training.

        THEORY: Preprocessing reduces vocabulary size and noise.
        Treating "The" and "the" as the same word halves the vocabulary entries
        for "the", giving the model more data to learn from each word.
        """
        text = raw_text.lower()
        text = re.sub(r"[^a-z\s\.,;:!?'\-]", ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize(self, text: str) -> list:
        """
        Build vocabulary and convert text to integer sequences.

        THEORY: Neural networks work with numbers, not words.
        WordTokenizer builds a frequency-sorted mapping:
            word -> integer: {"the": 2, "alice": 3, ...}
            integer -> word: {2: "the", 3: "alice", ...}

        Index 0 = padding, Index 1 = <OOV>, real words start at 2.
        vocab_size = len(word_index) + 1  (+1 reserves slot 0 for padding)
        """
        self.tokenizer = WordTokenizer()
        self.tokenizer.fit_on_texts([text])
        # +1 because index 0 (padding) is outside word_index
        self.vocab_size = len(self.tokenizer.word_index) + 1
        sequences = self.tokenizer.texts_to_sequences([text])[0]

        print(f"Vocabulary size: {self.vocab_size} unique words")
        print(f"Total tokens:    {len(sequences)}")
        print(f"Sample mapping:  { {w: i for w, i in list(self.tokenizer.word_index.items())[:5]} }")
        return sequences

    def create_sequences(self, token_list: list):
        """
        Create input-output pairs using a sliding window.

        THEORY:
          X = sequence of n consecutive words (the "context")
          y = the word immediately following (class index, NOT one-hot)

        PyTorch's nn.CrossEntropyLoss expects integer class indices directly —
        no one-hot encoding is required (unlike Keras categorical_crossentropy).

        Sliding window (sequence_length=3):
          Tokens: [1, 2, 3, 4, 5]
          Window 1: X=[1,2,3]  y=4
          Window 2: X=[2,3,4]  y=5
        """
        X_sequences, y_targets = [], []
        for i in range(self.sequence_length, len(token_list)):
            context = token_list[i - self.sequence_length : i]
            target  = token_list[i]
            X_sequences.append(context)
            y_targets.append(target)

        X = np.array(X_sequences)   # (num_samples, sequence_length)
        y = np.array(y_targets)     # (num_samples,)  — integer class labels

        print(f"Created {len(X)} training sequences")
        print(f"X shape: {X.shape}  (samples × sequence_length)")
        print(f"y shape: {y.shape}  (samples,)  — class indices")
        return X, y

    # ─── PHASE 2: MODEL CONSTRUCTION ─────────────────────────────────────────

    def build_model(self) -> LSTMTextModel:
        """Construct and return the LSTM model, moved to self.device."""
        self.model = LSTMTextModel(
            vocab_size   = self.vocab_size,
            embed_dim    = self.embedding_dim,
            lstm_units   = self.lstm_units,
            dropout_rate = self.dropout_rate,
        ).to(self.device)

        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        print(f"\nModel architecture:\n{self.model}")
        print(f"\nTotal parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        return self.model

    # ─── PHASE 3: TRAINING ───────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray,
              model_dir: str = 'models') -> dict:
        """
        Train the LSTM with a manual PyTorch training loop.

        THEORY: Training minimises the cross-entropy loss via gradient descent.
        For each batch:
          1. Forward pass: logits = model(X_batch)
          2. Loss:  L = CrossEntropyLoss(logits, y_batch)
          3. Backward (BPTT): compute dL/dW for all weight matrices W
          4. Gradient clipping: prevents exploding gradients (||g|| <= 1.0)
          5. Weight update: W <- W - alpha * dL/dW  (Adam adaptive step)

        Callbacks equivalent:
          - Best-model saving: torch.save() when val_loss improves
          - Early stopping:    manual patience counter (stops at 10 epochs)
          - ReduceLROnPlateau: torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        os.makedirs(model_dir, exist_ok=True)
        best_path = os.path.join(model_dir, 'best_model.pt')

        # ── Train / Validation split ─────────────────────────────────────────
        val_split = self.config['validation_split']
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # ── DataLoaders ──────────────────────────────────────────────────────
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.long)
        )
        train_dl = DataLoader(train_ds, batch_size=self.config['batch_size'],
                              shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.config['batch_size'])

        # ── Optimizer, loss, scheduler ───────────────────────────────────────
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-6
        )

        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        best_val_loss   = float('inf')
        patience_counter = 0

        print(f"\nStarting training: {self.config['epochs']} max epochs, "
              f"batch_size={self.config['batch_size']}, "
              f"validation_split={val_split}")

        for epoch in range(self.config['epochs']):

            # ── Training pass ────────────────────────────────────────────────
            self.model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0
            for X_b, y_b in train_dl:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_b)
                loss   = criterion(logits, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                t_loss    += loss.item() * len(y_b)
                t_correct += (logits.argmax(1) == y_b).sum().item()
                t_total   += len(y_b)

            train_loss = t_loss    / t_total
            train_acc  = t_correct / t_total

            # ── Validation pass ───────────────────────────────────────────────
            self.model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for X_b, y_b in val_dl:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    logits = self.model(X_b)
                    loss   = criterion(logits, y_b)
                    v_loss    += loss.item() * len(y_b)
                    v_correct += (logits.argmax(1) == y_b).sum().item()
                    v_total   += len(y_b)

            val_loss = v_loss    / v_total
            val_acc  = v_correct / v_total

            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)

            print(f"Epoch {epoch+1:3d}/{self.config['epochs']}  "
                  f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  "
                  f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

            scheduler.step(val_loss)

            # ── Save best / early stopping ────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_path)
                print(f"  ✓ val_loss improved → saved best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"(no improvement for 10 epochs)")
                    break

        # Restore best weights
        self.model.load_state_dict(
            torch.load(best_path, map_location=self.device)
        )
        self.training_history = history

        history_path = os.path.join(model_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(
                {k: [float(v) for v in vals] for k, vals in history.items()},
                f, indent=2
            )
        print(f"Training history saved to {history_path}")
        return history

    # ─── PHASE 4: TEXT GENERATION ────────────────────────────────────────────

    def generate_text(self, seed_text: str, num_words: int = 50,
                      temperature: float = 1.0) -> str:
        """
        Generate text auto-regressively from a seed phrase.

        THEORY: Generation is inference in a loop.
          1. Tokenise current text → integer list
          2. Truncate/left-pad to sequence_length
          3. Forward pass → logits (vocab_size,)
          4. Apply temperature: logits / tau → softmax → probability vector
          5. Sample: np.random.choice(..., p=probs)
          6. Append sampled word; repeat

        Temperature parameter tau:
          tau < 1.0: sharper distribution (more deterministic)
          tau = 1.0: model's natural distribution
          tau > 1.0: flatter distribution (more random/creative)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained or loaded first.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not found. Train or load the model first.")

        self.model.eval()
        generated_text = seed_text.lower()

        with torch.no_grad():
            for _ in range(num_words):
                # ── Tokenise + truncate/pad ───────────────────────────────────
                token_list = self.tokenizer.texts_to_sequences([generated_text])[0]
                token_list = token_list[-self.sequence_length:]          # truncate
                pad_len    = self.sequence_length - len(token_list)
                token_list = [0] * pad_len + token_list                  # left-pad

                # ── Forward pass ──────────────────────────────────────────────
                x      = torch.tensor([token_list], dtype=torch.long).to(self.device)
                logits = self.model(x)[0]           # (vocab_size,)

                # ── Temperature sampling ──────────────────────────────────────
                logits = logits / temperature
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()

                # ── Sample from distribution ──────────────────────────────────
                predicted_index = np.random.choice(len(probs), p=probs)
                predicted_word  = self.tokenizer.index_word.get(predicted_index, '')
                if predicted_word:
                    generated_text += ' ' + predicted_word

        return generated_text

    # ─── PHASE 5: MODEL PERSISTENCE ──────────────────────────────────────────

    def save(self, save_dir: str = 'models') -> dict:
        """
        Save model weights, tokenizer, and config to disk.

        In PyTorch, model.state_dict() contains only the weight tensors.
        Architecture information (layer definitions) lives in the LSTMTextModel
        class itself, so we must also save config.json to reconstruct the
        model on load.
        """
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)

        tokenizer_path = os.path.join(save_dir, 'tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump({**self.config, 'vocab_size': self.vocab_size}, f, indent=2)

        print(f"Model saved to {save_dir}/")
        return {'model': model_path, 'tokenizer': tokenizer_path,
                'config': config_path}

    def load(self, save_dir: str = 'models') -> None:
        """Load a previously saved model, tokenizer, and configuration."""
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path) as f:
            saved_config = json.load(f)
        self.config.update(saved_config)
        self.vocab_size      = saved_config['vocab_size']
        self.sequence_length = self.config['sequence_length']
        self.embedding_dim   = self.config['embedding_dim']
        self.lstm_units      = self.config['lstm_units']
        self.dropout_rate    = self.config['dropout_rate']

        tokenizer_path = os.path.join(save_dir, 'tokenizer.pkl')
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        model_path = os.path.join(save_dir, 'model.pt')
        self.model = LSTMTextModel(
            vocab_size   = self.vocab_size,
            embed_dim    = self.embedding_dim,
            lstm_units   = self.lstm_units,
            dropout_rate = self.dropout_rate,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        print(f"Model loaded from {save_dir}/")

    # ─── PHASE 6: VISUALISATION ──────────────────────────────────────────────

    def plot_training_history(self, output_dir: str = 'visualizations') -> str:
        """Generate and save training loss / accuracy plots."""
        if not self.training_history:
            raise RuntimeError("No training history. Train the model first.")

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_history.png')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#1e1e2e')

        for ax in axes:
            ax.set_facecolor('#2e2e3e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#555577')

        axes[0].plot(self.training_history['loss'],
                     label='Training Loss',     color='#7c6af5', linewidth=2)
        if 'val_loss' in self.training_history:
            axes[0].plot(self.training_history['val_loss'],
                         label='Validation Loss', color='#f56a6a',
                         linewidth=2, linestyle='--')
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cross-Entropy Loss')
        axes[0].legend(facecolor='#2e2e3e', labelcolor='white')
        axes[0].grid(alpha=0.3, color='#555577')

        if 'accuracy' in self.training_history:
            axes[1].plot(self.training_history['accuracy'],
                         label='Training Accuracy', color='#6af5a2', linewidth=2)
            if 'val_accuracy' in self.training_history:
                axes[1].plot(self.training_history['val_accuracy'],
                             label='Val Accuracy', color='#f5d76a',
                             linewidth=2, linestyle='--')
        axes[1].set_title('Training & Validation Accuracy',
                          fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend(facecolor='#2e2e3e', labelcolor='white')
        axes[1].grid(alpha=0.3, color='#555577')

        plt.suptitle('LSTM Training Metrics', fontsize=16,
                     fontweight='bold', color='white')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='#1e1e2e')
        plt.close()
        print(f"Training plot saved to {output_path}")
        return output_path

    def get_model_info(self) -> dict:
        """Return model statistics as a dict (for the API /model/info endpoint)."""
        if self.model is None:
            return {"status": "not_loaded"}

        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)

        layers = []
        for name, module in self.model.named_children():
            param_shapes = [list(p.shape) for p in module.parameters()]
            layers.append({
                "name":         name,
                "type":         type(module).__name__,
                "param_shapes": param_shapes,
            })

        return {
            "status":               "loaded",
            "vocab_size":           self.vocab_size,
            "sequence_length":      self.sequence_length,
            "embedding_dim":        self.embedding_dim,
            "lstm_units":           self.lstm_units,
            "total_parameters":     total_params,
            "trainable_parameters": trainable_params,
            "layers":               layers,
            "config":               self.config,
        }
