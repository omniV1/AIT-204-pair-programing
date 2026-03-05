"""
train.py - Training Orchestrator
=================================
Run this script to train the LSTM model from scratch.

Usage:
    python train.py                          # Use default settings
    python train.py --epochs 50             # Custom epochs
    python train.py --text data/my_text.txt # Custom text file
    python train.py --help                  # Show all options

This script:
  1. Loads and preprocesses training text
  2. Tokenizes and creates sequences
  3. Builds the LSTM model
  4. Trains with callbacks (checkpoint, early stopping)
  5. Saves model artifacts
  6. Tests generation
  7. Creates training visualization plots
"""

import os
import sys
import argparse
import urllib.request

# Add parent directory to path so we can import text_generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.text_generator import TextGenerator, DEFAULT_CONFIG


# ─── SAMPLE TRAINING DATA ──────────────────────────────────────────────────
# Used if no training file is provided. This is an Alice in Wonderland excerpt.
SAMPLE_TEXT = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do: once or twice she had peeped into the book her sister
was reading, but it had no pictures or conversations in it, and what is the use of a
book, thought Alice, without pictures or conversations? So she was considering in her
own mind (as well as she could, for the hot day made her feel very sleepy and stupid),
whether the pleasure of making a daisy-chain would be worth the trouble of getting up
and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
There was nothing so very remarkable in that; nor did Alice think it so very much out
of the way to hear the Rabbit say to itself, Oh dear! Oh dear! I shall be late!
When she thought it over afterwards, it occurred to her that she ought to have
wondered at this, but at the time it all seemed quite natural; but when the Rabbit
actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried
on, Alice started to her feet, for it flashed across her mind that she had never before
seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning
with curiosity, she ran across the field after it, and fortunately was just in time to
see it pop down a large rabbit-hole under the hedge. In another moment down went Alice
after it, never once considering how in the world she was to get out again. The rabbit-hole
went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly
that Alice had not a moment to think about stopping herself before she found herself
falling down a very deep well. Either the well was very deep, or she fell very slowly,
for she had plenty of time as she went down to look about her and to wonder what was
going to happen next. First, she tried to look down and make out what she was coming
to, but it was too dark to see anything; then she looked at the sides of the well, and
noticed that they were filled with cupboards and book-shelves; here and there she saw
maps and pictures hung upon pegs. She took down a jar from one of the shelves as she
passed; it was labelled ORANGE MARMALADE, but to her great disappointment it was empty:
she did not like to drop the jar for fear of killing somebody, so managed to put it
into one of the cupboards as she fell past it. Well! thought Alice to herself, after
such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll
all think me at home! Why, I wouldn't say anything about it, even if I fell off the
top of the house! Which was very likely true. Down, down, down. Would the fall never
come to an end! I wonder how many miles I've fallen by this time? she said aloud.
I must be getting somewhere near the centre of the earth. Let me see: that would be
four thousand miles down, I think, for you see Alice had learnt several things of this
sort in her lessons in the schoolroom, and though this was not a very good opportunity
for showing off her knowledge, as there was no one to listen to her, still it was good
practice to say it over. Yes, that's about the right distance, but then I wonder what
Latitude or Longitude I've got to? Alice had no idea what Latitude was, or Longitude
either, but thought they were nice grand words to say. Presently she began again.
I wonder if I shall fall right through the earth! How funny it'll seem to come out
among the people that walk with their heads downward! The Antipathies, I think,
she was rather glad there was no one listening, this time, as it didn't sound at all
the right word, but she had a dim idea that she had it. She thought it so funny to
think about it, but then I shan't have to go to a lot of trouble. I wonder what
they do there. I know they drink tea and eat marmalade and all that sort of thing.
"""


def download_gutenberg_text(url: str, output_path: str) -> bool:
    """Download text from Project Gutenberg."""
    try:
        print(f"Downloading training text from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def load_training_text(text_path: str) -> str:
    """Load training text from file, falling back to sample text."""
    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        print(f"Loaded training text: {len(text)} characters from {text_path}")
        return text
    else:
        print(f"Warning: {text_path} not found. Using built-in sample text.")
        print("Tip: Download a book from Project Gutenberg for better results!")
        return SAMPLE_TEXT


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train the LSTM text generation model')
    parser.add_argument('--text', type=str, default='data/training_text.txt',
                        help='Path to training text file')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save model artifacts')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Training batch size')
    parser.add_argument('--seq-len', type=int, default=DEFAULT_CONFIG['sequence_length'],
                        help='Sequence length (context window size)')
    parser.add_argument('--lstm-units', type=int, default=DEFAULT_CONFIG['lstm_units'],
                        help='Number of LSTM units per layer')
    parser.add_argument('--embed-dim', type=int, default=DEFAULT_CONFIG['embedding_dim'],
                        help='Word embedding dimension')
    parser.add_argument('--download', type=str, default=None,
                        help='URL to download training text from (Project Gutenberg)')
    parser.add_argument('--seed', type=str, default='the cat sat',
                        help='Seed text for test generation after training')

    args = parser.parse_args()

    # ── STEP 0: Optional download ─────────────────────────────────────────────
    if args.download:
        download_gutenberg_text(args.download, args.text)

    # ── STEP 1: Load training text ────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: Loading Training Data")
    print("="*60)
    raw_text = load_training_text(args.text)

    # ── STEP 2: Initialize TextGenerator with custom config ───────────────────
    print("\n" + "="*60)
    print("STEP 2: Initializing TextGenerator")
    print("="*60)
    config = {
        'sequence_length': args.seq_len,
        'embedding_dim': args.embed_dim,
        'lstm_units': args.lstm_units,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
    }
    generator = TextGenerator(config=config)

    # ── STEP 3: Preprocess text ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Preprocessing Text")
    print("="*60)
    clean_text = generator.preprocess_text(raw_text)
    print(f"Original length: {len(raw_text)} chars")
    print(f"Cleaned length:  {len(clean_text)} chars")
    print(f"Sample: {clean_text[:100]}...")

    # ── STEP 4: Tokenize ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Tokenizing Text (Building Vocabulary)")
    print("="*60)
    token_list = generator.tokenize(clean_text)

    # ── STEP 5: Create sequences ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5: Creating Training Sequences (Sliding Window)")
    print("="*60)
    X, y = generator.create_sequences(token_list)

    if len(X) < 100:
        print("\nWarning: Very few training sequences. Add more text for better results.")
        print("Consider downloading a full book from Project Gutenberg:")
        print("  python train.py --download https://www.gutenberg.org/files/11/11-0.txt")

    # ── STEP 6: Build model ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6: Building LSTM Model")
    print("="*60)
    generator.build_model()

    # ── STEP 7: Train ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7: Training Model")
    print("="*60)
    generator.train(X, y, model_dir=args.model_dir)

    # ── STEP 8: Save model ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 8: Saving Model")
    print("="*60)
    saved_paths = generator.save(save_dir=args.model_dir)
    print(f"Saved: {saved_paths}")

    # ── STEP 9: Test generation ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 9: Testing Text Generation")
    print("="*60)
    for temp in [0.5, 1.0, 1.5]:
        print(f"\nTemperature {temp}:")
        generated = generator.generate_text(args.seed, num_words=30, temperature=temp)
        print(f"  {generated}")

    # ── STEP 10: Generate visualization ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 10: Generating Training Visualizations")
    print("="*60)
    viz_path = generator.plot_training_history(output_dir='visualizations')
    print(f"Plot saved to {viz_path}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Next steps:")
    print(f"  1. Start the API server:  uvicorn app.main:app --reload")
    print(f"  2. Open Streamlit UI:     streamlit run ../frontend/streamlit_app.py")
    print(f"  3. Or test via curl:")
    print(f'       curl -X POST http://localhost:8000/generate \\')
    print(f'         -H "Content-Type: application/json" \\')
    print(f'         -d \'{{"seed_text": "{args.seed}", "num_words": 50, "temperature": 1.0}}\'')


if __name__ == '__main__':
    main()
