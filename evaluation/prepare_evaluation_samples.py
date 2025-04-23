"""Script to prepare fixed evaluation samples from CNN/DailyMail and OpenWebText datasets.

This script will:
1. Load CNN/DailyMail validation set and save 200 filtered articles as binary tokens
2. Load OpenWebText validation set and save 200 samples as binary tokens
"""

import json
import numpy as np
import random
from datasets import load_dataset
from transformers import AutoTokenizer
import os

def prepare_cnndm_samples(max_samples=10, max_token_threshold=768, output_file="data/cnn_dailymail/val_200.bin", save_metadata=True):
    """Prepare and save CNN/DailyMail samples as binary token file."""
    print("Preparing CNN/DailyMail samples...")
    
    # Load the validation set
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation", streaming=False)
    articles = dataset["article"]
    
    # Use a standard tokenizer for tokenization
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Filter and tokenize articles
    tokenized_articles = []
    article_lengths = []
    original_texts = []  # Store original texts for metadata
    
    print("Filtering and tokenizing articles...")
    for article in articles:
        prompt = f"Summarize:\n{article.strip()}\nSummary:"
        tokens = tokenizer.encode(prompt, truncation=False, add_special_tokens=False)
        token_length = len(tokens)
        
        if token_length < max_token_threshold:
            tokenized_articles.append(tokens)
            article_lengths.append(token_length)
            original_texts.append(article)
    
    print(f"Found {len(tokenized_articles)} articles under {max_token_threshold} tokens")
    print(f"Token length range: {min(article_lengths)} - {max(article_lengths)}")
    
    # Randomly select articles
    random.seed(42)  # For reproducibility
    indices = list(range(len(tokenized_articles)))
    selected_indices = random.sample(indices, max_samples)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Prepare the binary data
    max_length = max(len(tokens) for tokens in tokenized_articles)
    output_data = np.zeros((max_samples, max_length), dtype=np.uint16)
    
    # Fill the output array with padded token sequences
    for i, idx in enumerate(selected_indices):
        tokens = tokenized_articles[idx]
        output_data[i, :len(tokens)] = tokens
    
    # Save to binary file
    output_data.tofile(output_file)
    print(f"Saved {max_samples} CNN/DailyMail articles to {output_file}")
    
    # Save metadata (original texts and lengths) if requested
    if save_metadata:
        metadata = {
            "articles": [original_texts[i] for i in selected_indices],
            "lengths": [len(tokenized_articles[i]) for i in selected_indices],
            "max_length": max_length
        }
        metadata_file = output_file.replace('.bin', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        print(f"Saved metadata to {metadata_file}")

def prepare_owt_samples(max_samples=10, context_length=768, input_file="data/openwebtext/val.bin", output_file="data/openwebtext/val_200.bin"):
    """Prepare and save OpenWebText samples."""
    print("Preparing OpenWebText samples...")
    
    # Load the full validation data
    print(f"Loading OpenWebText data from {input_file}")
    data = np.memmap(input_file, dtype=np.uint16, mode='r')
    
    # Calculate how many complete contexts we can extract
    num_complete_contexts = len(data) // context_length
    print(f"Found {num_complete_contexts} complete contexts of length {context_length}")
    
    # Randomly select starting points for our samples
    random.seed(42)  # For reproducibility
    selected_starts = random.sample(range(num_complete_contexts), max_samples)
    
    # Create the output array
    output_data = np.zeros((max_samples, context_length), dtype=np.uint16)
    
    # Extract the selected contexts
    for i, start_idx in enumerate(selected_starts):
        start_pos = start_idx * context_length
        end_pos = start_pos + context_length
        output_data[i] = data[start_pos:end_pos]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to binary file
    output_data.tofile(output_file)
    print(f"Saved {max_samples} OpenWebText samples to {output_file}")

def main():
    # Prepare CNN/DailyMail samples
    prepare_cnndm_samples()
    
    # Prepare OpenWebText samples
    prepare_owt_samples()

if __name__ == "__main__":
    main() 