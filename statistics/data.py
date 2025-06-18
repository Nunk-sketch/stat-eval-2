import numpy as np
import matplotlib.pyplot as plt
import os
import nltk.corpus
import string
import csv
import re

nltk.download('words')
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

with open("danish_words.txt", "r", encoding="utf-8") as f:
    danish_text = f.read()
danish_vocab = set(word.lower() for word in danish_text.splitlines() if word)

folder_path = "output_1_5"
csv_files = {}
direction = folder_path + "_csv"

def get_type_from_filename(filename):
    # Example: output_CREATIVE PROMPTS_20250610_180253_1.txt
    match = re.match(r"output_([^_]+(?: [^_]+)*)_", filename)
    return match.group(1) if match else "UNKNOWN"

for filename in os.listdir(folder_path):
    print(f"Processing file: {filename}")
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = ""
                next(f)
                for line in f:
                    text += line
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = ""
                next(f)
                for line in f:
                    text += line

        tokens = [word.strip(string.punctuation).lower() for word in text.split() if word.isalpha()]
        english_only = [w for w in tokens if w in english_vocab and w not in danish_vocab]
        danish_only = [w for w in tokens if w in danish_vocab and w not in english_vocab]
        ambiguous_words = [w for w in tokens if w in english_vocab and w in danish_vocab]
        neither = [w for w in tokens if w not in english_vocab and w not in danish_vocab]

        type_name = get_type_from_filename(filename)
        
        if folder_path == "output":
            csv_filename = f"{type_name}.csv"
        else:
            csv_filename = f"{type_name}_1.5.csv"
        csv_path = os.path.join(direction, csv_filename)

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["filename", "total_words", "english_only", "danish_only", "ambiguous", "neither"])
            writer.writerow([
                filename,
                len(tokens),
                len(english_only),
                len(danish_only),
                len(ambiguous_words),
                len(neither)
            ])
