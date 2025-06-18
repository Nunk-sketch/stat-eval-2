import pandas as pd
import matplotlib.pyplot as plt
import json
import re  # For regular expressions to parse the text file

# --- Configuration ---
CSV_FILE_PATH = 'transactions1-5.csv'
DANISH_DICT_FILE_PATH = 'danish_words.txt'
ENGLISH_WORDS_FILE = 'potential_english_words_identified.json' # Load this file
CATEGORIZED_WORDS_FILE = 'categorized_english_words_final.txt'
OUTPUT_PLOT_FILE = 'all_non_danish_words_bar_chart.png'

def load_danish_dictionary(file_path):
    """Loads the Danish dictionary from the specified file."""
    danish_words = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    danish_words.add(word)
        print(f"Loaded {len(danish_words)} words from '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: Danish dictionary file '{file_path}' not found.")
        return set()
    return danish_words

def load_english_words(file_path):
    """Loads the identified English words from the JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            english_words = json.load(f)
        print(f"Loaded {len(english_words)} English words from '{file_path}'.")
        return set(word.lower() for word in english_words)
    except FileNotFoundError:
        print(f"Error: English words file '{file_path}' not found.")
        return set()

def identify_non_danish_non_english_words(csv_file, danish_words, english_words):
    """Identifies words from the CSV that are neither English nor Danish."""
    non_danish_non_english = set()
    try:
        df = pd.read_csv(csv_file)
        all_words = []
        for col in df.columns:
            all_words.extend(df[col].astype(str).str.split().sum()) # Split words in each column
        unique_words = set(word.lower() for word in all_words)

        for word in unique_words:
            if word.isalpha() and word not in danish_words and word not in english_words:
                non_danish_non_english.add(word)
        print(f"Found {len(non_danish_non_english)} Non-Danish/Non-English words.")
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
    return sorted(list(non_danish_non_english))

def parse_categorized_text_file(file_path):
    """Parses the categorized English words from the text file."""
    data = []
    current_word = None
    current_category = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Word: "):
                    current_word = line[len("Word: "):].strip()
                elif line.startswith("Category: "):
                    current_category = line[len("Category: "):].strip()
                elif not line and current_word is not None and current_category is not None:
                    data.append({'word': current_word, 'category': current_category})
                    current_word = None
                    current_category = None
            if current_word is not None and current_category is not None:
                data.append({'word': current_word, 'category': current_category})
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it exists.")
        return []
    return data

def visualize_all_non_danish_words(english_categories, non_danish_non_english, output_file):
    """Creates and displays a bar chart of all non-Danish word categories."""
    all_data = []
    for item in english_categories:
        all_data.append({'word': item['word'], 'category': item['category']})
    for word in non_danish_non_english:
         all_data.append({'word': word, 'category': 'Non-Danish/Non-English'})

    df = pd.DataFrame(all_data)

    if df.empty:
        print("No data to visualize. DataFrame is empty.")
        return

    category_counts = df['category'].value_counts()

    if category_counts.empty:
        print("No categories found to visualize.")
        return

    plt.figure(figsize=(12, 7))
    category_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of All Non-Danish Words')
    plt.xlabel('Category')
    plt.ylabel('Number of Words')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    for index, value in enumerate(category_counts):
        plt.text(index, value + 0.5, str(value), ha='center', va='bottom')

    plt.show()

    try:
        plt.savefig(output_file)
        print(f"Plot saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving plot to '{output_file}': {e}")

# --- Main execution ---
if __name__ == "__main__":
    print("Starting comprehensive non-Danish word analysis and visualization...")

    danish_words = load_danish_dictionary(DANISH_DICT_FILE_PATH)
    english_words = load_english_words(ENGLISH_WORDS_FILE)
    non_danish_non_english_words = identify_non_danish_non_english_words(CSV_FILE_PATH, danish_words, english_words)
    english_categories = parse_categorized_text_file(CATEGORIZED_WORDS_FILE)

    if danish_words and english_words:
        visualize_all_non_danish_words(english_categories, non_danish_non_english_words, OUTPUT_PLOT_FILE)
    else:
        print("Could not generate visualization. Missing data.")
    print("Process complete.")