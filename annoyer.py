import pandas as pd
import json
import os
import nltk
import time
from google import genai # IMPORTANT CHANGE: Using 'google-genai' library

# --- NLTK setup ---
try:
    nltk.data.find('corpora/words')
except nltk.downloader.DownloadError:
    print("NLTK 'words' corpus not found. Downloading...")
    nltk.download('words')
    print("NLTK 'words' corpus downloaded.")

from nltk.corpus import words as nltk_words_corpus

# --- Configuration ---
CSV_FILE_PATH = 'transactions1-5.csv'
DANISH_DICT_FILE_PATH = 'danish_words.txt'
OUTPUT_ENGLISH_WORDS_FILE = 'potential_english_words_identified.json'
OUTPUT_CATEGORIZED_WORDS_FILE = 'categorized_english_words_final.txt'
API_CHUNK_SIZE = 50 # Number of words to send per API request
API_REQUEST_DELAY = 1 # Seconds to wait between API requests to avoid rate limits

# --- API Key Setup ---
try:
    from API_KEY import API_KEY
    # IMPORTANT CHANGE: 'google-genai' client uses genai.Client directly
    # The API_KEY() might be if your API_KEY.py defines it as a function.
    # If API_KEY.py is just API_KEY = "YOUR_KEY", then remove the ()
    # For robust handling, I'm assuming API_KEY is a string directly.
    # If your API_KEY.py has API_KEY = "YOUR_KEY_HERE" then use api_key=API_KEY
    # If your API_KEY.py has def get_api_key(): return "YOUR_KEY_HERE" then use api_key=API_KEY()
    # Let's assume the simplest: API_KEY is a string.
    genai_client = genai.Client(api_key=API_KEY())
    print("API_KEY imported and Generative AI client configured successfully.")
except ImportError:
    print("ERROR: API_KEY.py not found or API_KEY not defined.")
    print("Please create 'API_KEY.py' in the same directory with 'API_KEY = \"YOUR_KEY_HERE\"'.")
    exit("Exiting: Cannot proceed without API Key.")
except Exception as e:
    print(f"An error occurred during API key configuration: {e}")
    exit("Exiting: API Key configuration failed.")

# --- Load Danish dictionary ---
common_danish_words = set()
try:
    with open(DANISH_DICT_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word:
                common_danish_words.add(word)
    print(f"Loaded {len(common_danish_words)} words from '{DANISH_DICT_FILE_PATH}'.")
except FileNotFoundError:
    print(f"Error: Danish dictionary file '{DANISH_DICT_FILE_PATH}' not found. This is critical for accurate filtering.")
    exit(f"Exiting: Danish dictionary file '{DANISH_DICT_FILE_PATH}' is required.")
except Exception as e:
    print(f"An error occurred while loading the Danish dictionary: {e}")
    exit("Exiting: Error loading Danish dictionary.")

# --- Load English words from NLTK ---
common_english_words = set(word.lower() for word in nltk_words_corpus.words())
print(f"Loaded {len(common_english_words)} English words from NLTK corpus.")

def identify_potential_english_words(word_list):
    """
    Identifies words that are likely English based on NLTK's English word corpus
    and by explicitly excluding words found in the Danish dictionary.
    Assumes a word is Danish if it's in both dictionaries.
    """
    potential_english = []
    print("Filtering words: assuming Danish if found in Danish dictionary...")
    for word in word_list:
        lower_word = word.lower()
        if len(lower_word) > 1 and lower_word.isalpha():
            if lower_word in common_english_words and lower_word not in common_danish_words:
                potential_english.append(word)
    return sorted(list(set(potential_english)))

def categorize_words_with_gemini(words_to_categorize):
    """
    Sends words to Gemini 2.0 Flash API for categorization in chunks using google-genai.
    """
    categorized_results = []
    total_words = len(words_to_categorize)
    print(f"\nStarting API categorization for {total_words} words (in chunks of {API_CHUNK_SIZE})...")

    for i in range(0, total_words, API_CHUNK_SIZE):
        chunk = words_to_categorize[i:i + API_CHUNK_SIZE]
        prompt_text = (
            "Categorize the following English words into these categories: "
            "**Technical Terms**, **Slang**, **General Vocabulary**, **Brand Names**, "
            "or **Other** if none apply. Provide the output as a JSON array of objects, "
            "where each object has 'word' and 'category' keys. For example: "
            "[{'word': 'software', 'category': 'Technical Terms'}, {'word': 'cool', 'category': 'Slang'}].\n\n"
            f"Words to categorize: {', '.join(chunk)}"
        )
        current_chunk_num = i // API_CHUNK_SIZE + 1
        total_chunks = (total_words + API_CHUNK_SIZE - 1) // API_CHUNK_SIZE
        print(f"\n--- Processing Chunk {current_chunk_num}/{total_chunks} (Words {i+1}-{min(i+API_CHUNK_SIZE, total_words)}) ---")
        # print(f"Prompt sent (first 200 chars): {prompt_text[:200]}...") # Uncomment for full prompt debug

        try:
            response = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt_text,
            )

            if response and hasattr(response, 'text') and response.text:
                json_string = response.text
                print(f"API Response Text (first 200 chars): {json_string[:200]}...")

                # --- ADDED LINES TO STRIP MARKDOWN FENCES ---
                # Remove leading "```json\n" and trailing "\n```" if they exist
                if json_string.startswith("```json"):
                    json_string = json_string[len("```json"):].lstrip('\n')
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].rstrip('\n')
                # Also handle cases where it might just be "```\n" or similar
                if json_string.startswith("```"):
                    json_string = json_string[len("```"):].lstrip('\n')
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].rstrip('\n')
                # --- END ADDED LINES ---

                try:
                    chunk_categorized = json.loads(json_string)
                    if isinstance(chunk_categorized, list):
                        categorized_results.extend(chunk_categorized)
                        print(f"  Successfully categorized {len(chunk_categorized)} words in this chunk.")
                    else:
                        print(f"  Warning: API returned non-list JSON for chunk {current_chunk_num}. Raw response: {json_string[:100]}...")
                except json.JSONDecodeError as e:
                    print(f"  ERROR: JSON decoding failed for chunk {current_chunk_num}: {e}. Raw response: {json_string[:200]}...")
            else:
                print(f"  Warning: No content in API response for chunk {current_chunk_num}.")
                print(f"  Raw response object: {response}") # Print the raw object for inspection

        except Exception as e: # Catching general exceptions as specific ones might differ
            print(f"  API call failed for chunk {current_chunk_num}: {e}")
            if "quota" in str(e).lower() or "billing" in str(e).lower() or "permission" in str(e).lower():
                 print("  Likely an API Key, Billing, or Quota problem.")
            print(f"  Retrying after {API_REQUEST_DELAY * 2} seconds...")
            time.sleep(API_REQUEST_DELAY * 2)
            continue

        if i + API_CHUNK_SIZE < total_words:
            time.sleep(API_REQUEST_DELAY)

    return categorized_results

def write_categorized_words_to_txt(categorized_data, output_file_path):
    """
    Writes the combined categorized word data to a plain text file.
    """
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("--- Categorized English Words ---\n\n")
        sorted_data = sorted(categorized_data, key=lambda x: x.get('word', '').lower())
        for item in sorted_data:
            word = item.get('word', 'N/A')
            category = item.get('category', 'Uncategorized')
            f.write(f"Word: {word}\nCategory: {category}\n\n")
    print(f"\nCategorized English words written to '{output_file_path}'.")

# --- Main execution ---
if __name__ == "__main__":
    print(f"Starting comprehensive word analysis from '{CSV_FILE_PATH}'...")
    try:
        df_header = pd.read_csv(CSV_FILE_PATH, nrows=0)
        all_unique_words = df_header.columns.tolist()
        print(f"Total unique words (columns) found in CSV: {len(all_unique_words)}")

        potential_english_words = identify_potential_english_words(all_unique_words)
        print(f"Identified {len(potential_english_words)} potential English words.")
        print(f"First 20 identified English words: {potential_english_words[:20]}")

        with open(OUTPUT_ENGLISH_WORDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(potential_english_words, f, ensure_ascii=False, indent=4)
        print(f"Raw identified English words saved to '{OUTPUT_ENGLISH_WORDS_FILE}' for review.")

        if potential_english_words:
            final_categorized_data = categorize_words_with_gemini(potential_english_words)
            print(f"Finished API categorization. Total categorized results: {len(final_categorized_data)}")

            if final_categorized_data:
                write_categorized_words_to_txt(final_categorized_data, OUTPUT_CATEGORIZED_WORDS_FILE)
            else:
                print("No words were categorized by the API. Output file not created.")
        else:
            print("No potential English words found. Skipping API categorization.")

    except FileNotFoundError:
        print(f"Error: Make sure '{CSV_FILE_PATH}' and '{DANISH_DICT_FILE_PATH}' are in the same directory as this script.")
    except Exception as e:
        print(f"An unexpected error occurred during the main process: {e}")