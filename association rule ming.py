import pandas as pd
import json
import os
import nltk
import re
from nltk.corpus import words as nltk_words_corpus
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time  # Added for timing
from datetime import datetime  # Added for timestamp in filename

# --- NLTK setup ---
try:
    nltk.data.find('corpora/words')
except nltk.downloader.DownloadError:
    print("NLTK 'words' corpus not found. Downloading...")
    nltk.download('words')
    print("NLTK 'words' corpus downloaded.")

# --- Configuration ---
PROMPTS_FOLDER = 'prompts'
OUTPUT_FOLDER = 'output_1_5'
DANISH_DICT_FILE_PATH = 'danish_words.txt'
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.33

MAX_ANTECEDENT_LEN = 3
MAX_ITEMSET_LEN = MAX_ANTECEDENT_LEN + 1

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
    print(
        f"Error: Danish dictionary file '{DANISH_DICT_FILE_PATH}' not found. Please ensure it is in the same directory.")
    exit("Exiting: Danish dictionary is required.")

# --- Load English words from NLTK ---
common_english_words = set(word.lower() for word in nltk_words_corpus.words())
print(f"Loaded {len(common_english_words)} English words from NLTK corpus.")


def is_potential_english_word(word):
    """
    Checks if a word is likely English based on NLTK and excluding Danish words.
    """
    lower_word = word.lower()
    return len(lower_word) > 1 and lower_word.isalpha() and \
        lower_word in common_english_words and lower_word not in common_danish_words


def process_prompts_and_answers_for_association(prompts_folder, output_folder):
    """
    Reads prompt and answer files, extracts words, and creates transactions for AR mining.
    Each transaction represents a prompt-answer pair.
    Items include: P_word (from prompt), A_word (from answer), HAS_ENGLISH_WORD_IN_ANSWER.
    If multiple answers exist for a prompt, each prompt-answer pair becomes a separate transaction.
    """
    transactions = []
    all_unique_words_in_data = set()
    specific_english_words_in_answers = set()  # This is collected, but will not be used for rules now

    # Get all original prompt files
    original_prompt_files = {os.path.splitext(f)[0]: os.path.join(prompts_folder, f)
                             for f in os.listdir(prompts_folder) if f.endswith('.txt')}

    if not original_prompt_files:
        print(f"No .txt files found in '{prompts_folder}'. Please check the folder path and contents.")
        return pd.DataFrame(), set(), set()

    print(f"Processing prompt-answer pairs from '{prompts_folder}' and '{output_folder}'...")
    processing_start_time = time.time()  # Start timing for processing

    # Get all answer files from the output folder once to optimize lookup
    all_output_files = os.listdir(output_folder)
    output_file_map = {}
    for filename in all_output_files:
        # Example: output_CREATIVE PROMPTS_20250610_180253_1.txt
        match = re.match(r"output_(.*?)_(\d{8}_\d{6})_(\d+)\.txt", filename)
        if match:
            original_prompt_base_name = match.group(1)
            block_num = int(match.group(3))
            if original_prompt_base_name not in output_file_map:
                output_file_map[original_prompt_base_name] = {}
            if block_num not in output_file_map[original_prompt_base_name]:
                output_file_map[original_prompt_base_name][block_num] = []
            output_file_map[original_prompt_base_name][block_num].append(os.path.join(output_folder, filename))

    # Iterate through each original prompt file to get its blocks (individual prompts)
    for original_prompt_base_name, original_prompt_file_path in original_prompt_files.items():
        try:
            with open(original_prompt_file_path, 'r', encoding='utf-8') as f:
                original_prompt_blocks = [block.strip() for block in f.read().strip().split('\n\n') if block.strip()]

            print(
                f"  - Original Prompt File '{original_prompt_base_name}.txt': Found {len(original_prompt_blocks)} blocks.")

            for i, prompt_text_from_block in enumerate(original_prompt_blocks, start=1):
                # Retrieve all matching answer files for this block
                matching_answer_files = output_file_map.get(original_prompt_base_name, {}).get(i, [])

                if not matching_answer_files:
                    print(
                        f"    - WARNING: No answer file found for '{original_prompt_base_name}.txt' block {i}. Skipping this prompt-answer set.")
                    continue

                # Process EACH answer file as a separate transaction
                for answer_file_path in matching_answer_files:
                    try:
                        with open(answer_file_path, 'r', encoding='utf-8') as af:
                            answer_file_content = af.read()

                        # Extract answer text from the content
                        answer_match = re.search(r"Answer: (.*)", answer_file_content, re.DOTALL)
                        answer_text_full = answer_match.group(1).strip() if answer_match else ""

                        # Tokenize prompt words and prefix them
                        prompt_words = [f"P_{word.strip().lower()}" for word in
                                        re.findall(r'\b\w+\b', prompt_text_from_block) if word.strip()]
                        all_unique_words_in_data.update(prompt_words)

                        # Tokenize answer words
                        answer_words = [word.strip().lower() for word in re.findall(r'\b\w+\b', answer_text_full) if
                                        word.strip()]
                        all_unique_words_in_data.update(answer_words)

                        # Determine if the answer contains any English words
                        has_english_word_in_answer = False
                        english_words_found_in_answer = []
                        for word in answer_words:
                            if is_potential_english_word(word):
                                has_english_word_in_answer = True
                                english_words_found_in_answer.append(word)
                                # specific_english_words_in_answers.add(word) # Keep for now if you want to inspect later, but not used for rules

                        # Create a transaction for this specific prompt-answer pair
                        current_transaction_items = set(prompt_words)
                        current_transaction_items.update(answer_words)

                        if has_english_word_in_answer:
                            current_transaction_items.add('HAS_ENGLISH_WORD_IN_ANSWER')

                        # current_transaction_items.update(english_words_found_in_answer) # This line adds specific English words to transaction, commenting out for Option 1 focus

                        if current_transaction_items:
                            transactions.append(list(current_transaction_items))

                    except Exception as e:
                        print(f"    - Error processing answer file '{answer_file_path}': {e}")
                        continue

        except Exception as e:
            print(f"Error processing original prompt file '{original_prompt_file_path}': {e}")
            continue

    processing_end_time = time.time()  # End timing for processing
    print(f"Data processing completed in {processing_end_time - processing_start_time:.2f} seconds.")

    # Create a boolean DataFrame for FP-Growth
    te = TransactionEncoder()
    if not transactions:
        print("No valid prompt-answer transactions generated.")
        return pd.DataFrame(), set(), set()

    df_transactions_start_time = time.time()
    te_ary = te.fit(transactions).transform(transactions)
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
    df_transactions_end_time = time.time()
    print(
        f"TransactionEncoder and DataFrame creation completed in {df_transactions_end_time - df_transactions_start_time:.2f} seconds.")

    print(
        f"\nFinal Transaction DataFrame created with {len(transactions)} rows (individual prompt-answer pairs) and {len(te.columns_)} columns.")
    return df_transactions, all_unique_words_in_data, specific_english_words_in_answers


# --- Main execution ---
if __name__ == "__main__":
    start_total_time = time.time()  # Start total timer
    print("Starting association rule mining on prompts and answers (using FP-Growth)...")
    print(f"Configured MIN_SUPPORT: {MIN_SUPPORT:.1%}")
    print(f"Configured MIN_CONFIDENCE: {MIN_CONFIDENCE:.1%}")
    print(f"Configured MAX_ANTECEDENT_LEN: {MAX_ANTECEDENT_LEN}")
    print(f"Configured MAX_ITEMSET_LEN (for FP-Growth): {MAX_ITEMSET_LEN}")

    df_transactions, all_words_in_data, specific_english_words_in_answers = \
        process_prompts_and_answers_for_association(PROMPTS_FOLDER, OUTPUT_FOLDER)

    if not df_transactions.empty:
        print("\nRunning FP-Growth to find frequent itemsets...")  # Added more specific print
        fpgrowth_start_time = time.time()  # Start timing FP-Growth
        frequent_itemsets = fpgrowth(df_transactions, min_support=MIN_SUPPORT, use_colnames=True,
                                     max_len=MAX_ITEMSET_LEN)
        fpgrowth_end_time = time.time()  # End timing FP-Growth
        print(f"FP-Growth completed in {fpgrowth_end_time - fpgrowth_start_time:.2f} seconds.")
        print(f"Found {len(frequent_itemsets)} frequent itemsets.")

        if frequent_itemsets.empty:
            print(
                f"\n⚠️  WARNING: No frequent itemsets found with min_support={MIN_SUPPORT * 100:.1f}% and max_len={MAX_ITEMSET_LEN}.")
            print("Consider lowering MIN_SUPPORT or increasing MAX_ITEMSET_LEN if this is unexpected.")
        else:
            print("\nGenerating association rules...")
            rules_start_time = time.time()  # Start timing rules generation
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
            rules_end_time = time.time()  # End timing rules generation
            print(f"Rule generation completed in {rules_end_time - rules_start_time:.2f} seconds.")
            print(f"Found {len(rules)} association rules.")

            print(f"DEBUG: Columns in 'rules' DataFrame: {rules.columns.tolist()}")
            if 'antecedents' not in rules.columns or 'consequents' not in rules.columns:
                print(
                    "FATAL ERROR: 'antecedents' or 'consequents' columns are missing from the generated rules DataFrame.")
                exit()

            # Filter for rules where the consequent (what 'leads to') involves 'HAS_ENGLISH_WORD_IN_ANSWER'
            # Antecedents should start with 'P_' (from prompt)
            rules_leading_to_english_in_answer = rules[
                rules['consequents'].apply(lambda x: 'HAS_ENGLISH_WORD_IN_ANSWER' in x) &
                rules['antecedents'].apply(lambda x: all(item.startswith('P_') for item in x))
                ]

            print(f"DEBUG: rules_leading_to_english_in_answer has {len(rules_leading_to_english_in_answer)} rows.")

            rules_leading_to_english_in_answer = rules_leading_to_english_in_answer[
                rules_leading_to_english_in_answer['antecedents'].apply(len) <= MAX_ANTECEDENT_LEN
                ]

            print(
                f"DEBUG: rules_leading_to_english_in_answer (after len filter) has {len(rules_leading_to_english_in_answer)} rows.")

            rules_leading_to_english_in_answer = rules_leading_to_english_in_answer.sort_values(by="confidence",
                                                                                                ascending=False)

            print("\n--- Rules where prompt antecedent leads to 'HAS_ENGLISH_WORD_IN_ANSWER' ---")
            print(f"(Max Antecedent Length: {MAX_ANTECEDENT_LEN})")
            if not rules_leading_to_english_in_answer.empty:
                for i, row in rules_leading_to_english_in_answer.head(10).iterrows():
                    antecedents_str = ', '.join(sorted(list(row['antecedents'])))
                    consequents_str = ', '.join(sorted(list(row['consequents'])))
                    print(f"Rule: {{{antecedents_str}}} => {{{consequents_str}}} "
                          f"(Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f})")

                # --- ADDED CODE TO WRITE RULES TO FILE ---
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Create a more descriptive filename
                output_filename = os.path.join(
                    OUTPUT_FOLDER,
                    f"rules_len{MAX_ANTECEDENT_LEN}_sup{MIN_SUPPORT:.2f}_conf{MIN_CONFIDENCE:.2f}_{timestamp}.txt"
                )

                print(f"\nWriting all {len(rules_leading_to_english_in_answer)} rules to '{output_filename}'...")
                try:
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        f.write(f"--- Association Rules: Danish Prompt Antecedent => HAS_ENGLISH_WORD_IN_ANSWER ---\n")
                        f.write(
                            f"Configuration: MIN_SUPPORT={MIN_SUPPORT:.1%}, MIN_CONFIDENCE={MIN_CONFIDENCE:.1%}, MAX_ANTECEDENT_LEN={MAX_ANTECEDENT_LEN}\n")
                        f.write(f"Total Rules Found (after filters): {len(rules_leading_to_english_in_answer)}\n\n")

                        # Sort by confidence for the file as well, if not already sorted
                        # (It is already sorted by confidence, so just iterate)
                        for i, row in rules_leading_to_english_in_answer.iterrows():
                            antecedents_str = ', '.join(sorted(list(row['antecedents'])))
                            consequents_str = ', '.join(sorted(list(row['consequents'])))

                            # Write all metrics for each rule
                            rule_line = (
                                f"Rule: {{{antecedents_str}}} => {{{consequents_str}}} "
                                f"(Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, "
                                f"Lift: {row['lift']:.3f}, Representativity: {row['representativity']:.3f}, "
                                f"Leverage: {row['leverage']:.3f}, Conviction: {row['conviction']:.3f}, "
                                f"Zhangs_Metric: {row['zhangs_metric']:.3f}, Jaccard: {row['jaccard']:.3f}, "
                                f"Certainty: {row['certainty']:.3f}, Kulczynski: {row['kulczynski']:.3f})\n"
                            )
                            f.write(rule_line)
                    print(f"Successfully wrote rules to '{output_filename}'.")
                except Exception as e:
                    print(f"Error writing rules to file '{output_filename}': {e}")
                # --- END OF ADDED CODE ---

            else:
                print(
                    "No rules found where consequent is 'HAS_ENGLISH_WORD_IN_ANSWER' with the given thresholds and antecedent length limit.")
    else:
        print("No transactions processed. Cannot perform association rule mining.")

    end_total_time = time.time()  # End total timer
    print(f"\nTotal script execution time: {end_total_time - start_total_time:.2f} seconds.")
    print("Association rule mining complete.")