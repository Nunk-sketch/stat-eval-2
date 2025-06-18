import csv
import re
from collections import Counter
import pandas as pd
from datetime import datetime
import time

# --- Prerequisites: Install mlxtend and pandas ---
# Run 'pip install mlxtend pandas' in your terminal first.
try:
    from mlxtend.preprocessing import TransactionEncoder
    # We use fpgrowth, which is more memory-efficient than apriori
    from mlxtend.frequent_patterns import fpgrowth, association_rules
except ImportError:
    print("Required libraries missing. Run 'pip install mlxtend pandas' in your terminal.")
    exit()

# Start total timer
total_start_time = time.time()
print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# --- 1) Load word lists and configure NLTK ---
print("Loading word dictionaries...")
dict_start_time = time.time()

try:
    from nltk.corpus import words

    english_words = set(w.lower() for w in words.words())
except ImportError:
    print("NLTK is not installed. Run 'pip install nltk'.")
    exit()
except LookupError:
    print("NLTK 'words' corpus not found. Run 'import nltk; nltk.download(\"words\")' in a Python console.")
    exit()

try:
    with open("danish_words.txt", encoding='utf-8') as f:
        danish_words = set(line.strip().lower() for line in f if line.strip())
except FileNotFoundError:
    print("File 'danish_words.txt' not found. Make sure it's in the same folder as the script.")
    exit()

dict_end_time = time.time()
print(f"Dictionaries loaded in {dict_end_time - dict_start_time:.2f} seconds")
print(f"  - English words: {len(english_words):,}")
print(f"  - Danish words: {len(danish_words):,}")

# --- 2) Read header from CSV and classify words ---
print("\nReading CSV header and classifying words...")
header_start_time = time.time()

csv_path = "transactions1-5.csv"
try:
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        word_list = header[1:]
except FileNotFoundError:
    print(f"File '{csv_path}' not found.")
    exit()

word_is_english = {
    w: (w.lower() in english_words and w.lower() not in danish_words and not re.search(r'\d', w))
    for w in word_list
}

header_end_time = time.time()
english_count = sum(word_is_english.values())
danish_count = len(word_list) - english_count
print(f"Word classification completed in {header_end_time - header_start_time:.2f} seconds")
print(f"  - Total words in CSV: {len(word_list):,}")
print(f"  - Classified as English: {english_count:,}")
print(f"  - Classified as Danish/Other: {danish_count:,}")

# --- 3) Process transactions ---
print("\nProcessing transactions...")
transaction_start_time = time.time()

ENGLISH_TOKEN = '__ENGLISH_WORD__'
processed_transactions = []
num_total_transactions = 0

with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for i, row in enumerate(reader):
        num_total_transactions += 1
        # Progress update every 500 transactions
        if (i + 1) % 500 == 0:
            elapsed = time.time() - transaction_start_time
            rate = (i + 1) / elapsed
            print(f"  Processed {i + 1} transactions... ({rate:.0f} transactions/sec)")

        present_words = [w for w, val in zip(word_list, row[1:]) if val == '1']
        new_transaction = set()
        contains_english = False
        for word in present_words:
            if word_is_english[word]:
                contains_english = True
            else:
                new_transaction.add(word)
        if contains_english:
            new_transaction.add(ENGLISH_TOKEN)
        if new_transaction:
            processed_transactions.append(list(new_transaction))

transaction_end_time = time.time()
transaction_duration = transaction_end_time - transaction_start_time
print(f"Transaction processing completed in {transaction_duration:.2f} seconds")
print(f"  - Total transactions: {num_total_transactions:,}")
print(f"  - Valid transactions: {len(processed_transactions):,}")
print(f"  - Processing rate: {num_total_transactions / transaction_duration:.0f} transactions/sec")

if not processed_transactions:
    print("No transactions to analyze. Stopping.")
    exit()

# --- 4) Optimized analysis with mlxtend and FP-Growth ---

# TUNABLE PARAMETERS - Adjust these to control the analysis
MIN_SUPPORT = 0.10  # Minimum support threshold (10%)
MIN_CONFIDENCE = 0.50  # Minimum confidence threshold (50%)
MAX_ANTECEDENT_LEN = 4  # Maximum number of Danish words in rule antecedent
MAX_ITEMSET_LEN = MAX_ANTECEDENT_LEN + 1  # Maximum itemset length for FP-Growth

print(f"\nPreparing data for mlxtend...")
prep_start_time = time.time()

te = TransactionEncoder()
te_ary = te.fit(processed_transactions).transform(processed_transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

prep_end_time = time.time()
print(f"Data preparation completed in {prep_end_time - prep_start_time:.2f} seconds")
print(f"  - DataFrame shape: {df.shape}")
print(f"  - Unique items: {len(te.columns_)}")

print(
    f"\nRunning FP-Growth to find frequent itemsets (min_support={MIN_SUPPORT * 100:.1f}%, max_len={MAX_ITEMSET_LEN})...")
print("This may take some time depending on data size...")

# FP-Growth timing
fpgrowth_start_time = time.time()
frequent_itemsets_df = fpgrowth(df, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_ITEMSET_LEN)
fpgrowth_end_time = time.time()
fpgrowth_duration = fpgrowth_end_time - fpgrowth_start_time

print(f"Found {len(frequent_itemsets_df):,} frequent itemsets in {fpgrowth_duration:.1f} seconds.")

# Check if we found any itemsets
if len(frequent_itemsets_df) == 0:
    print(f"\n⚠️  WARNING: No frequent itemsets found with min_support={MIN_SUPPORT * 100:.1f}%")
    print("This means no word combinations occur in at least 10% of transactions.")
    print("Try lowering MIN_SUPPORT (e.g., 0.05 for 5% or 0.02 for 2%)")
    print("Stopping analysis.")
    exit()

print(f"\nGenerating association rules (min_confidence={MIN_CONFIDENCE * 100:.1f}%)...")
rules_start_time = time.time()
rules_df = association_rules(frequent_itemsets_df, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules_end_time = time.time()
rules_duration = rules_end_time - rules_start_time

print(f"Found {len(rules_df):,} rules total in {rules_duration:.1f} seconds.")

# --- 5) Filter and display results ---
print("\nFiltering rules...")
filter_start_time = time.time()

# We're only interested in rules where the consequent (right-hand side) is our English token
english_rules_df = rules_df[rules_df['consequents'] == {ENGLISH_TOKEN}]

# Filter by antecedent length
english_rules_df = english_rules_df[english_rules_df['antecedents'].apply(len) <= MAX_ANTECEDENT_LEN]

# Sort by confidence and then support for best results
sorted_rules = english_rules_df.sort_values(by=['confidence', 'support'], ascending=False)

filter_end_time = time.time()
print(f"Rule filtering completed in {filter_end_time - filter_start_time:.2f} seconds")
print(f"  - Rules after filtering: {len(sorted_rules):,}")

print("\n--- Top Association Rules (Danish => English) ---")
print(
    f"Showing rules with max {MAX_ANTECEDENT_LEN} words, Support >= {MIN_SUPPORT * 100:.1f}% and Confidence >= {MIN_CONFIDENCE * 100:.1f}%\n")

if sorted_rules.empty:
    print("No rules found. Try lowering MIN_SUPPORT or MIN_CONFIDENCE in the script.")
else:
    display_start_time = time.time()

    pd.set_option('display.max_colwidth', None)
    sorted_rules['antecedents_str'] = sorted_rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))

    output_df = sorted_rules[['antecedents_str', 'confidence', 'support']].copy()
    output_df = output_df.rename(columns={'antecedents_str': 'If these words are found (Antecedent)'})

    output_df.index = range(1, len(output_df) + 1)

    # Display top 20 on screen
    print(output_df.head(20).to_string())

    display_end_time = time.time()
    print(f"\nDisplay formatting completed in {display_end_time - display_start_time:.2f} seconds")

    # --- 6) Save all rules to txt file ---
    print(f"\nSaving results...")
    save_start_time = time.time()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"association_rules_{timestamp}.txt"

    print(f"Saving all {len(output_df):,} rules to '{output_filename}'...")

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("ASSOCIATION RULES ANALYSIS - DANISH TO ENGLISH\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Transactions analyzed: {num_total_transactions:,}\n")
        f.write(f"Minimum support: {MIN_SUPPORT * 100:.1f}%\n")
        f.write(f"Minimum confidence: {MIN_CONFIDENCE * 100:.1f}%\n")
        f.write(f"Maximum antecedent length: {MAX_ANTECEDENT_LEN}\n")
        f.write(f"Maximum itemset length: {MAX_ITEMSET_LEN}\n")
        f.write(f"Rules found: {len(output_df):,}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("TIMING SUMMARY:\n")
        f.write(f"  - Dictionary loading: {dict_end_time - dict_start_time:.2f}s\n")
        f.write(f"  - Word classification: {header_end_time - header_start_time:.2f}s\n")
        f.write(f"  - Transaction processing: {transaction_duration:.2f}s\n")
        f.write(f"  - Data preparation: {prep_end_time - prep_start_time:.2f}s\n")
        f.write(f"  - FP-Growth algorithm: {fpgrowth_duration:.2f}s\n")
        f.write(f"  - Rule generation: {rules_duration:.2f}s\n")
        f.write(f"  - Rule filtering: {filter_end_time - filter_start_time:.2f}s\n")
        f.write("\n" + "=" * 60 + "\n\n")

        # Save all rules in nice format
        for idx, row in output_df.iterrows():
            f.write(f"Rule {idx}:\n")
            f.write(f"  If word '{row['If these words are found (Antecedent)']}' is found\n")
            f.write(f"  Then there's {row['confidence']:.1%} probability of English words\n")
            f.write(
                f"  Support: {row['support']:.1%} (occurs in {row['support'] * num_total_transactions:.0f} transactions)\n")
            f.write("-" * 40 + "\n")

        # Add summary
        f.write("\n" + "=" * 60 + "\n")
        f.write("SUMMARY OF TOP 10 RULES:\n")
        f.write("=" * 60 + "\n")
        top_10 = output_df.head(10)
        for idx, row in top_10.iterrows():
            f.write(
                f"{idx:2d}. {row['If these words are found (Antecedent)']:15s} - Confidence: {row['confidence']:.1%}, Support: {row['support']:.1%}\n")

    save_end_time = time.time()
    print(f"File saved in {save_end_time - save_start_time:.2f} seconds")
    print(f"All rules saved to '{output_filename}'")

# Calculate total time
total_end_time = time.time()
total_duration = total_end_time - total_start_time

print(f"\n" + "=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
print(f"Dictionary loading:      {dict_end_time - dict_start_time:8.2f}s")
print(f"Word classification:     {header_end_time - header_start_time:8.2f}s")
print(f"Transaction processing:  {transaction_duration:8.2f}s")
print(f"Data preparation:        {prep_end_time - prep_start_time:8.2f}s")
print(f"FP-Growth algorithm:     {fpgrowth_duration:8.2f}s ({fpgrowth_duration / total_duration * 100:.1f}% of total)")
print(f"Rule generation:         {rules_duration:8.2f}s")
print(f"Rule filtering:          {filter_end_time - filter_start_time:8.2f}s")
if 'save_start_time' in locals():
    print(f"File saving:             {save_end_time - save_start_time:8.2f}s")
print("-" * 60)
print(f"TOTAL ANALYSIS TIME:     {total_duration:8.2f}s ({total_duration / 60:.1f} minutes)")
print(f"Analysis completed at:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\nAnalysis completed! You can now open '{output_filename}' to see all rules.")

# --- 7) Parameter tuning suggestions ---
print(f"\n--- PARAMETER TUNING SUGGESTIONS ---")
print(f"Current settings:")
print(f"  MIN_SUPPORT = {MIN_SUPPORT} ({MIN_SUPPORT * 100:.1f}%)")
print(f"  MIN_CONFIDENCE = {MIN_CONFIDENCE} ({MIN_CONFIDENCE * 100:.1f}%)")
print(f"  MAX_ANTECEDENT_LEN = {MAX_ANTECEDENT_LEN}")
print(f"  MAX_ITEMSET_LEN = {MAX_ITEMSET_LEN}")
print(f"\nTo find MORE rules: Lower MIN_SUPPORT or MIN_CONFIDENCE")
print(f"To find FEWER/STRONGER rules: Raise MIN_SUPPORT or MIN_CONFIDENCE")
print(f"To find more complex patterns: Increase MAX_ANTECEDENT_LEN or MAX_ITEMSET_LEN")
print(f"WARNING: Higher MAX_ITEMSET_LEN dramatically increases computation time!")