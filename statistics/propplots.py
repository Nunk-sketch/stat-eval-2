import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

creative = r"proportions1.5\CREATIVE PROMPTS_1.5_proportions.csv"
faq = r"proportions1.5\FAQ_1.5_proportions.csv"
multiple = r"proportions1.5\MULTIPLE_1.5_proportions.csv"
non = r"proportions1.5\NON_1.5_proportions.csv"
riddles = r"proportions1.5\RIDDLES_1.5_proportions.csv"
stem = r"proportions1.5\STEM_1.5_proportions.csv"

dcrea15 = pd.read_csv(creative, sep=',')
dfaq15 = pd.read_csv(faq, sep=',')
dmulti15 = pd.read_csv(multiple, sep=',')
dnon15 = pd.read_csv(non, sep=',')
dridd15 = pd.read_csv(riddles, sep=',')
dstem15 = pd.read_csv(stem, sep=',')

# Drop specified columns from all datasets if they exist
columns_to_drop = ['filename', 'total_words', 'english_only', 'danish_only', 'ambiguous', 'neither']
for df in [dcrea15, dfaq15, dmulti15, dnon15, dridd15, dstem15]:
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

datasets = {
    'Creative': dcrea15,
    'FAQ': dfaq15,
    'Riddles': dridd15,
    'Multiple': dmulti15,
    'Non': dnon15,
    'STEM': dstem15
}


output_dir = 'images1.5'
os.makedirs(output_dir, exist_ok=True)

num_datasets = len(datasets)
num_cols = 3
num_rows = 4  # 2 rows for main histograms, 2 for english_only

fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows))
axes_flat = axes.flatten()

# Plot main histograms (first 2 rows)
for idx, (name, df) in enumerate(datasets.items()):
    row = idx // num_cols
    col = idx % num_cols
    ax_main = axes[row, col]
    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in ['total_words', 'prop_english_only']]
    for colname in numeric_cols:
        sns.histplot(df[colname], kde=True, label=colname, alpha=0.5, ax=ax_main)
    ax_main.set_title(f'Histogram for proportions of {name}_1.5')
    ax_main.set_xlabel('Proportion')
    ax_main.set_ylabel('Frequency')
    ax_main.legend()

# Plot english_only histograms (bottom 2 rows)
for idx, (name, df) in enumerate(datasets.items()):
    row = (idx // num_cols) + 2  # rows 2 and 3
    col = idx % num_cols
    ax_eng = axes[row, col]
    if 'prop_english_only' in df.columns:
        sns.histplot(df['prop_english_only'], kde=True, color='orange', ax=ax_eng)
        ax_eng.set_title(f'prop_english_only for {name}_1.5')
        ax_eng.set_xlabel('Proportion')
        ax_eng.set_ylabel('Frequency')
    else:
        ax_eng.set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/all_histograms_1.5_proportion_grid_new.png')
plt.close()
