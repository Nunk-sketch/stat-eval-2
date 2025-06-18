import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from newer_stats import dcreative, dfaq, dmultiple, dnon, driddles, dstem
import os


dcrea = dcreative()
dfaqs = dfaq()
dmulti = dmultiple()
dnons = dnon()
dridd = driddles()
dstems = dstem()

datasets = {
    'Creative': dcrea,
    'FAQ': dfaqs,
    'Riddles': dridd,
    'Multiple': dmulti,
    'Non': dnons,
    'STEM': dstems
}

### normalize data
for name, df in datasets.items():
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0  # or np.nan if you prefer



output_dir = 'images2.0'
os.makedirs(output_dir, exist_ok=True)

num_datasets = len(datasets)
num_cols = num_datasets
num_rows = 2  # Two rows: main and english_only

fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows))
if num_datasets == 1:
    axes = np.array([[axes[0]], [axes[1]]])

for col_idx, (name, df) in enumerate(datasets.items()):
    # Main histogram (excluding 'english_only')
    ax_main = axes[0, col_idx] if num_datasets > 1 else axes[0][col_idx]
    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in ['total_words', 'english_only']]
    for col in numeric_cols:
        sns.histplot(df[col], kde=True,stat= "density", label=col, alpha=0.5, ax=ax_main)
    ax_main.set_title(f'Histogram for {name}_2.0')
    ax_main.set_xlabel('Word Count (Normalized)')
    ax_main.set_ylabel('Frequency')
    ax_main.legend()

    # english_only histogram (if present)
    ax_eng = axes[1, col_idx] if num_datasets > 1 else axes[1][col_idx]
    if 'english_only' in df.columns:
        sns.histplot(df['english_only'], kde=True,stat= "density", color='orange', ax=ax_eng)
        ax_eng.set_title(f'english_only for {name}_2.0')
        ax_eng.set_xlabel('Word Count (Normalized)')
        ax_eng.set_ylabel('Frequency')
    else:
        ax_eng.set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/all_histograms_2.0_grid.png')
plt.close()