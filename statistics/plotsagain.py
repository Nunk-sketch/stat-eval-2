import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re

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


output_dir = 'images1.5'
os.makedirs(output_dir, exist_ok=True)

def extract_title_from_path(filepath):
    # Example: "proportions1.5\CREATIVE PROMPTS_1.5_proportions.csv"
    filename = os.path.basename(filepath)
    match = re.match(r"(.+?)_(\d+\.\d+)_proportions\.csv", filename)
    if match:
        prompt = match.group(1).replace('_', ' ')
        version = match.group(2)
        return f"{prompt} (v{version})"
    else:
        return filename

def plot_english_proportions(df, output_dir, title):
    plt.figure(figsize=(8, 5))
    plt.hist(df["prop_english_only"], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of prop_english_only - {title}')
    plt.xlabel('prop_english_only Proportion')
    plt.ylabel('Frequency')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'prop_english_only_hist_{title.replace(" ", "_")}.png'))
    plt.close()

def plot_other_proportions(df, output_dir, title):
    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col != 'prop_english_only']
    for colname in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[colname], kde=True, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {colname} - {title}')
        plt.xlabel(f'{colname} Proportion')
        plt.ylabel('Frequency')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{colname}_hist_{title.replace(" ", "_")}.png'))
        plt.close()

def plot_other_together(df, output_dir, title):
    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col != 'prop_english_only']
    plt.figure(figsize=(10, 6))
    for colname in numeric_cols:
        sns.histplot(df[colname], kde=True, label=colname, alpha=0.5)
    plt.title(f'Distribution of Other Proportions - {title}')
    plt.xlabel('Proportion')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{title.replace(" ", "_")}_other_proportions_hist.png'))
    plt.close()


title = extract_title_from_path(creative)
plot_english_proportions(dcrea15, output_dir, title)
plot_other_together(dcrea15, output_dir, title)