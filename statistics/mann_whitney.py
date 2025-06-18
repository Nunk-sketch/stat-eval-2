from scipy.stats import mannwhitneyu
import pandas as pd
import os


folder_path = "output_csv"
csv_files_to_process = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for filename in csv_files_to_process:
    input_csv_path = os.path.join(folder_path, filename)
    base_name, _ = os.path.splitext(filename)
    input_path_1_5 = f"output_1_5_csv/{base_name}_1.5.csv"
    input_path_2_0 = f"output_csv/{base_name}.csv"
    df1 = pd.read_csv(input_path_1_5)
    df2 = pd.read_csv(input_path_2_0)
    
    data1 = df1["english_only"] / df1["total_words"]
    data2 = df2["english_only"] / df2["total_words"]

    stat, p = mannwhitneyu(data1.dropna(), data2.dropna(), alternative='two-sided')
    print(f"Mann-Whitney U {base_name}: {stat}, p-value: {p}")
