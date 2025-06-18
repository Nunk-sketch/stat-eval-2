import pandas as pd
import os
import numpy as np


folder_path = "output_csv"

csv_files_to_process = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for filename in csv_files_to_process:
    input_csv_path = os.path.join(folder_path, filename)
    base_name, _ = os.path.splitext(filename)

    df = pd.read_csv(input_csv_path)

    df["prop_english_only"] = df.apply(
        lambda row: row["english_only"] / row['total_words'] if row['total_words'] != 0 else 0,
        axis=1
    )

    se_mean = 1 / len(df["prop_english_only"]) * sum(df["prop_english_only"])

    se_stand_dev = np.sqrt(1 / (len(df["prop_english_only"]) - 1) * sum((df["prop_english_only"] - se_mean)**2))

    conf_int = [se_mean - 1.96 * se_stand_dev/np.sqrt(len(df["prop_english_only"])), se_mean + 1.96 * se_stand_dev/np.sqrt(len(df["prop_english_only"]))]

    print(f"mean {base_name}", np.mean(df["prop_english_only"]))
    print(f"CI {base_name}", conf_int)