import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import NegativeBinomial
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
    df1["version"] = "1.5"
    df2["version"] = "2.0"

    combined_df = pd.concat([df1, df2], ignore_index=True)

    nb_model = smf.glm(
        formula="english_only ~ version",
        data=combined_df,
        family=NegativeBinomial()
    ).fit()

    print(base_name)
    print(nb_model.summary())