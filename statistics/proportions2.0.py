import pandas as pd
import numpy as np
import os

# --- Configuration based on your existing script's logic ---
# This assumes your first script creates CSVs in 'output_1_5_csv'
folder_path = "output"
input_csv_folder = folder_path + "_csv"  # This will resolve to "output_1_5_csv"

# Define the output directory where new proportion analysis CSVs will be saved
output_proportions_folder = "proportions2.0"

# --- Main Script Logic ---
if __name__ == "__main__":
    # Ensure the output directory exists; create it if it doesn't.
    os.makedirs(output_proportions_folder, exist_ok=True)
    print(f"Output directory '{output_proportions_folder}' ensured.")

    # Get a list of all CSV files in the input directory
    csv_files_to_process = [f for f in os.listdir(input_csv_folder) if f.endswith('.csv')]

    if not csv_files_to_process:
        print(f"No CSV files found in the specified input directory: '{input_csv_folder}'.")
        print("Please ensure your previous script has run and created CSVs in this location.")
    else:
        # Define the columns from which proportions will be calculated
        proportion_source_cols = ['english_only', 'danish_only', 'ambiguous', 'neither']
        # Define the names for the new proportion columns that will be created
        proportion_output_cols = [f'prop_{col}' for col in proportion_source_cols]

        # Process each CSV file found in the input directory
        for filename in csv_files_to_process:
            input_csv_path = os.path.join(input_csv_folder, filename)

            # Construct the output filename.
            # Example: "CREATIVE PROMPTS_1.5.csv" will become "CREATIVE PROMPTS_1.5_proportions.csv"
            base_name, _ = os.path.splitext(filename)
            output_filename = f"{base_name}_proportions.csv"
            output_csv_path = os.path.join(output_proportions_folder, output_filename)

            print(f"\nProcessing input file: '{filename}'")

            try:
                # 1. Read the input CSV file into a pandas DataFrame
                df = pd.read_csv(input_csv_path)

                # Check if 'total_words' and other proportion source columns exist in the DataFrame
                required_cols = ['total_words'] + proportion_source_cols
                if not all(col in df.columns for col in required_cols):
                    print(f"Skipping '{filename}': Missing one or more required columns ({required_cols}).")
                    continue

                # 2. Calculate proportions for each row and add them as new columns to the DataFrame
                for col in proportion_source_cols:
                    # Calculate proportion. If 'total_words' for a row is 0, the proportion is set to 0
                    # to prevent division by zero errors and ensure valid proportions for that row.
                    df[f'prop_{col}'] = df.apply(
                        lambda row: row[col] / row['total_words'] if row['total_words'] != 0 else 0,
                        axis=1
                    )

                # 3. Prepare the main data DataFrame for output. This includes original columns
                # plus the newly calculated proportion columns.
                columns_for_output = ['filename', 'total_words'] + proportion_source_cols + proportion_output_cols
                output_df_main = df[columns_for_output]

                # 4. Calculate summary statistics (mean, standard deviation, standard error)
                # for each of the newly created proportion columns.
                summary_statistics = {}
                for prop_col in proportion_output_cols:
                    mean_val = df[prop_col].mean()
                    std_dev_val = df[prop_col].std()
                    # Standard error is calculated as standard deviation divided by the square root of the count
                    # of valid (non-NaN) data points. Handle cases where count is 0 to avoid division by zero.
                    std_err_val = std_dev_val / np.sqrt(df[prop_col].count()) if df[prop_col].count() > 0 else np.nan

                    summary_statistics[prop_col] = {
                        'Mean': mean_val,
                        'Standard Deviation': std_dev_val,
                        'Standard Error': std_err_val
                    }

                # 5. Convert the summary statistics dictionary into a pandas DataFrame.
                # This makes it easy to append as a table to the output CSV.
                summary_rows_list = []
                for metric_name in ['Mean', 'Standard Deviation', 'Standard Error']:
                    row_data = {'Metric': metric_name}
                    for prop_col_name in proportion_output_cols:
                        row_data[prop_col_name] = summary_statistics[prop_col_name][metric_name]
                    summary_rows_list.append(row_data)

                summary_df = pd.DataFrame(summary_rows_list)
                # Rename the columns in the summary DataFrame for clarity (remove 'prop_' prefix).
                summary_df.rename(columns={col: col.replace('prop_', '') for col in proportion_output_cols}, inplace=True)

                # 6. Write the combined data to the new CSV file.
                # First, write the main data (original + proportions).
                output_df_main.to_csv(output_csv_path, index=False)

                # Then, append the summary statistics after adding a few blank lines for visual separation
                # and a clear header for the summary section.
                with open(output_csv_path, 'a', newline='') as f:
                    f.write('\n\n') # Adds two blank lines
                    f.write('Summary Statistics\n') # Adds a header
                summary_df.to_csv(output_csv_path, mode='a', header=True, index=False)

                print(f"Successfully processed '{filename}' and saved results to '{output_csv_path}'.")

            except FileNotFoundError:
                print(f"Error: Input file not found at '{input_csv_path}'. Skipping this file.")
            except KeyError as e:
                print(f"Error: Missing expected column in '{filename}': {e}. Please ensure the input CSV structure is correct. Skipping this file.")
            except Exception as e:
                print(f"An unexpected error occurred while processing '{filename}': {e}. Skipping this file.")

        print("\nAll analysis complete for all files! Check the output CSV files in the specified output directory.")
