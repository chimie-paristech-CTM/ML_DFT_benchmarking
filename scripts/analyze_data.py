import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def preprocess_dat_file(dat_file_path):
    # Read the content of the file
    with open(dat_file_path, 'r') as file:
        lines = file.readlines()

    # Identify lines where [3+2] cycloaddition needs to be treated as a single field
    modified_lines = []
    for line in lines:
        # Replace only the problematic part with a quoted version
        line = re.sub(r'\[(\d[\+\d]+)\] (\w+)', r'[\1]\2', line)
        line = re.sub(r'\[(\d[\,\d]+)\] (\w+)', r'[\1]\2', line)
        modified_lines.append(line)

    # Write the modified content back to the file
    with open(dat_file_path, 'w') as file:
        file.writelines(modified_lines)

def remove_vert_line(entry):
    if str(entry)[-1] == '|':
        return entry[:-1]
    else:
        return entry


def read_dat_files_to_dataframes(folder_path):
    # Loop through all .dat files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.dat'):
            dat_file_path = os.path.join(folder_path, filename)

            # Preprocess the .dat file to handle [3+2] cycloaddition
            preprocess_dat_file(dat_file_path)

            functional = filename[:-4]

            # Read the .dat file into a DataFrame using regex to handle varying spaces
            try:
                df = pd.read_csv(dat_file_path, delim_whitespace=True, skiprows=2, engine='python', 
                                 names=['Name', 'Type', f'{functional}-DFT-forward', 'REF-forward', 
                                        f'{functional}-DELTA-forward', f'{functional}-DFT-reverse', 
                                        'REF-reverse', f'{functional}-DELTA-reverse', 
                                        f'{functional}-DFT-reaction', 'REF-reaction', 
                                        f'{functional}-DELTA-reaction'])
                for column in df.columns:
                    df[column] = df[column].apply(lambda x: remove_vert_line(x))

            except pd.errors.ParserError:
                print(f"Warning: Skipping file '{filename}' due to a parsing error.")


def combine_and_compute_std(folder_path):
    all_dfs = []
    # Loop through all .csv files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path, index_col=0)
            all_dfs.append(df)

    merged_df = all_dfs[0]
    for df in all_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['Name', 'Type', 'REF-forward', 'REF-reverse', 'REF-reaction'])
    
    dft_forward_columns = merged_df.filter(like='DFT-forward')
    dft_reverse_columns = merged_df.filter(like='DFT-reverse')
    dft_reaction_columns = merged_df.filter(like='DFT-reaction')

    merged_df['Range_DFT_forward'] = dft_forward_columns.apply(lambda row: row.max() - row.min(), axis=1)
    merged_df['Std_DFT_forward'] = dft_forward_columns.std(axis=1)

    merged_df['Range_DFT_reverse'] = dft_reverse_columns.apply(lambda row: row.max() - row.min(), axis=1)
    merged_df['Std_DFT_reverse'] = dft_reverse_columns.std(axis=1)

    merged_df['Range_DFT_reaction'] = dft_reaction_columns.apply(lambda row: row.max() - row.min(), axis=1)
    merged_df['Std_DFT_reaction'] = dft_reaction_columns.std(axis=1)

    for column in ['Range_DFT_forward', 'Std_DFT_forward', 'Range_DFT_reverse',
                   'Std_DFT_reverse', 'Range_DFT_reaction', 'Std_DFT_reaction']:
        merged_df = merged_df.loc[abs(merged_df[column]) <= 1000] # TODO: fix this so that you don't lose these datapoints?

    print(len(merged_df))
    merged_df.to_csv('../final_overview_data.csv')

if __name__ == '__main__':
    read_dat_files_to_dataframes('../raw_data')
    combine_and_compute_std('../raw_data')