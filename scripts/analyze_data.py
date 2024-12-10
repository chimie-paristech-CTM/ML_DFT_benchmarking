import os
import pandas as pd
import re
from glob import glob
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--raw_data', type=str, default='../data/raw_data',
                    help='path to folder containing the .dat files')
parser.add_argument('--iter', type=str, help='initial round, first, second ...')
parser.add_argument('--generate_initial_data', default=False, action="store_true", help='From XYZ to fingerprints')
parser.add_argument('--pool_file', type=str, default=None,
                    help='path to the input file')


def combine_and_compute_std(folder_path, iter, pool_file = None):
    all_dfs = []
    # Loop through all .csv files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path, index_col=0)
            all_dfs.append(df)

    merged_df = all_dfs[0]
    if ['Type', 'REF-forward', 'REF-reverse', 'REF-reaction'] in merged_df.columns.tolist():
        for df in all_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on=['Name', 'Type', 'REF-forward', 'REF-reverse', 'REF-reaction'])
    else:
        for df in all_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on=['Name'])

    
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
        merged_df = merged_df.loc[abs(merged_df[column]) <= 1000]

    merged_df.to_csv(f'{folder_path}/final_overview_data_{iter}.csv')

    if iter != 'initial' and pool_file is not None:
        add_rxn_smiles_iter(f'{folder_path}/final_overview_data_{iter}.csv', pool_file)


def generate_file_fps(csv_file, xyz_files):

    df = pd.read_csv(csv_file, index_col=0)
    df['rxn_smiles'] = df['Name'].apply(lambda x: generate_rxn_smiles(x, xyz_files))
    df.to_csv('data_smiles.csv')


def generate_rxn_smiles(idx, directory):

    products = glob(f"{directory}/{idx}P*")
    product_smi = [generate_smiles_from_xyz(product) for product in products]
    reactants = glob(f"{directory}/{idx}R*")
    reactants_smi = [generate_smiles_from_xyz(reactant) for reactant in reactants]
    reactants_smi = ".".join(reactants_smi)
    products_smi = ".".join(product_smi)
    rxn_smi = f"{reactants_smi}>>{products_smi}"

    return rxn_smi


def generate_smiles_from_xyz(xyz_file):

    raw_mol = Chem.MolFromXYZFile(xyz_file)
    mol = Chem.Mol(raw_mol)

    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except ValueError:
        return "ERROR"

    mol = Chem.RemoveHs(mol)
    smi = Chem.MolToSmiles(mol)

    return smi


def add_rxn_smiles_iter(csv_file, pool_file):

    df_pool = pd.read_csv(pool_file, index_col=0)
    df_new_rxns = pd.read_csv(csv_file, index_col=0)
    rxns_smiles_list = []
    for row in df_new_rxns.itertuples():
        idx = int(row.Name.split('_')[-1])
        rxns_smiles_list.append(df_pool.loc[idx].rxn_smiles)
    df_new_rxns['rxn_smiles'] = rxns_smiles_list
    df_new_rxns.to_csv('../data/data_augmentation.csv')


if __name__ == '__main__':

    args = parser.parse_args()

    combine_and_compute_std(args.raw_data, args.iter, args.pool_file)

    if args.generate_initial_data:
        generate_file_fps('../data/final_overview_data.csv', '../data/XYZ_files')
