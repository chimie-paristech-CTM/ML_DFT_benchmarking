import os
import pandas as pd
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression
import numpy as np

parser = ArgumentParser()
parser.add_argument('--csv_file_gap', type=str, default='../data/homo_lumo_gap.csv',
                    help='path to csv file containing gaps')
parser.add_argument('--csv_file', type=str, default='../data/data_smiles_curated.csv',
                    help='path to csv file')


if __name__ == '__main__':
    args = parser.parse_args()
    data = pd.read_csv(args.csv_file, sep=';', index_col=0)
    data_gaps = pd.read_csv(args.csv_file_gap, index_col=0)
    data['gap_TS'] = data['Name'].apply(lambda x: data_gaps.loc[data_gaps['name'] == f"{x}TS"].gap.values[0])
    X, y = np.array(data[['gap_TS']]), np.array(data[['Std_DFT_forward']])
    model = LinearRegression()
    model.fit(X, y.ravel())
    print(model.score(X, y.ravel())) # 0.24
    print(data.corr(numeric_only=True)['gap_TS']['Std_DFT_forward']**2) # 0.24

    data_DA = data.loc[data.Type == 'Diels-Alder']
    print(data_DA.corr(numeric_only=True)['gap_TS']['Std_DFT_forward'] ** 2) # 0.12





