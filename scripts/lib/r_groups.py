import pandas as pd
from rdkit import Chem
import pandas as pd
from rdkit.Chem import rdRGroupDecomposition


def unmap_smiles(smiles):
    """Unmap atoms of SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]

    return Chem.MolToSmiles(mol)


def extract_r_groups(patt_smi, smi):
    mol = Chem.MolFromSmiles(smi)
    patt = Chem.MolFromSmarts(patt_smi)
    if mol.HasSubstructMatch(patt):
        gs, _ = rdRGroupDecomposition.RGroupDecompose([patt], [mol], asSmiles=True)
        subs = [unmap_smiles(gs[0][key]) for key in gs[0].keys() if key != 'Core']
        core = gs[0]['Core']
        return subs, core
    else:
        return None, None


if __name__ == "__main__":

    df = pd.read_csv('data/data_smiles_curated.csv', sep=';')
    patterns_smiles = {'Diels-Alder': ['C1C=CC=CN1C', 'C1C=CCCC1', 'C1C=CCC=C1', 'N1C=CCC=C1', 'C1N=CCC=C1',
                                       'C1N=NCCC1', 'C1N=NCC=C1', 'C1=CCN=NC1', 'c12c(CN=NC2)cccc1', 'C1C=CC=CN1C',
                                       'C1CCC=NN1']}
    r_groups_list = []
    cores = []
    names = []
    subs = []
    patterns = []

    for row in df.itertuples():
        rxn = row.rxn_smiles
        patt_smis = patterns_smiles[row.Type]
        rs, ps = rxn.split('>>')
        attempts = 0
        for patt_smi in patt_smis:
            match = 0
            attempts += 1
            r_groups, core = extract_r_groups(patt_smi, ps)
            if r_groups:
                match = 1
                cores.append(core)
                r_groups_list.append(r_groups)
                names.append(row.Name)
                subs.append(len(r_groups))
                patterns.append(patt_smi)
            else:
                if attempts == len(patt_smis):
                    cores.append('ERROR')
                    r_groups_list.append('ERROR')
                    names.append(row.Name)
                    subs.append('ERROR')
                    patterns.append('ERROR')

            if match:
                break

    df_1 = pd.DataFrame()
    df_1['cores'] = cores
    df_1['r_groups'] = r_groups_list
    df_1['name'] = names
    df_1['subs'] = subs
    df_1['main_core'] = patterns
    df_1.to_csv('aa.csv')

    df_1['std_DFT_forward'] = df_1['name'].apply(lambda x: df.loc[df['Name'] == x].Std_DFT_forward.values[0])

    df_1.groupby(['main_core'])['std_DFT_forward'].std()
    df_1.groupby(['main_core']).count()


