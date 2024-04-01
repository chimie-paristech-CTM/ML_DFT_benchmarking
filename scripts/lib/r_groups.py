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


def alpha_aromatic(smiles_list):
    value = 0
    for smis in smiles_list:
        for smi in smis.split('.'):
            mol = Chem.MolFromSmiles(smi)
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "*":
                    idx = atom.GetIdx()
                    if idx == 0:
                        atom_alpha = mol.GetAtomWithIdx(idx + 1)
                    else:
                        atom_alpha = mol.GetAtomWithIdx(idx - 1)
                    if atom_alpha.GetIsAromatic():
                        value = 1
                if value:
                    return value
    return value


def alpha_carbonyl(smiles_list):
    value = 0
    for smis in smiles_list:
        for smi in smis.split('.'):
            mol = Chem.MolFromSmiles(smi)
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "*":
                    idx = atom.GetIdx()
                    if idx == 0:
                        atom_alpha = mol.GetAtomWithIdx(idx + 1)
                    else:
                        atom_alpha = mol.GetAtomWithIdx(idx - 1)
                    if atom_alpha.GetHybridization().name == 'SP2':
                        for ngh in atom_alpha.GetNeighbors():
                            if ngh.GetSymbol() == 'O' and len(ngh.GetNeighbors()) == 1:
                                value = 1
                if value:
                    return value
    return value


def alpha_double_bond(smiles_list):
    value = 0
    for smis in smiles_list:
        for smi in smis.split('.'):
            mol = Chem.MolFromSmiles(smi)
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "*":
                    idx = atom.GetIdx()
                    if idx == 0:
                        atom_alpha = mol.GetAtomWithIdx(idx + 1)
                    else:
                        atom_alpha = mol.GetAtomWithIdx(idx - 1)
                    if atom_alpha.GetHybridization().name == 'SP2':
                        for ngh in atom_alpha.GetNeighbors():
                            if ngh.GetSymbol() == 'C' and ngh.GetHybridization().name == 'SP2' and len(ngh.GetNeighbors()) == 2:
                                if not ngh.GetIsAromatic():
                                    value = 1
                if value:
                    return value
    return value


def ring_formation(smiles_list, members):
    value = 0
    for smis in smiles_list:
        for smi in smis.split('.'):
            mol = Chem.MolFromSmiles(smi)
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "*":
                    idx = atom.GetIdx()
                    if idx == 0:
                        atom_alpha = mol.GetAtomWithIdx(idx + 1)
                    else:
                        atom_alpha = mol.GetAtomWithIdx(idx - 1)
                    if atom_alpha.GetHybridization().name == 'SP2':
                        for ngh in atom_alpha.GetNeighbors():
                            if ngh.GetSymbol() == 'C' and ngh.GetHybridization().name == 'SP2' and len(ngh.GetNeighbors()) == 2:
                                if not ngh.GetIsAromatic():
                                    value = 1
                if value:
                    return value
    return value


if __name__ == "__main__":

    df = pd.read_csv('data/data_smiles_curated.csv', sep=';')
    patterns_smiles = {'Diels-Alder': ['C1C=CC=CN1C', 'C1C=CCCC1', 'C1C=CCC=C1', 'N1C=CCC=C1', 'C1N=CCC=C1',
                                       'C1N=NCCC1', 'C1N=NCC=C1', 'C1=CCN=NC1', 'C1C=CC=CN1C', 'C1CCC=NN1',
                                       '[#6,#7]1[#6,#7]=[#6,#7][#6,#7]-,=[#6,#7]-,=[#6,#7]1', 'c12c(CN=NC2)cccc1'],
                       '[3+2]cycloaddition': ['C1ONCC1', 'C1ONCO1', 'N1=NNCC1', 'n1nncc1', 'N1=[N+]CCC1', 'N1=NCCC1', 'N1[N+]CCC1'],
                       'Electrocyclic': ['C1CC=CC=C1', 'C1CCC=CC=C1', 'C=CC=C', 'C1=CC=CCCCNCC1', 'C1C=CC1'],
                       '[3,3]rearrangement': ['C=C=CCC=C', 'C=CCCC=C', 'C=CCCC=O'],}

    r_groups_list = []
    cores = []
    names = []
    subs = []
    patterns = []
    type = []

    df = df.loc[df.Type.isin(patterns_smiles.keys())]

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
                type.append(row.Type)
            else:
                if attempts == len(patt_smis):
                    cores.append('ERROR')
                    r_groups_list.append('ERROR')
                    names.append(row.Name)
                    subs.append('ERROR')
                    patterns.append('ERROR')
                    type.append('ERROR')

            if match:
                break

    df_1 = pd.DataFrame()
    df_1['cores'] = cores
    df_1['r_groups'] = r_groups_list
    df_1['name'] = names
    df_1['subs'] = subs
    df_1['main_core'] = patterns
    df_1['type'] = type

    df_1['std_DFT_forward'] = df_1['name'].apply(lambda x: df.loc[df['Name'] == x].Std_DFT_forward.values[0])
    df_1['Range_DFT_forward'] = df_1['name'].apply(lambda x: df.loc[df['Name'] == x].Range_DFT_forward.values[0])
    df_1['Num_Reacs'] = df_1['name'].apply(
        lambda x: len(df.loc[df['Name'] == x].rxn_smiles.values[0].split('>>')[0].split('.')))
    #df_1['Alpha_aromatic'] = df_1['r_groups'].apply(lambda x: alpha_aromatic(x))
    #df_1['Alpha_carbonyl'] = df_1['r_groups'].apply(lambda x: alpha_carbonyl(x))
    #df_1['Alpha_double_bond'] = df_1['r_groups'].apply(lambda x: alpha_double_bond(x))
    df_1.to_csv('aa.csv')

    df_1.groupby(['main_core'])['std_DFT_forward'].std()
    df_1.groupby(['main_core'])['Range_DFT_forward'].std()
    df_1.groupby(['main_core'])['Range_DFT_forward'].max()
    df_1.groupby(['main_core'])['Range_DFT_forward'].min()
    df_group = pd.DataFrame()

    df_1.groupby(['main_core']).count()


