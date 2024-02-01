import pandas as pd
from rdkit import Chem
import itertools
import re
import numpy as np

# substituent list
subs_list_LR = ['C', 'F', 'OC', 'O', 'C#N', 'C(=O)OC', 'OC(=O)C', 'NC', 'SC', None]

# make easy the replacement
labels = ['[Os]', '[Ir]', '[Pt]', '[Au]']


def single_edit_mol(mol, label, subs):
    if subs != None:
        mod_mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles(label), Chem.MolFromSmiles(subs), replaceAll=True)[0]
    else:
        mod_mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles(label))
    return mod_mol


def modify_mol(dipole, subs_comb_LR, labels):
    mol = Chem.MolFromSmiles(dipole)
    for i, subs in enumerate(subs_comb_LR):
        mol = single_edit_mol(mol, labels[i], subs)

    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))


def unmap_smiles(smiles):
    """Unmap atoms of SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]

    return Chem.MolToSmiles(mol)


def create_combination(rxn_smiles, type):

    reacs, prods = rxn_smiles.split(">>")

    r_groups = 0
    for label in labels:
        if label in reacs:
            r_groups += 1

    substituent_combs = itertools.product(subs_list_LR, repeat=r_groups)

    new_reacs, new_prods = [], []
    for subs_comb in substituent_combs:
        if r_groups > 2:
            if len(set(subs_comb)) == 1:
                continue
            if subs_comb.count(None) < r_groups - 2:   # to avoid a combinatorial explosion
                continue
        new_reacs.append(modify_mol(reacs, subs_comb, labels))
        new_prods.append(modify_mol(prods, subs_comb, labels))

    new_rxns = []
    for reac, prod in zip(new_reacs, new_prods):
        reac_smi = Chem.MolToSmiles(reac)
        prod_smi = Chem.MolToSmiles(prod)
        new_rxns.append(f"{reac_smi}>>{prod_smi}")

    df_rxns = pd.DataFrame()
    df_rxns['Type'] = len(new_rxns) * [type]
    df_rxns['rxn_smiles'] = new_rxns

    return df_rxns


if __name__ == '__main__':

    df = pd.read_csv('../hypothetical_space_core.csv', sep=';')
    df_space = pd.DataFrame(columns=['Type', 'rxn_smiles'])
    for row in df.itertuples():
        df_space = pd.concat([df_space, create_combination(row.rxn_smiles, row.Type)], axis=0, ignore_index=True)
    df_space.to_csv('../hypothetical_chemical_space.csv')