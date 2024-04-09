import pandas as pd
from rdkit import Chem
import itertools


# Functions to decorate the cores


# substituent list
subs_list_LR = ['c1ccccc1', 'C=O', 'O', 'C#N', 'C(=O)OC', 'OC(=O)C', 'C(C)(C)C', None]

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

# Functions to generate fused cycles


def bicyclic_core(rxn, sub):

    reac, prod = rxn.split('>>')
    reac = reac.replace('Mg', sub)
    prod = prod.replace('Mg', f"({sub}2)")
    mol_reac = Chem.MolFromSmiles(reac)
    mol_prod = Chem.MolFromSmiles(prod)
    Chem.SanitizeMol(mol_reac)
    Chem.SanitizeMol(mol_prod)
    reac = Chem.MolToSmiles(mol_reac)
    prod = Chem.MolToSmiles(mol_prod)
    rxn = f"{reac}>>{prod}"

    return rxn


def tricyclic_core(rxn, sub):

    reac, prod = rxn.split('>>')
    reac = reac.replace('Mg', sub)
    prod = prod.replace('Mg', f"{sub}")
    mol_reac = Chem.MolFromSmiles(reac)
    mol_prod = Chem.MolFromSmiles(prod)
    Chem.SanitizeMol(mol_reac)
    Chem.SanitizeMol(mol_prod)
    reac = Chem.MolToSmiles(mol_reac)
    prod = Chem.MolToSmiles(mol_prod)
    rxn = f"{reac}>>{prod}"
    return rxn


if __name__ == '__main__':

    rxns = []
    for sub in ['C', 'C([*])C', 'CCC', 'CC([*])', 'C(=O)C(=O)', 'CC(=O)O']:
        rxn = bicyclic_core('C=CMgC=CC=C', 'C1MgC2C=CCC1', sub)
        rxns.append(rxn)
        rxn = tricyclic_core('C1=CC=CMgC=CMg1', 'C12C=CC3MgC3C1Mg2', sub)
        rxns.append(rxn)
    rxns = list(set(rxns))

