import pandas as pd
from rdkit import Chem
import re


def bicyclic_core(reac, prod, sub):

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


def tricyclic_core(reac, prod, sub):

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

