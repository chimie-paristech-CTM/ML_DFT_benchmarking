#!/usr/bin/python
from drfp import DrfpEncoder
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def encode(smile_reaction, rad, nbits):
    """ Encode a smile reaction """

    return DrfpEncoder.encode(smile_reaction, radius=rad, n_folded_length=nbits)


def get_fingerprints_DRFP(df, rad=3, nbits=2048):

    rxns = df.rxn_smiles.tolist()
    all_rxn_fps = []

    for rxn in rxns:
        fps = encode(rxn, rad=rad, nbits=nbits)
        all_rxn_fps.append(fps)
    
    dr = pd.DataFrame([s for s in all_rxn_fps], columns=['Fingerprints'])
    dr[['Name', 'Std_DFT_forward']] = df[['Name', 'Std_DFT_forward']]

    return dr


def get_difference_fingerprint(rxn_smiles, rad, nbits):
    """
    Get difference Morgan fingerprint between reactants and products

    Args:
        rxn_smiles (str): the full reaction SMILES
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        np.array: the difference fingerprint
    """
    reactants = rxn_smiles.split('>>')[0]
    products = rxn_smiles.split('>>')[-1]

    reactants_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactants), radius=rad, nBits=nbits))
    products_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(products), radius=rad, nBits=nbits))

    return reactants_fp - products_fp


def get_fingerprints_Morgan(df, rad, nbits):

    df['Fingerprints'] = df['rxn_smiles'].apply(lambda x: get_difference_fingerprint(x, rad, nbits))

    return df[['Name', 'Fingerprints', 'Std_DFT_forward']]
