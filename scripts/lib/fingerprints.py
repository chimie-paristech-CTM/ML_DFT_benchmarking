#!/usr/bin/python
from drfp import DrfpEncoder
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from collections import Counter
import numpy as np

def encode(smile_reaction, rad, nbits):
    """ Encode a smile reaction """

    return DrfpEncoder.encode(smile_reaction, radius=rad, n_folded_length=nbits)


def get_fingerprints_DRFP(df, rad=3, nbits=2048):

    rxns = df.rxn_smiles.tolist()
    all_rxn_fps = []

    for rxn in rxns:
        fps = encode(rxn, rad=rad, nbits=nbits)
        ring_info = count_rings(rxn)
        all_rxn_fps.append([np.concatenate((fps[0], ring_info))])
    
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

    ring_info = count_rings(rxn_smiles)
    d_fps = reactants_fp - products_fp

    return np.concatenate((d_fps, ring_info))


def get_fingerprints_Morgan(df, rad, nbits, labeled=True):

    df['Fingerprints'] = df['rxn_smiles'].apply(lambda x: get_difference_fingerprint(x, rad, nbits))

    if labeled:
        return df[['Name', 'Fingerprints', 'Std_DFT_forward']]
    else:
        return df[['Fingerprints']]


def count_rings(rxn):
    """
    Encode information related to the formation of new rings.
    :param rxn: smiles reaction
    :return: one-hot enconding of new ring formed

    """

    reac, prod = rxn.split('>>')
    reac_mol = Chem.MolFromSmiles(reac)
    prod_mol = Chem.MolFromSmiles(prod)
    ssr_reac = Chem.GetSSSR(reac_mol)
    ssr_prod = Chem.GetSSSR(prod_mol)

    membered_rings_reac = [len(list(ssr_reac[i])) for i in range(len(ssr_reac))]
    membered_rings_prod = [len(list(ssr_prod[i])) for i in range(len(ssr_prod))]

    counter_reac = Counter(membered_rings_reac)
    counter_prod = Counter(membered_rings_prod)

    ohe = np.array([0, 0, 0, 0, 0, 0])
    map = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}

    for member in [3, 4, 5, 6, 7, 8]:
        if member in counter_prod and member not in counter_reac:
            ohe[map[member]] = 1
        elif member in counter_prod and member in counter_reac:
            if counter_prod[member] > counter_reac[member]:
                ohe[map[member]] = 1
            else:
                ohe[map[member]] = 0
        else:
            ohe[map[member]] = 0

    return ohe
