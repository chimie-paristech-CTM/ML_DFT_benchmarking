#!/usr/bin/python
from drfp import DrfpEncoder
import pandas as pd


def encode(smile_reaction, mapping_bool=False):
    """ Encode a smile reaction """

    return DrfpEncoder.encode(smile_reaction, mapping=mapping_bool, radius=2)


def get_fingerprints_all_rxn(df):

    rxns = df.rxn_smiles.tolist()
    all_rxn_fps = []

    for rxn in rxns:
        fps = encode(rxn)
        all_rxn_fps.append(fps)
    
    dr = pd.DataFrame([s for s in all_rxn_fps], columns=['Fingerprints'])
    dr[['Name', 'Std_DFT_forward']] = df[['Name', 'Std_DFT_forward']]

    return dr
