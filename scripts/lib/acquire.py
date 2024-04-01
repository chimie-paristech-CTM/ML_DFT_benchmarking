#!/usr/bin/python
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance
import random


def get_seeds(seed=7, k=5):
    random.seed(seed)
    return random.sample(range(1000, 9999), k=k)


def upper_confidence_bound(predictions, variance, beta=2):
    """ Upper Confidence Bound acquisition function"""

    return predictions + beta * variance


def iterative_sampling(df_pool, logger, column='ucb', initial_sample=10, cutoff=0.5):
    """ This function takes a random samples of rxns and filters until all the rxns are above a specific treshold """

    df_pool_sorted = df_pool.sort_values(by=[column], ascending=False)

    # create the final dataset of reactions, the temporal dataset of reactions and the final dataset with the fingerprints.
    rxns = pd.DataFrame()
    rxns_temp = df_pool_sorted.iloc[:initial_sample]
    rxns = pd.concat([rxns, rxns_temp])
    steps = 0
    rxns_fps = pd.DataFrame()
    rxns_explored = initial_sample
    fixed_rxns = 0

    while True:

        # Drop the temporal_reactions from the initial dataset of 1 million
        rxns_temp_idx = [df_pool_sorted[df_pool_sorted.rxn_smiles == row.rxn_smiles].index[0] for row in rxns_temp.itertuples()]
        df_pool_sorted = df_pool_sorted.drop(labels=rxns_temp_idx)

        # Add the temp encoded reactions to the final dataset of fingerprints
        rxns_fps = pd.concat([rxns_fps, rxns_temp], ignore_index=True)

        # Elements that for the cutoff should be dropped
        to_remove = []

        # Calculating all the distances ... squared matrix ... but I only need a half
        # if an element, will be removed, it is not necesary to calculate the distance
        # and for the elements that have passed to the next step, it is also not necesary to calculate the distance
        for i in range(len(rxns)):
            if i in to_remove:
                continue
            for j in range(fixed_rxns, len(rxns)):
                if (j > i) & (j not in to_remove):
                    cos_d = cosine_distance(rxns_fps.iloc[i].Fingerprints, rxns_fps.iloc[j].Fingerprints)
                    if cos_d < cutoff:
                        to_remove.append(j)

        # Eliminating duplicated elements
        to_remove = set(to_remove)
        to_remove_rxns = [rxns.iloc[i].name for i in to_remove]

        # Eliminating reactions
        rxns_fps = rxns_fps.drop(labels=to_remove)
        rxns = rxns.drop(to_remove_rxns)

        # Check if I need to take more samples
        fixed_rxns = len(rxns)
        next_samples = initial_sample - fixed_rxns

        if next_samples == 0:
            logger.info(f"All the reactions are above {cutoff}.")
            break
        if next_samples > len(df_pool):
            logger.info(f"Just could find {len(rxns)} that are above {cutoff}.")
            break

        rxns_temp = df_pool_sorted.iloc[:next_samples]
        rxns = pd.concat([rxns, rxns_temp])
        steps += 1
        rxns_explored += next_samples

    return rxns


