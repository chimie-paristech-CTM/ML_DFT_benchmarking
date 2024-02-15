import pandas as pd
from argparse import ArgumentParser
from lib.fingerprints import get_fingerprints_Morgan
from lib.acquire import upper_confidence_bound, iterative_sampling
from lib.RForest import RForest
from lib.utils import create_logger

parser = ArgumentParser()
parser.add_argument('--train_file', type=str, default='../data/data_smiles_curated.csv',
                    help='path to file containing the rxn-smiles')
parser.add_argument('--pool_file', type=str, default='../data/hypothetical_chemical_space.csv',
                    help='path to the input file')
parser.add_argument('--iteration', type=int, help='iteration')
parser.add_argument('--beta', type=float, default=1.2, help='beta value for UCB acquisition function')


if __name__ == "__main__":

    args = parser.parse_args()
    logger = create_logger(name='next_data')
    df_train = pd.read_csv(args.train_file, sep=';', index_col=0)
    df_pool = pd.read_csv(args.pool_file, index_col=0)
    df_train_fps = get_fingerprints_Morgan(df_train, rad=2, nbits=2048)
    df_pool_fps = get_fingerprints_Morgan(df_pool, rad=2, nbits=2048, labeled=False)

    model = RForest(n_estimators=100, max_features=0.3, min_sample_leaf=1)
    model.train(train=df_train_fps)
    preds, vars = model.get_means_and_vars(df_pool_fps)
    ucb = upper_confidence_bound(preds, vars, args.beta)
    df_pool['prediction'] = preds
    df_pool['variance'] = vars
    df_pool['ucb'] = ucb
    df_pool['ucb 1.8'] = upper_confidence_bound(preds, vars, 1.8)
    df_pool['ucb 0.2'] = upper_confidence_bound(preds, vars, 0.2)
    df_pool.to_csv('chemical_space_ucb.csv')

    logger.info(f"Iteration: {args.iteration}, beta: {args.beta}")
    next_rxns = iterative_sampling(df_pool, logger)
    logger.info(f"Next reactions:")
    logger.info(f"{next_rxns[['Type', 'rxn_smiles', 'prediction', 'variance', 'ucb']].to_string()}")

    df_preds = pd.DataFrame()
    df_preds['Std_DFT_forward'] = preds.reshape(-1)
    df_preds['Vars'] = vars.reshape(-1)
    df_preds.to_csv(f'Prediction_iter_{args.iteration}.csv', sep=';')

