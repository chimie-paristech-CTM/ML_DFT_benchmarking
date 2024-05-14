import pandas as pd
from argparse import ArgumentParser
from lib.fingerprints import get_fingerprints_Morgan
from lib.acquire import upper_confidence_bound, iterative_sampling, get_seeds
from lib.RForest import RForest
from lib.utils import create_logger
from lib.input_autode import create_input


parser = ArgumentParser()
parser.add_argument('--train_file', type=str, default='../data/data_smiles_curated.csv',
                    help='path to file containing the rxn-smiles')
parser.add_argument('--pool_file', type=str, default='../data/hypothetical_chemical_space.csv',
                    help='path to the input file')
parser.add_argument('--iteration', type=int, help='iteration')
parser.add_argument('--seed', type=int, default=10, help='initial seed')
parser.add_argument('--beta', type=float, default=1.2, help='beta value for UCB acquisition function')
parser.add_argument('--final_dir', type=str, default='../data/autodE_input',
                    help='path to the folder with the autodE input')
parser.add_argument('--conda_env', type=str, default='autodE',
                    help='conda environment of autodE package')


if __name__ == "__main__":

    args = parser.parse_args()
    logger = create_logger(name='next_data')
    df_train = pd.read_csv(args.train_file, sep=';', index_col=0)
    df_pool = pd.read_csv(args.pool_file, index_col=0)
    df_train_fps = get_fingerprints_Morgan(df_train, rad=2, nbits=2048)
    df_pool_fps = get_fingerprints_Morgan(df_pool, rad=2, nbits=2048, labeled=False)

    logger.info(f"Iteration: {args.iteration}, beta: {args.beta}")
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    seeds = get_seeds(args.seed, 5)

    logger.info(f"seeds: {seeds}")

    seed_max_value = [0, 0]
    for seed in seeds:
        model = RForest(n_estimators=150, max_features=0.4, min_samples_leaf=1, seed=seed, n_jobs=4)
        model.train(train=df_train_fps)
        preds, vars = model.get_means_and_vars(df_pool_fps)
        ucb = upper_confidence_bound(preds, vars, args.beta)
        df_pool[f'prediction {seed}'] = preds
        df_pool[f'variance {seed}'] = vars
        df_pool[f'ucb {seed}'] = ucb
        if ucb.max() > seed_max_value[1]:
            seed_max_value[0] = seed
            seed_max_value[1] = ucb.max()

    df_pool.to_csv('chemical_space_ucb.csv')

    next_rxns = iterative_sampling(df_pool, logger, column=f"ucb {seed_max_value[0]}", cutoff=0.75)
    logger.info(f"Next reactions:")
    logger.info(f"{next_rxns[['Type', 'rxn_smiles', f'prediction {seed_max_value[0]}', f'variance {seed_max_value[0]}', f'ucb {seed_max_value[0]}']].to_string()}")

    create_input(next_rxns, args.final_dir, args.conda_env)

    df_preds = pd.DataFrame()
    df_preds['Std_DFT_forward'] = df_pool[f'prediction {seed_max_value[0]}']
    df_preds['Vars'] = df_pool[f'variance {seed_max_value[0]}']
    df_preds['UCB'] = df_pool[f'ucb {seed_max_value[0]}']
    df_preds.to_csv(f'Prediction_iter_{args.iteration}.csv', sep=';')

