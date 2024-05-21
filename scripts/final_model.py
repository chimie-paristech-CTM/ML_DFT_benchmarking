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
parser.add_argument('--new_data_file', type=str, default='../data/data_augmentation.csv',
                    help='path to the additional file')
parser.add_argument('--iteration', type=int, help='iteration')
parser.add_argument('--seed', type=int, default=10, help='initial seed')
parser.add_argument('--beta', type=float, default=1.2, help='beta value for UCB acquisition function')
parser.add_argument('--final_dir', type=str, default='../data/autodE_input',
                    help='path to the folder with the autodE input')
parser.add_argument('--cutoff', type=float, default=0.75,
                    help='cutoff of similarity')
parser.add_argument('--conda_env', type=str, default='autodE',
                    help='conda environment of autodE package')
parser.add_argument('--new_data', type=int, default = 10,
                    help='Next data')
parser.add_argument('--selective_sampling', type=str, default=None,
                    help='Sample one kind of reaction')
parser.add_argument('--selective_sampling_data', type=int, default=5,
                    help='Next data of that kind of sampling')


if __name__ == "__main__":

    args = parser.parse_args()
    logger = create_logger(name='next_data')
    df_train = pd.read_csv(args.train_file, sep=';', index_col=0)
    df_pool = pd.read_csv(args.pool_file, index_col=0)
    if args.iteration > 1:
        df_additional = pd.read_csv(args.new_data_file, index_col=0)
        df_train = pd.concat([df_train, df_additional], ignore_index=True)
        df_train.to_csv(f'../data/data_smiles_curated_{args.iteration}.csv')
    df_train_fps = get_fingerprints_Morgan(df_train, rad=2, nbits=2048)
    df_pool_fps = get_fingerprints_Morgan(df_pool, rad=2, nbits=2048, labeled=False)

    logger.info(f"Iteration: {args.iteration}, beta: {args.beta}")
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    seeds = get_seeds(args.seed, 1)

    logger.info(f"seeds: {seeds}")

    seed_max_value = [0, 0]
    for seed in seeds:
        model = RForest(n_estimators=150, max_features=0.4, min_samples_leaf=1, seed=seed)
        model.train(train=df_train_fps)
        preds, vars = model.get_means_and_vars(df_pool_fps)
        ucb = upper_confidence_bound(preds, vars, args.beta)
        df_pool[f'prediction {seed}'] = preds
        df_pool[f'variance {seed}'] = vars
        df_pool[f'ucb {seed}'] = ucb
        if ucb.max() > seed_max_value[1]:
            seed_max_value[0] = seed
            seed_max_value[1] = ucb.max()

    df_pool.to_csv(f'chemical_space_ucb_{args.iteration}.csv')

    if args.selective_sampling:
        if args.selective_sampling_data >= args.new_data:
            df_pool = df_pool.loc[df_pool.Type == args.selective_sampling]
            next_rxns = iterative_sampling(df_pool, logger, column=f"ucb {seed_max_value[0]}", initial_sample=args.new_data, cutoff=args.cutoff)
        else:
            df_pool_selective = df_pool.loc[df_pool.Type == args.selective_sampling]
            df_pool = df_pool.loc[df_pool.Type != args.selective_sampling]
            next_rxns_selective = iterative_sampling(df_pool_selective, logger, column=f"ucb {seed_max_value[0]}",
                                                     initial_sample=args.selective_sampling_data, cutoff=args.cutoff)
            next_rxns = iterative_sampling(df_pool, logger, column=f"ucb {seed_max_value[0]}", initial_sample= (args.new_data - args.selective_sampling_data), cutoff=args.cutoff)
            next_rxns = pd.concat([next_rxns_selective, next_rxns])
    else:
        next_rxns = iterative_sampling(df_pool, logger, column=f"ucb {seed_max_value[0]}", initial_sample=args.new_data, cutoff=args.cutoff)

    logger.info(f"Next reactions:")
    logger.info(f"{next_rxns[['Type', 'rxn_smiles', f'prediction {seed_max_value[0]}', f'variance {seed_max_value[0]}', f'ucb {seed_max_value[0]}']].to_string()}")

    create_input(next_rxns, args.final_dir, args.conda_env)

    df_pool.drop(index=next_rxns.index, inplace=True)
    df_pool.to_csv(f'../data/hypothetical_chemical_space_iter_{args.iteration}.csv')

    df_preds = pd.DataFrame()
    df_preds['Std_DFT_forward'] = df_pool[f'prediction {seed_max_value[0]}']
    df_preds['Vars'] = df_pool[f'variance {seed_max_value[0]}']
    df_preds['UCB'] = df_pool[f'ucb {seed_max_value[0]}']
    df_preds.to_csv(f'Prediction_iter_{args.iteration}.csv', sep=';')

