#!/usr/bin/python
import pandas as pd
from argparse import ArgumentParser
from lib.utils import create_logger
from lib.final_functions import get_optimal_parameters_knn_fp, get_optimal_parameters_knn
from lib.final_functions import get_cross_val_accuracy_knn_fp, get_cross_val_accuracy_knn
from lib.final_functions import get_optimal_parameters_rf, get_optimal_parameters_rf_fp
from lib.final_functions import get_cross_val_accuracy_rf, get_cross_val_accuracy_rf_fps
from lib.final_functions import get_optimal_parameters_xgboost, get_optimal_parameters_xgboost_fp
from lib.final_functions import get_cross_val_accuracy_xgboost, get_cross_val_accuracy_xgboost_fps
from lib.fingerprints import get_fingerprints_DRFP, get_fingerprints_Morgan
from lib.final_functions import get_nested_cross_val_accuracy_knn_fps
from lib.final_functions import get_nested_cross_val_accuracy_rf_fps
from lib.final_functions import get_nested_cross_val_accuracy_xgboost_fps
from lib.final_functions import get_cross_val_accuracy_means_values


parser = ArgumentParser()
parser.add_argument('--csv_file', type=str, default='../data/data_smiles_curated.csv',
                    help='path to file containing the rxn-smiles')
parser.add_argument('--input_file', type=str, default='../data/final_overview_data.csv',
                    help='path to the input file')
parser.add_argument('--n_fold', type=int, default=6,
                    help='the number of folds to use during cross validation')
parser.add_argument('--final_cv', default=False, action="store_true", help='final cross validation')


if __name__ == '__main__':
    # set up
    args = parser.parse_args()
    logger = create_logger()
    df = pd.read_csv(args.input_file)
    df_rxn_smiles = pd.read_csv(args.csv_file, sep=';')
    n_fold = args.n_fold

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    if args.final_cv:
        df_fps_morgan = get_fingerprints_Morgan(df_rxn_smiles, rad=2, nbits=2048)
        optimal_parameters_rf_fps = get_optimal_parameters_rf_fp(df_fps_morgan, logger, max_eval=64)
        get_cross_val_accuracy_rf_fps(df_fps_morgan, logger, n_fold, optimal_parameters_rf_fps,)
    else:
        get_cross_val_accuracy_means_values(df, logger, n_fold)

        # # fingerprints
        logger.info(f"Fingerprints")
        nbits = [16, 32, 64, 128, 256, 512, 1024, 2048]
        rads = [1, 2, 3]
        for nbit in nbits:
            for rad in rads:
                df_fps_drfp = get_fingerprints_DRFP(df_rxn_smiles, rad=rad, nbits=nbit)
                df_fps_morgan = get_fingerprints_Morgan(df_rxn_smiles, rad=rad, nbits=nbit)

                # KNN fingerprints
                logger.info(f"Fingerprint: DRFP (radius={rad}, nbits={nbit})")
                get_nested_cross_val_accuracy_knn_fps(df_fps_drfp, logger, n_fold)

                logger.info(f"Fingerprint: Morgan (radius={rad}, nbits={nbit})")
                get_nested_cross_val_accuracy_knn_fps(df_fps_morgan, logger, n_fold)

                # RF fingerprints
                logger.info(f"Fingerprint: DRFP (radius={rad}, nbits={nbit})")
                get_nested_cross_val_accuracy_rf_fps(df_fps_drfp, logger, n_fold)

                logger.info(f"Fingerprint: Morgan (radius={rad}, nbits={nbit})")
                get_nested_cross_val_accuracy_rf_fps(df_fps_morgan, logger, n_fold)

                # XGboost fingerprints
                logger.info(f"Fingerprint: DRFP (radius={rad}, nbits={nbit})")
                get_nested_cross_val_accuracy_xgboost_fps(df_fps_drfp, logger, n_fold)

                logger.info(f"Fingerprint: Morgan (radius={rad}, nbits={nbit})")
                get_nested_cross_val_accuracy_xgboost_fps(df_fps_morgan, logger, n_fold)



