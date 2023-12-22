#!/usr/bin/python
import pandas as pd
from argparse import ArgumentParser
from lib.utils import create_logger
from lib.final_functions import get_optimal_parameters_knn_fp, get_optimal_parameters_knn
from lib.final_functions import get_cross_val_accuracy_linear_regression
from lib.final_functions import get_cross_val_accuracy_knn_fp, get_cross_val_accuracy_knn
from lib.final_functions import get_optimal_parameters_rf, get_optimal_parameters_rf_fp
from lib.final_functions import get_cross_val_accuracy_rf, get_cross_val_accuracy_rf_fps
from lib.final_functions import get_optimal_parameters_xgboost, get_optimal_parameters_xgboost_fp
from lib.final_functions import get_cross_val_accuracy_xgboost, get_cross_val_accuracy_xgboost_fps
from lib.fingerprints import get_fingerprints_DRFP, get_fingerprints_Morgan


parser = ArgumentParser()
parser.add_argument('--csv-file', type=str, default='../data_smiles_curated.csv',
                    help='path to file containing the rxn-smiles')
parser.add_argument('--input-file', type=str, default='../final_overview_data.csv',
                    help='path to the input file')
parser.add_argument('--split_dir', type=str, default=None,
                    help='path to the folder containing the requested splits for the cross validation')
parser.add_argument('--n-fold', type=int, default=4,
                    help='the number of folds to use during cross validation')
# interactive way
parser.add_argument("--mode", default='client', action="store", type=str)
parser.add_argument("--host", default='127.0.0.1', action="store", type=str)
parser.add_argument("--port", default=57546, action="store", type=int)


if __name__ == '__main__':
    # set up
    args = parser.parse_args()
    logger = create_logger()
    df = pd.read_csv(args.input_file)
    df_rxn_smiles = pd.read_csv(args.csv_file, sep=';')
    df_fps_drfp = get_fingerprints_DRFP(df_rxn_smiles)
    df_fps_morgan = get_fingerprints_Morgan(df_rxn_smiles, rad=2, nbits=2048)
    n_fold = args.n_fold
    split_dir = args.split_dir

    # linear regression
    # get_cross_val_accuracy_linear_regression(df, logger, n_fold, split_dir)

    # KNN desc
    optimal_parameters_knn = get_optimal_parameters_knn(df, logger, max_eval=64)
    get_cross_val_accuracy_knn(df, logger, n_fold, optimal_parameters_knn, split_dir)

    # KNN fingerprints
    logger.info("Fingerprint: DRFP (radius = 2)")
    optimal_parameters_knn_fps = get_optimal_parameters_knn_fp(df_fps_drfp, logger, max_eval=64)
    get_cross_val_accuracy_knn_fp(df_fps_drfp, logger, n_fold, optimal_parameters_knn_fps, split_dir)

    logger.info("Fingerprint: Morgan (radius = 2)")
    optimal_parameters_knn_fps = get_optimal_parameters_knn_fp(df_fps_morgan, logger, max_eval=64)
    get_cross_val_accuracy_knn_fp(df_fps_morgan, logger, n_fold, optimal_parameters_knn_fps, split_dir)

    # RF descriptors
    optimal_parameters_rf = get_optimal_parameters_rf(df, logger, max_eval=64)
    get_cross_val_accuracy_rf(df, logger, n_fold, optimal_parameters_rf, split_dir)

    # RF fingerprints
    logger.info("Fingerprint: DRFP (radius = 2)")
    optimal_parameters_rf_fps = get_optimal_parameters_rf_fp(df_fps_drfp, logger, max_eval=64)
    get_cross_val_accuracy_rf_fps(df_fps_drfp, logger, n_fold, optimal_parameters_rf_fps, split_dir)

    logger.info("Fingerprint: Morgan (radius = 2)")
    optimal_parameters_rf_fps = get_optimal_parameters_rf_fp(df_fps_morgan, logger, max_eval=64)
    get_cross_val_accuracy_rf_fps(df_fps_morgan, logger, n_fold, optimal_parameters_rf_fps, split_dir)

    # XGboost
    optimal_parameters_xgboost = get_optimal_parameters_xgboost(df, logger, max_eval=128)
    get_cross_val_accuracy_xgboost(df, logger, n_fold, optimal_parameters_xgboost, split_dir)

    # XGboost fingerprints
    logger.info("Fingerprint: DRFP (radius = 2)")
    optimal_parameters_xgboost_fp = get_optimal_parameters_xgboost_fp(df_fps_drfp, logger, max_eval=128)
    get_cross_val_accuracy_xgboost_fps(df_fps_drfp, logger, n_fold, optimal_parameters_xgboost_fp, split_dir)

    logger.info("Fingerprint: Morgan (radius = 2)")
    optimal_parameters_xgboost_fp = get_optimal_parameters_xgboost_fp(df_fps_morgan, logger, max_eval=128)
    get_cross_val_accuracy_xgboost_fps(df_fps_morgan, logger, n_fold, optimal_parameters_xgboost_fp, split_dir)

