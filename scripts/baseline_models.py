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
from lib.nested_cross_val import nested_cross_val_fp
from lib.bayesian_opt import objective_knn_fp, objective_rf_fp, objective_xgboost_fp
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from hyperopt import hp


parser = ArgumentParser()
parser.add_argument('--csv_file', type=str, default='../data/data_smiles_curated.csv',
                    help='path to file containing the rxn-smiles')
parser.add_argument('--input_file', type=str, default='../data/final_overview_data.csv',
                    help='path to the input file')
parser.add_argument('--split_dir', type=str, default=None,
                    help='path to the folder containing the requested splits for the cross validation')
parser.add_argument('--n_fold', type=int, default=4,
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
    n_fold = args.n_fold
    split_dir = args.split_dir

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    # one-hot encoding
    # KNN
    #optimal_parameters_knn = get_optimal_parameters_knn(df, logger, max_eval=64)
    #get_cross_val_accuracy_knn(df, logger, n_fold, optimal_parameters_knn, split_dir)

    ## RF
    # optimal_parameters_rf = get_optimal_parameters_rf(df, logger, max_eval=64)
    # get_cross_val_accuracy_rf(df, logger, n_fold, optimal_parameters_rf, split_dir)
    #
    # ## XGboost
    # optimal_parameters_xgboost = get_optimal_parameters_xgboost(df, logger, max_eval=128)
    # get_cross_val_accuracy_xgboost(df, logger, n_fold, optimal_parameters_xgboost, split_dir)
    #

    space_knn = {
        'n_neighbors': hp.quniform('n_neighbors', low=3, high=15, q=2),
    }
    space_rf = {
        'n_estimators': hp.choice('n_estimators', [10, 30, 50, 100, 150, 200, 300, 400, 600]),
        'max_features': hp.quniform('max_features', low=0.1, high=1, q=0.1),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 5, 10, 20, 50])
    }
    space_xgb = {
        'max_depth': hp.quniform('max_depth', low=2, high=10, q=1),
        'gamma': hp.qloguniform('gamma', low=0.0, high=6.0, q=2.0),
        'n_estimators': hp.quniform('n_estimators', low=100, high=800, q=100),
        'learning_rate': hp.quniform('learning_rate', low=0.05, high=0.20, q=0.05),
        'min_child_weight': hp.quniform('min_child_weight', low=2, high=10, q=2.0)
    }



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
            #optimal_parameters_knn_fps = get_optimal_parameters_knn_fp(df_fps_drfp, logger, max_eval=64)
            #get_cross_val_accuracy_knn_fp(df_fps_drfp, logger, n_fold, optimal_parameters_knn_fps, split_dir)
            rmse, mae, r2 = nested_cross_val_fp(df_fps_drfp, 6, space=space_knn, objective=objective_knn_fp,
                                                model_class=KNeighborsRegressor,
                                                max_eval=32, logger=logger)
            logger.info(f'6-fold nested-cv RMSE, MAE and R^2 for KNN fps {rmse:.3f} {mae:.3f} {r2:.3f}')

            logger.info(f"Fingerprint: Morgan (radius={rad}, nbits={nbit})")
            #optimal_parameters_knn_fps = get_optimal_parameters_knn_fp(df_fps_morgan, logger, max_eval=64)
            #get_cross_val_accuracy_knn_fp(df_fps_morgan, logger, n_fold, optimal_parameters_knn_fps, split_dir)
            rmse, mae, r2 = nested_cross_val_fp(df_fps_morgan, 6, space=space_knn, objective=objective_knn_fp,
                                                model_class=KNeighborsRegressor,
                                                max_eval=32, logger=logger)
            logger.info(f'6-fold nested-cv RMSE, MAE and R^2 for KNN fps {rmse:.3f} {mae:.3f} {r2:.3f}')

            # RF fingerprints
            logger.info(f"Fingerprint: DRFP (radius={rad}, nbits={nbit})")
            #optimal_parameters_rf_fps = get_optimal_parameters_rf_fp(df_fps_drfp, logger, max_eval=64)
            #get_cross_val_accuracy_rf_fps(df_fps_drfp, logger, n_fold, optimal_parameters_rf_fps, split_dir)
            rmse, mae, r2 = nested_cross_val_fp(df_fps_drfp, 6, space=space_rf, objective=objective_rf_fp,
                                               model_class=RandomForestRegressor,
                                               max_eval=64, logger=logger)
            logger.info(f'6-fold nested-cv RMSE, MAE and R^2 for RF fps {rmse:.3f} {mae:.3f} {r2:.3f}')

            logger.info(f"Fingerprint: Morgan (radius={rad}, nbits={nbit})")
            #optimal_parameters_rf_fps = get_optimal_parameters_rf_fp(df_fps_morgan, logger, max_eval=64)
            #get_cross_val_accuracy_rf_fps(df_fps_morgan, logger, n_fold, optimal_parameters_rf_fps, split_dir)
            rmse, mae, r2 = nested_cross_val_fp(df_fps_morgan, 6, space=space_rf, objective=objective_rf_fp,
                                                model_class=RandomForestRegressor,
                                                max_eval=64, logger=logger)
            logger.info(f'6-fold nested-cv RMSE, MAE and R^2 for RF fps {rmse:.3f} {mae:.3f} {r2:.3f}')

            # XGboost fingerprints
            logger.info(f"Fingerprint: DRFP (radius={rad}, nbits={nbit})")
            #optimal_parameters_xgboost_fp = get_optimal_parameters_xgboost_fp(df_fps_drfp, logger, max_eval=128)
            #get_cross_val_accuracy_xgboost_fps(df_fps_drfp, logger, n_fold, optimal_parameters_xgboost_fp, split_dir)
            rmse, mae, r2 = nested_cross_val_fp(df_fps_drfp, 6, space=space_xgb, objective=objective_xgboost_fp,
                                                model_class=XGBRegressor,
                                                max_eval=128, logger=logger)
            logger.info(f'6-fold nested-cv RMSE, MAE and R^2 for XGBoost fps {rmse:.3f} {mae:.3f} {r2:.3f}')

            logger.info(f"Fingerprint: Morgan (radius={rad}, nbits={nbit})")
            #optimal_parameters_xgboost_fp = get_optimal_parameters_xgboost_fp(df_fps_morgan, logger, max_eval=128)
            #get_cross_val_accuracy_xgboost_fps(df_fps_morgan, logger, n_fold, optimal_parameters_xgboost_fp, split_dir)
            rmse, mae, r2 = nested_cross_val_fp(df_fps_morgan, 6, space=space_xgb, objective=objective_xgboost_fp,
                                                model_class=XGBRegressor,
                                                max_eval=128, logger=logger)
            logger.info(f'6-fold nested-cv RMSE, MAE and R^2 for XGBoost fps {rmse:.3f} {mae:.3f} {r2:.3f}')

