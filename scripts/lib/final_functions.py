#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lib.cross_val import cross_val, cross_val_fp
from hyperopt import hp
from lib.bayesian_opt import bayesian_opt
from lib.bayesian_opt import objective_knn_fp, objective_knn
from lib.bayesian_opt import objective_rf, objective_rf_fp
from lib.bayesian_opt import objective_xgboost, objective_xgboost_fp
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

n_estimator_dict = {0: 10, 1: 30, 2: 50, 3: 100, 4: 150, 5: 200, 6: 300, 7: 400, 8: 600}
min_samples_leaf_dict = {0: 1, 1: 2, 2: 5, 3: 10, 4: 20, 5: 50}


def prepare_df(input_file, features):

    data = pd.read_pickle(input_file)

    features = features + ['rxn_id'] + ['DG_TS'] + ['G_r'] + ['DG_TS_tunn']

    columns_remove = [column for column in data.columns if column not in features]

    df = data.drop(columns=columns_remove)

    return df


def get_optimal_parameters_knn(df, logger, max_eval=32):
    """
    Get the optimal descriptors for KNN (fingerprints) through Bayesian optimization.

    Args:
        df_fp (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'n_neighbors': hp.quniform('n_neighbors', low=3, high=15, q=2),
    }

    optimal_parameters = bayesian_opt(df, space, objective_knn, KNeighborsRegressor, max_eval=max_eval)
    logger.info(f'Optimal parameters for KNN -- one-hot-encoding: {optimal_parameters}')

    return optimal_parameters


def get_cross_val_accuracy_knn(df, logger, n_fold, parameters, split_dir=None):
    """
    Get the linear regression accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """

    model = KNeighborsRegressor(n_neighbors=int(parameters['n_neighbors']), weights='distance')
    rmse, mae, r2 = cross_val(df, model, n_fold, split_dir=split_dir)

    logger.info(f'{n_fold}-fold CV RMSE, MAE and R^2  for k-NN one-hot-encoding: {rmse} {mae} {r2}')


def get_optimal_parameters_knn_fp(df_fp, logger, max_eval=32):
    """
    Get the optimal descriptors for KNN (fingerprints) through Bayesian optimization.

    Args:
        df_fp (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'n_neighbors': hp.quniform('n_neighbors', low=3, high=15, q=2),
    }

    optimal_parameters = bayesian_opt(df_fp, space, objective_knn_fp, KNeighborsRegressor, max_eval=max_eval)
    logger.info(f'Optimal parameters for KNN -- fingerprints: {optimal_parameters}')

    return optimal_parameters


def get_cross_val_accuracy_knn_fp(df_fp, logger, n_fold, parameters, split_dir=None):
    """
    Get the linear regression accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """

    model = KNeighborsRegressor(n_neighbors=int(parameters['n_neighbors']), weights='distance')
    rmse, mae, r2 = cross_val_fp(df_fp, model, n_fold, split_dir=split_dir)

    logger.info(f'{n_fold}-fold CV RMSE, MAE and R^2  for k-NN fingerprints: {rmse} {mae} {r2}')


def get_optimal_parameters_rf(df, logger, max_eval=32):
    """
    Get the optimal descriptors for random forest (descriptors) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'n_estimators': hp.choice('n_estimators', [10, 30, 50, 100, 150, 200, 300, 400, 600]),
        'max_features': hp.quniform('max_features', low=0.1, high=1, q=0.1),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 5, 10, 20, 50])
    }

    optimal_parameters = bayesian_opt(df, space, objective_rf, RandomForestRegressor, max_eval=max_eval)
    optimal_parameters['n_estimators'] = n_estimator_dict[optimal_parameters['n_estimators']]
    optimal_parameters['min_samples_leaf'] = min_samples_leaf_dict[optimal_parameters['min_samples_leaf']]
    logger.info(f'Optimal parameters for RF -- one-hot-encoding: {optimal_parameters}')

    return optimal_parameters


def get_cross_val_accuracy_rf(df, logger, n_fold, parameters, split_dir=None):
    """
    Get the random forest (descriptors) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = RandomForestRegressor(n_estimators=int(parameters['n_estimators']),
            max_features=parameters['max_features'], min_samples_leaf=int(parameters['min_samples_leaf']))
    rmse, mae, r2 = cross_val(df, model, n_fold, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE, MAE and R^2  for RF -- one-hot-encoding: {rmse} {mae} {r2}')


def get_optimal_parameters_rf_fp(df_fp, logger, max_eval=32):
    """
    Get the optimal descriptors for random forest (fingerprints) through Bayesian optimization.

    Args:
        df_fp (pd.DataFrame): input dataframe containing the fingerprints
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'n_estimators': hp.choice('n_estimators', [10, 30, 50, 100, 150, 200, 300, 400, 600]),
        'max_features': hp.quniform('max_features', low=0.1, high=1, q=0.1),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 5, 10, 20, 50])
    }
    optimal_parameters = bayesian_opt(df_fp, space, objective_rf_fp, RandomForestRegressor, max_eval=max_eval)
    optimal_parameters['n_estimators'] = n_estimator_dict[optimal_parameters['n_estimators']]
    optimal_parameters['min_samples_leaf'] = min_samples_leaf_dict[optimal_parameters['min_samples_leaf']]
    logger.info(f'Optimal parameters for RF -- fingerprints: {optimal_parameters}')

    return optimal_parameters


def get_cross_val_accuracy_rf_fps(df_fp, logger, n_fold, parameters, split_dir=None):
    """
    Get the random forest (descriptors) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = RandomForestRegressor(n_estimators=int(parameters['n_estimators']),
                                max_features=parameters['max_features'], 
                                min_samples_leaf=int(parameters['min_samples_leaf']))
    rmse, mae, r2 = cross_val_fp(df_fp, model, n_fold, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE, MAE and R^2  for RF -- fingerprints: {rmse} {mae} {r2}')


def get_optimal_parameters_xgboost(df, logger, max_eval=32):
    """
    Get the optimal descriptors for xgboost (descriptors) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'max_depth': hp.quniform('max_depth', low=2, high=10, q=1),
        'gamma': hp.qloguniform('gamma', low=0.0, high=6.0, q=2.0),
        'n_estimators': hp.quniform('n_estimators', low=100, high=800, q=100),
        'learning_rate': hp.quniform('learning_rate', low=0.05, high=0.20, q=0.05),
        'min_child_weight': hp.quniform('min_child_weight', low=2, high=10, q=2.0)
    }
    optimal_parameters = bayesian_opt(df, space, objective_xgboost, XGBRegressor, max_eval=max_eval)
    logger.info(f'Optimal parameters for xgboost -- one-hot-encoding: {optimal_parameters}')

    return optimal_parameters

def get_cross_val_accuracy_xgboost(df, logger, n_fold, parameters, split_dir=None):
    """
    Get the xgboost (descriptors) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = XGBRegressor(max_depth=int(parameters['max_depth']), 
                        gamma=parameters['gamma'], 
                        n_estimators=int(parameters['n_estimators']),
                        learning_rate=parameters['learning_rate'],
                        min_child_weight=parameters['min_child_weight'])
    rmse, mae, r2 = cross_val(df, model, n_fold, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE, MAE and R^2 for xgboost -- one-hot-encoding: {rmse} {mae} {r2}')

def get_optimal_parameters_xgboost_fp(df_fp, logger, max_eval=32):
    """
    Get the optimal descriptors for xgboost (fingerprints) through Bayesian optimization.

    Args:
        df_fp (pd.DataFrame): input dataframe containing the fingerprints
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'max_depth': hp.quniform('max_depth', low=2, high=10, q=1),
        'gamma': hp.qloguniform('gamma', low=0.0, high=6.0, q=2.0),
        'n_estimators': hp.quniform('n_estimators', low=100, high=800, q=100),
        'learning_rate': hp.quniform('learning_rate', low=0.05, high=0.20, q=0.05),
        'min_child_weight': hp.quniform('min_child_weight', low=2, high=10, q=2.0)
    }
    optimal_parameters = bayesian_opt(df_fp, space, objective_xgboost_fp, XGBRegressor, max_eval=max_eval)
    logger.info(f'Optimal parameters for xgboost -- fingerprints: {optimal_parameters}')

    return optimal_parameters

def get_cross_val_accuracy_xgboost_fps(df_fp, logger, n_fold, parameters, split_dir=None):
    """
    Get the xgboost (descriptors) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = XGBRegressor(max_depth=int(parameters['max_depth']), 
                        gamma=parameters['gamma'], 
                        n_estimators=int(parameters['n_estimators']),
                        learning_rate=parameters['learning_rate'],
                        min_child_weight=parameters['min_child_weight'])
    rmse, mae, r2 = cross_val_fp(df_fp, model, n_fold, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE, MAE and R^2 for xgboost -- fingerprints: {rmse} {mae} {r2}')

