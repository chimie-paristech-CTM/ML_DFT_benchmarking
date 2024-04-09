import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from lib.bayesian_opt import bayesian_opt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

n_estimator_dict = {0: 10, 1: 30, 2: 50, 3: 100, 4: 150, 5: 200, 6: 300, 7: 400, 8: 600}
min_samples_leaf_dict = {0: 1, 1: 2, 2: 5, 3: 10, 4: 20, 5: 50}



def nested_cross_val(df, n_folds, space, objective, model_class, max_eval, logger, encode_columns,
                     target_column='Std_DFT_forward', sample=None, split_dir=None):
    """
    Function to perform nested cross-validation

    Args:
        df (pd.DataFrame): the DataFrame containing features and targets
        model (sklearn.Regressor): An initialized sklearn model
        n_folds (int): the number of folds
        target_column (str): target column
        sample(int): the size of the subsample for the training set (default = None)
        split_dir (str): the path to a directory containing data splits. If None, random splitting is performed.

    Returns:
        int: the obtained RMSE and MAE
    """
    rmse_list, mae_list, r2_list = [], [], []

    if split_dir == None:
        df = df.sample(frac=1, random_state=0)
        chunk_list = np.array_split(df, n_folds)

    for i in range(n_folds):
        if split_dir == None:
            df_train = pd.concat([chunk_list[j] for j in range(n_folds) if j != i])
            if sample != None:
                df_train = df_train.sample(n=sample)
            df_test = chunk_list[i]
        else:
            rxn_ids_train1 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/train.csv'))[['rxn_id']].values.tolist()
            rxn_ids_train2 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/valid.csv'))[['rxn_id']].values.tolist()
            rxn_ids_train = list(np.array(rxn_ids_train1 + rxn_ids_train2).reshape(-1))
            df['train'] = df['rxn_id'].apply(lambda x: int(x) in rxn_ids_train)
            df_train = df[df['train'] == True]
            df_test = df[df['train'] == False]

        X_train, y_train = df_train[encode_columns], df_train[[target_column]]
        X_test, y_test = df_test[encode_columns], df_test[[target_column]]

        optimal_parameters = bayesian_opt(df_train, space, objective, model_class, max_eval=max_eval, encode_columns=encode_columns)
        match model_class.__name__:
            case 'RandomForestRegressor':
                optimal_parameters['n_estimators'] = n_estimator_dict[optimal_parameters['n_estimators']]
                optimal_parameters['min_samples_leaf'] = min_samples_leaf_dict[optimal_parameters['min_samples_leaf']]
                model = RandomForestRegressor(n_estimators=int(optimal_parameters['n_estimators']),
                                              max_features=optimal_parameters['max_features'],
                                              min_samples_leaf=int(optimal_parameters['min_samples_leaf']))
            case 'KNeighborsRegressor':
                model = KNeighborsRegressor(n_neighbors=int(optimal_parameters['n_neighbors']), weights='distance')
            case 'XGBRegressor':
                model = XGBRegressor(max_depth=int(optimal_parameters['max_depth']),
                                     gamma=optimal_parameters['gamma'],
                                     n_estimators=int(optimal_parameters['n_estimators']),
                                     learning_rate=optimal_parameters['learning_rate'],
                                     min_child_weight=optimal_parameters['min_child_weight'])

        #logger.info(f'Optimal parameters for {model_class.__name__} -- one-hot-encoding: {optimal_parameters}')

        target_scaler = StandardScaler()
        target_scaler.fit(y_train)
        y_train = target_scaler.transform(y_train)
        y_test = target_scaler.transform(y_test)

        # fit and compute rmse and mae
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1, 1)

        rmse_fold = np.sqrt(
            mean_squared_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test)))
        rmse_list.append(rmse_fold)

        mae_fold = mean_absolute_error(target_scaler.inverse_transform(predictions),
                                       target_scaler.inverse_transform(y_test))
        mae_list.append(mae_fold)

        r2_fold = r2_score(target_scaler.inverse_transform(y_test), target_scaler.inverse_transform(predictions))
        r2_list.append(r2_fold)

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))
    r2 = np.mean(np.array(r2_list))

    return rmse, mae, r2


def nested_cross_val_fp(df_fp, n_folds, space, objective, model_class, max_eval, logger,
                        target_column='Std_DFT_forward', split_dir=None):
    """
    Function to perform cross-validation with fingerprints

    Args:
        df_fp (pd.DataFrame): the DataFrame containing fingerprints and targets
        model (sklearn.Regressor): An initialized sklearn model
        n_folds (int): the number of folds
        target_column (str): target column
        split_dir (str): the path to a directory containing data splits. If None, random splitting is performed.

    Returns:
        int: the obtained RMSE
    """
    rmse_list, mae_list, r2_list = [], [], []

    if split_dir == None:
        df_fp = df_fp.sample(frac=1, random_state=0)
        chunk_list = np.array_split(df_fp, n_folds)

    for i in range(n_folds):
        if split_dir == None:
            df_train = pd.concat([chunk_list[j] for j in range(n_folds) if j != i])
            df_test = chunk_list[i]
        else:
            rxn_ids_train1 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/train.csv'))[['rxn_id']].values.tolist()
            rxn_ids_train2 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/valid.csv'))[['rxn_id']].values.tolist()
            rxn_ids_train = list(np.array(rxn_ids_train1 + rxn_ids_train2).reshape(-1))
            df_fp['train'] = df_fp['rxn_id'].apply(lambda x: int(x) in rxn_ids_train)
            df_train = df_fp[df_fp['train'] == True]
            df_test = df_fp[df_fp['train'] == False]

        y_train = df_train[[target_column]]
        y_test = df_test[[target_column]]

        optimal_parameters = bayesian_opt(df_train, space, objective, model_class, max_eval=max_eval)
        match model_class.__name__:
            case 'RandomForestRegressor':
                optimal_parameters['n_estimators'] = n_estimator_dict[optimal_parameters['n_estimators']]
                optimal_parameters['min_samples_leaf'] = min_samples_leaf_dict[optimal_parameters['min_samples_leaf']]
                model = RandomForestRegressor(n_estimators=int(optimal_parameters['n_estimators']),
                                              max_features=optimal_parameters['max_features'],
                                              min_samples_leaf=int(optimal_parameters['min_samples_leaf']))
            case 'KNeighborsRegressor':
                model = KNeighborsRegressor(n_neighbors=int(optimal_parameters['n_neighbors']), weights='distance')
            case 'XGBRegressor':
                model = XGBRegressor(max_depth=int(optimal_parameters['max_depth']),
                                     gamma=optimal_parameters['gamma'],
                                     n_estimators=int(optimal_parameters['n_estimators']),
                                     learning_rate=optimal_parameters['learning_rate'],
                                     min_child_weight=optimal_parameters['min_child_weight'])

        #logger.info(f'Optimal parameters for {model_class.__name__} {i} fold: {optimal_parameters}')

        # scale targets
        target_scaler = StandardScaler()
        target_scaler.fit(y_train)
        y_train = target_scaler.transform(y_train)
        y_test = target_scaler.transform(y_test)

        X_train = []
        for fp in df_train['Fingerprints'].values.tolist():
            X_train.append(list(fp))
        X_test = []
        for fp in df_test['Fingerprints'].values.tolist():
            X_test.append(list(fp))

        # fit and compute rmse
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1, 1)

        rmse_fold = np.sqrt(
            mean_squared_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test)))
        rmse_list.append(rmse_fold)

        mae_fold = mean_absolute_error(target_scaler.inverse_transform(predictions),
                                       target_scaler.inverse_transform(y_test))
        mae_list.append(mae_fold)

        r2_fold = r2_score(target_scaler.inverse_transform(y_test), target_scaler.inverse_transform(predictions))
        r2_list.append(r2_fold)

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))
    r2 = np.mean(np.array(r2_list))

    return rmse, mae, r2

