#!/usr/bin/python
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import re


def create_k_folder(df, n_folds, split_dir) -> None:
    """
    Function to perform cross-validation

    Args:
        df (pd.DataFrame): the DataFrame containing features and targets
        n_folds (int): the number of folds
        split_dir (str): the path to a directory when will be save the data splits.

    """

    df = df.sample(frac=1, random_state=0)
    chunk_list = np.array_split(df, n_folds)

    for i in range(n_folds):
        df_train = pd.concat([chunk_list[j] for j in range(n_folds) if j != i])
        df_test = chunk_list[i]

        folder = f"fold_{i}"

        final_dir = os.path.join(split_dir, folder)

        if not os.path.isdir(final_dir):
            os.makedirs(final_dir)

        df_train.to_csv(f"{final_dir}/train.csv")
        df_test.to_csv(f"{final_dir}/test.csv")
    
    return None


def cross_val(df, model, n_folds, target_column='Std_DFT_forward', sample=None, split_dir=None):
    """
    Function to perform cross-validation

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
    df, encode_columns = one_hot_encoding(df)

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

        target_scaler = StandardScaler()
        target_scaler.fit(y_train)
        y_train = target_scaler.transform(y_train)
        y_test = target_scaler.transform(y_test)

        # fit and compute rmse and mae
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1,1)

        rmse_fold = np.sqrt(mean_squared_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test)))
        rmse_list.append(rmse_fold)

        mae_fold = mean_absolute_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test))
        mae_list.append(mae_fold)

        r2_fold = r2_score(target_scaler.inverse_transform(y_test), target_scaler.inverse_transform(predictions))
        r2_list.append(r2_fold)

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))
    r2 = np.mean(np.array(r2_list))

    return rmse, mae, r2



def cross_val_fp(df_fp, model, n_folds, target_column='Std_DFT_forward', split_dir=None):
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
        predictions = predictions.reshape(-1,1)

        rmse_fold = np.sqrt(mean_squared_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test)))
        rmse_list.append(rmse_fold)

        mae_fold = mean_absolute_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test))
        mae_list.append(mae_fold)

        r2_fold = r2_score(target_scaler.inverse_transform(y_test), target_scaler.inverse_transform(predictions))
        r2_list.append(r2_fold)

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))
    r2 = np.mean(np.array(r2_list))

    return rmse, mae, r2


def write_predictions(
        predicted_activation_energies,
        true_activation_energies,
        file_name,
):
    """Write predictions to a .csv file.

        Args:
            rxn_id (pd.DataFrame): dataframe consisting of the rxn_ids
            activation_energies_predicted (List): list of predicted activation energies
            reaction_energies_predicted (List): list of predicted reaction energies
            rxn_id_column (str): name of the rxn-id column
            file_name : name of .csv file to write the predicted values to
        """

    test_predicted = pd.DataFrame()
    test_predicted["predicted_activation_energy"] = predicted_activation_energies.tolist()
    test_predicted["true_activation_energy"] = true_activation_energies.tolist()
    test_predicted.to_csv(file_name)

    return None


def one_hot_encoding(df, encoding_column='Type'):

    # df[encoding_column].unique() # 10 elements
    df = pd.get_dummies(df, columns=[encoding_column])
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                  df.columns.values]
    encode_columns = [column for column in df.columns if 'Type_' in column]

    return df, encode_columns

