from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from argparse import ArgumentParser
from lib.fingerprints import get_fingerprints_Morgan
from sklearn.preprocessing import StandardScaler
import numpy as np


parser = ArgumentParser()
parser.add_argument('--train_file', type=str, default='../data_smiles_curated.csv',
                    help='path to file containing the rxn-smiles')
parser.add_argument('--pool_file', type=str, default='../hypothetical_chemical_space.csv',
                    help='path to the input file')
# interactive way
parser.add_argument("--mode", default='client', action="store", type=str)
parser.add_argument("--host", default='127.0.0.1', action="store", type=str)
parser.add_argument("--port", default=57546, action="store", type=int)


def evaluation(df_train, df_test, target_column='Std_DFT_forward'):

    model = RandomForestRegressor(n_estimators=30,
                                  max_features=0.5,
                                  min_samples_leaf=1)

    y_train = df_train[[target_column]]

    # scale targets
    target_scaler = StandardScaler()
    target_scaler.fit(y_train)
    y_train = target_scaler.transform(y_train)

    X_train = []
    for fp in df_train['Fingerprints'].values.tolist():
        X_train.append(list(fp))
    X_test = []
    for fp in df_test['Fingerprints'].values.tolist():
        X_test.append(list(fp))

    # fit and compute rmse
    model.fit(X_train, y_train.ravel())
    predictions, variance = make_prediction(model, X_test)
    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1))
    variance = target_scaler.inverse_transform(variance.reshape(-1, 1))

    return predictions, variance


def make_prediction(model, X_test):

    trees = [tree for tree in model.estimators_]
    preds = [tree.predict(X_test) for tree in trees]
    preds = np.array(preds)

    return np.mean(preds, axis=0), np.var(preds, axis=0)


def acq_function_ucb(predictions, variance, beta=1.2):

    return predictions + beta * variance


if __name__ == "__main__":
    args = parser.parse_args()
    df_train = pd.read_csv(args.train_file, sep=';', index_col=0)
    df_pool = pd.read_csv(args.pool_file, index_col=0)
    df_train_fps = get_fingerprints_Morgan(df_train, rad=1, nbits=2048)
    df_pool_fps = get_fingerprints_Morgan(df_pool, rad=1, nbits=2048, labeled=False)

    pred, var = evaluation(df_train_fps, df_pool_fps)
    ucb = acq_function_ucb(pred, var)
    idx = np.argmax(ucb)
    rxn_smiles = df_pool.iloc[idx]