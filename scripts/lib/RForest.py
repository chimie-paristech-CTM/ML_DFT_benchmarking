from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np


class RForest:
    """ A Random Forest model ensemble for estimating mean and variance """

    def __init__(
            self,
            n_estimators=30,
            max_features=0.5,
            min_samples_leaf=1,
            seed=3,
    ):

        self.model = RandomForestRegressor(n_estimators=n_estimators,
                                           max_features=max_features,
                                           min_samples_leaf=min_samples_leaf,
                                           random_state=seed)
        self.target_scaler = StandardScaler()

    def train(self, train, target_column='Std_DFT_forward'):

        y_train = train[[target_column]]

        # scale targets
        self.target_scaler.fit(y_train)
        y_train = self.target_scaler.transform(y_train)

        X_train = []
        for fp in train['Fingerprints'].values.tolist():
            X_train.append(list(fp))

        # fit and compute rmse
        self.model.fit(X_train, y_train.ravel())


    def get_means_and_vars(self, test):

        X_test = []
        for fp in test['Fingerprints'].values.tolist():
            X_test.append(list(fp))

        trees = [tree for tree in self.model.estimators_]
        preds = [tree.predict(X_test) for tree in trees]
        preds = np.array(preds)

        means = np.mean(preds, axis=0)
        vars = np.var(preds, axis=0)

        predictions = self.target_scaler.inverse_transform(means.reshape(-1, 1))
        variance = self.target_scaler.inverse_transform(vars.reshape(-1, 1))

        return predictions, variance

