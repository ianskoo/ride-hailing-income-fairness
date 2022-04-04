"""Predict ride hailing customers demand in a city"""
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import *  # LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor


class Prediction:
    """Predict ride hailing customers demand in a city"""

    def __init__(self, 
                 prepared_dataset, 
                 fst_square: tuple, 
                 poly_degree=1,
                 max_features=8, 
                 n_estimators=300, 
                 max_depth=100) -> None:
        
        self.ml_df = prepared_dataset
        self.seed = 144
        self.x_fst_label = 'is_holiday'
        self.x_lst_label = 'minute'
        self.y_fst_label = str(fst_square)
        self.poly_degree = poly_degree
        self.models_df = pd.read_csv('data/model_selection.csv', index_col=0)

        self.train_set, self.test_set = train_test_split(self.ml_df, test_size=0.2, random_state=self.seed)
        self.x = self.preprocess(self.train_set.loc[:, self.x_fst_label:self.x_lst_label], poly_degree)
        self.y = self.train_set.loc[:, self.y_fst_label:]

        self.x_test = self.preprocess(self.test_set.loc[:, self.x_fst_label:self.x_lst_label], poly_degree)
        self.y_test = self.test_set.loc[:, self.y_fst_label:]

        # Train the final model with final hyperparameters
        self.model = self.train_model(RandomForestRegressor(max_features=max_features,
                                                            n_estimators=n_estimators, 
                                                            max_depth=max_depth))

    def preprocess(self, data, poly_degree):
        # num_pipeline = Pipeline([
        #     ('std_scaler', StandardScaler())
        # ])
        # data = num_pipeline.fit_transform(data)
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        poly_features = PolynomialFeatures(degree=poly_degree)
        data = poly_features.fit_transform(data)
        return data

    def preprocess_test(self, data, poly_degree):
        scaler = MinMaxScaler()
        data = scaler.transform(data)
        poly_features = PolynomialFeatures(degree=poly_degree)
        data = poly_features.transform(data)
        return data

    def train_model(self, model=None):
        # model = LinearRegression()
        # model = DecisionTreeRegressor()
        # model = Ridge(alpha=0.03)
        # model = RandomForestRegressor()
        # model = Lasso(alpha=0.001)

        # print(f"X: {self.x}, y: {self.y}")
        model.fit(self.x, self.y)
        return model

    def compare_models(self):
        """Compare a few (regression) models quick & dirty to select the best one"""
        models = [
            LinearRegression(),
            Ridge(),
            DecisionTreeRegressor(),
            RandomForestRegressor(max_depth=10),
            ExtraTreesRegressor(max_depth=10),
            ElasticNet(alpha=0.1, l1_ratio=0.5),
            # MultiOutputRegressor(AdaBoostRegressor()), # Bad
            MultiOutputRegressor(SVR(kernel='poly')),
            # Lasso(max_iter=10000, tol=0.01), # Not working
            # MultiOutputRegressor(RANSACRegressor(LinearRegression(), max_trials=10)), # Not working
            # MultiOutputRegressor(MLPRegressor(alpha=0.1, hidden_layer_sizes=(30, 30, 30))), # Not working
            # MultiOutputRegressor(GaussianProcessRegressor(kernel=ExpSineSquared())), # Not working
            # GaussianProcessRegressor(kernel=1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)), # Not working
        ]

        for model in models:
            t0 = time()
            print(f"{str(model)}: ", end=" ")
            fit_model = model.fit(self.x, self.y)
            res = self.check_error(fit_model)
            print(f"Score: {np.sqrt(-res.mean()):2f}, training time: {time() - t0}")

            row = self.models_df[(self.models_df['model'] == str(model)) &
                                 (self.models_df['poly degree'] == self.poly_degree)]
            if row.shape[0] > 1:
                print("Warning: duplicate rows! Skipping...")
            elif row.empty:
                entry = {'model': str(model),
                         'poly degree': self.poly_degree,
                         'training time': time() - t0,
                         'neg mean squared error': res.mean(),
                         'std dev nmse': res.std(),
                         'mean error': np.sqrt(-res.mean()),
                         }
                self.models_df.loc[self.models_df.shape[0]] = entry
            else:
                entry = {'neg mean squared error': res.mean(),
                         'std dev nmse': res.std(),
                         'mean error': np.sqrt(-res.mean())}
                row_idx = np.where((self.models_df['model'] == str(model)) &
                                   (self.models_df['poly degree'] == self.poly_degree))[0][0]
                for key, val in entry.items():
                    self.models_df.loc[row_idx, key] = val
                # for i in (row_idx, key, val, row):
                #     print(i)

        self.models_df.to_csv('data/model_selection.csv')

    def check_error(self, print=True, model=RandomForestRegressor(max_depth=100, max_features=8, n_estimators=300)):
        # scores = cross_val_score(model, self.x, self.y, scoring='neg_mean_squared_error', cv=5)
        # scores = cross_val_score(model, self.x, self.y, scoring='neg_mean_absolute_percentage_error')
        scores = cross_val_score(model, self.x, self.y, cv=5)
        if print:
            # lin_reg_rmse_scores = -scores
            print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        return scores

    def check_gen_error(self):
        predictions = self.model.predict(self.x_test)
        final_mse = mean_squared_error(self.y_test, predictions)
        final_rmse = np.sqrt(final_mse)
        final_r2 = r2_score(self.y_test, predictions)
        print(f"Final MSE: {final_mse}, final mean error: {final_rmse}, R2: {final_r2}")

    def search_params(self, model_name: str):
        """Search the best hyperparameters for the chosen model"""
        if model_name == "random_forest":
            param_grid = [
                # {'n_estimators': [3, 10, 30, 100], 'max_features': [2, 4, 6, 8]},
                # {'bootstrap': [False], 'n_estimators': [10, 30], 'max_features': [3, 4, 5]},
                # {'n_estimators': [30, 100, 300, 1000], 'max_features': [6, 8, None]},
                # {'n_estimators': [300, 1000], 'max_features': [6, 8], 'max_depth': [30, 100]},
                # {'n_estimators': [300], 'max_features': [7, 8, 9, 10, 11], 'max_depth': [30, 100]},
                {'n_estimators': [300], 'max_features': [6, 8, 10], 'max_depth': [30, 100]},
            ]
            model = RandomForestRegressor()

        elif model_name == "ridge":
            param_grid = [
                {'alpha': [0.01, 0.03, 0.1, 0.3, 1]}
            ]
            model = Ridge()

        grid_search = GridSearchCV(model, param_grid, cv=5,
                                #    scoring='neg_mean_squared_error',
                                   scoring='neg_mean_absolute_percentage_error',
                                   return_train_score=True)
        grid_search.fit(self.x, self.y)
        return grid_search
