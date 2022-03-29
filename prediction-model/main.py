"""Instantiates the package classes and runs the program."""
from cgi import test
from functools import cache
from multiprocessing.spawn import prepare
import data
import map_plotting as map_plt
import prediction
import numpy as np

def grid_search(pr):
    grid_search = pr.search_params("random_forest")
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        # print(np.sqrt(-mean_score), params)
        print(-mean_score, params)
    print(f"Best parameters: {grid_search.best_params_}\n")
    
    # for poly_degree in range(2, 10):
    #     pr = prediction.Prediction(prepared_dataset=data1.ml_df, fst_square=data1.square_coords[0][0], poly_degree=poly_degree)
    #     grid_search = pr.search_params("ridge_regression")
    #
    #     cvres = grid_search.cv_results_
    #     for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #         print(f"poly. degree {poly_degree}: {np.sqrt(-mean_score), params}")
    #     print(f"Best alpha for polynomial degree {poly_degree}: {grid_search.best_params_}\n")

def build_df_and_train():
    for sq in [7]:
        data1 = data.Data(
            # grid_square_size=4,
            latitude_divisions=sq,
            use_cached_taxi_dataset=True,
            use_cached_ml_dataset=True,
            debug=True,
            )

        nr_squares = len(data1.square_coords[0])*len(data1.square_coords)
        pr = prediction.Prediction(prepared_dataset=data1.ml_df,
                                   fst_square=data1.ml_df.columns[11],
                                   pickle_path=f"data/random_forest_8_300_100_squares_{nr_squares}.pkl")

    print(pr.check_gen_error())


if __name__ == '__main__':
    
    data_1 = data.Data(# grid_square_size=4,
                   latitude_divisions=7,
                   use_cached_taxi_dataset=False,
                   use_cached_ml_dataset=True)

    # build_df_and_train()
        # grid_search(pr)
    
    # print(f"Model R2 score on test set: {pr.model.score(pr.x_test, pr.y_test):2f}")
    # print(pr.model.oob_score)
