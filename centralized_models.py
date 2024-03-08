from itertools import product
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import model_eval as ME
# from sklearn.preprocessing import StandardScaler
from preprocessing import GaussianFilter
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import PolynomialFeatures

# class CustomPipelineStep(BaseEstimator, TransformerMixin):
#     def __init__(self, cols):
#     def fit(self, X, y=None):
#     def transform(self, X, y=None):

if __name__ == "__main__":
    print("Combining all datasets for centralized learning")
    dfs = []
    N = 4
    for i in range(N):
        path = f"data\\train_FD00{i+1}.csv"
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs)
    cols = ["lpc_out_temp", "hpc_out_temp", "lpt_out_temp", "hpc_out_press", 
            "fan_speed", "core_speed", "hpc_stat_press", "flow_press_ratio", 
            "corr_fan_speed", "corr_core_speed", "bypass_ratio", "bleed_enthalpy", 
            "hpt_bleed", "lpt_bleed"]

    print("Vanilla LR Model:")
    pipe0 = Pipeline([("model", LinearRegression())])
    result0, _ = ME.train_and_eval_model(df.copy(), "RUL", cols, pipe0)
    print(result0)

    # print("\nScaled LR Model:")
    # pipe1 = Pipeline([("test", StandardScaler()),
    #                  ("model", LinearRegression())])
    # result1, _ = ME.train_and_eval_model(df.copy(), "RUL", cols, pipe1)
    # print(result1)


    print("\nFiltered LR Model N(0,1,55):")
    pipe2 = Pipeline([("filter", GaussianFilter(cols, 55, 1.0)),
                     ("model", LinearRegression())])
    result2, _ = ME.train_and_eval_model(df.copy(), "RUL", cols, pipe2)
    print(result2)

    # print("\nFiltered LR Model N(0,1,11):")
    # pipe3 = Pipeline([("filter", GaussianFilter(cols, 11, 1.0)),
    #                  ("model", LinearRegression())])
    # result3, _ = ME.train_and_eval_model(df.copy(), "RUL", cols, pipe3)
    # print(result3)

    # print("\nFiltered LR Model N(0,1,99):")
    # pipe4 = Pipeline([("filter", GaussianFilter(cols, 99, 1.0)),
    #                  ("model", LinearRegression())])
    # result4, _ = ME.train_and_eval_model(df.copy(), "RUL", cols, pipe4)
    # print(result4)

    # print("\nPCA LR Model:")
    # pipe3 = Pipeline([("pca", PCA(n_components=4, whiten=True)),
    #                  ("model", LinearRegression())])
    # result3, _ = ME.train_and_eval_model(df.copy(), "RUL", cols, pipe3)
    # print(result3)

    # print("\nFilter+PCA+Poly LR Model:")
    # pipe4 = Pipeline([("filter", GaussianFilter(cols, 55, 1.0)),
    #                   ("pca", PCA(n_components=4, whiten=True)),
    #                   ("poly", PolynomialFeatures(degree=2)),
    #                  ("model", LinearRegression())])
    # result4, _ = ME.train_and_eval_model(df.copy(), "RUL", cols, pipe4)
    # print(result4)
    
    # params = {"n_estimators": [1000],
    #           "max_depth": [64]}
    # paramsets = list(product(params["n_estimators"], params["max_depth"]))
    # # best_model = None
    # best_res = None
    # best_r2 = -1e32
    # for p in paramsets:
    #     print("\n")
    #     print(p)
    #     pipe3 = Pipeline([("filter", GaussianFilter(cols, 55, 1.0)),
    #                     ("model", RandomForestRegressor(p[0], max_depth=p[1], n_jobs=-1))])
    #     result3, _ = ME.train_and_eval_model(df.copy(), "RUL", cols, pipe3, k=5)
    #     r2 = result3.iloc[-1,-1]
    #     print(result3.iloc[-1,:])
    #     if r2 > best_r2:
    #         best_r2 = r2
    #         # best_model = model.copy()
    #         best_res = result3.copy()
    # print("\nFilter+RF Regressor:")
    # print(best_res)