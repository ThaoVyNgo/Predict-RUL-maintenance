import pandas as pd
from random import sample
from sklearn.metrics import mean_squared_error, r2_score
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_eval_model(data, target_col, x_cols, model, test_size=20, 
                         k=20, verbose=False, epochs=None):
    folds = [f+1 for f in range(k)] + ["avg"]
    train_rmse = []
    train_r2 = []
    val_rmse = []
    val_r2 = []
    units = list(data["unit"].unique())
    best_rmse = 1e10
    best_r2 = -1e10
    best_model = None
    for i in range(k):
        val_units = sample(units, test_size)
        if verbose:
            print(f"Fold {i+1} for {model}")
        train = data[data["unit"].isin(val_units) == False]
        val = data[data["unit"].isin(val_units)]
        x_train = train[x_cols]
        y_train = train[target_col]
        x_val = val[x_cols]
        y_val = val[target_col]
        if epochs is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, model__epochs=epochs)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        train_rmse.append(mean_squared_error(y_train, train_pred, squared=False))
        rmse_i = mean_squared_error(y_val, val_pred, squared=False)
        val_rmse.append(rmse_i)
        train_r2.append(r2_score(y_train, train_pred))
        r2_i = r2_score(y_val, val_pred)
        val_r2.append(r2_i)

        if (rmse_i <= (best_rmse * 1.1)) and (r2_i > best_r2):
            best_rmse = rmse_i
            best_r2 = r2_i
            best_model = None
            best_model = model

    train_rmse.append(sum(train_rmse)/len(train_rmse))
    val_rmse.append(sum(val_rmse)/len(val_rmse))    
    train_r2.append(sum(train_r2)/len(train_r2))
    val_r2.append(sum(val_r2)/len(val_r2))
    results = pd.DataFrame({"Fold": folds, "Train RMSE": train_rmse,
                            "Train R^2": train_r2, "Val RMSE":val_rmse,
                            "Val R^2": val_r2})
    return results, best_model

def eval_model_windowed(x, y, model, score=rmse, k=5, epochs=1):
    folds = [f+1 for f in range(k)] + ["avg"]
    train_scores = []
    val_scores = []
    samples = [s for s in range(x.shape[0])]
    test_size = int(x.shape[0] * 0.2)
    for i in range(k):
        val_samples = sample(samples, test_size)
        train_samples = [s for s in samples if s not in val_samples]
        print(f"Fold {i+1}:")
        x_train = x[train_samples,:,:]
        y_train = y[train_samples,]
        x_val = x[val_samples,:,:]
        y_val = y[val_samples,]
        model.fit(x_train, y_train, model__epochs=epochs, model__verbose=0)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        train_scores.append(score(y_train, train_pred))
        val_scores.append(score(y_val, val_pred))

    train_scores.append(sum(train_scores)/k)
    val_scores.append(sum(val_scores)/k)    
    results = pd.DataFrame({"Fold": folds, "Train Score": train_scores,
                            "Val Score": val_scores})
    return results, model

if __name__  == "__main__":
    print("Running in test mode:")
    print("Training LinearRegression on linear data with noise.")
    N_UNITS = 5
    N = 100
    x = []
    y = []
    units = []
    for i in range(N_UNITS):
        units += [i+1]*N
        xi = np.linspace(0, 100, N)
        m = 0.5
        b = 5
        noise = np.random.normal(0, 2, N)
        yi = m*xi + b + noise
        x += list(xi)
        y += list(yi)
    df = pd.DataFrame({"unit": units, "x": x, "y": y})
    results, model = train_and_eval_model(df, "y", ["x"], LinearRegression(), test_size=1, verbose=True)
    print(results)
    print(f"\nweights + intercept = {model.coef_} + ({model.intercept_})")