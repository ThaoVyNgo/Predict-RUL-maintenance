from math import ceil, pow, dist
from random import sample
import model_eval as ME
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from preprocessing import GaussianFilter, DataShaper, pad_data
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras import models, layers, optimizers, Input
from keras.models import load_model
import tensorflow as tf
from itertools import product
from MedianFilter import MedianFilter
from scipy.signal import medfilt
from random import random

K = 3 # k value for k-folds CV
P = 0.75 # chance of node getting selected during aggregation

class FederatedNode:
    def __init__(self):
        self.id = None
        self.df = None
        self.model = None
        self.active = False
        self.weights = []

    def initialize(self, df, x_cols, target, model, 
                   weight_update_func, weight_func, 
                   node_id, agg_weight=None):
        self.df = df
        self.x_cols = x_cols
        self.target = target
        self.model = model
        self.weight_update_func = weight_update_func
        self.weight_func = weight_func
        self.active = True
        self.id = node_id
        self.agg_weight = df.shape[0] if agg_weight is None else agg_weight
        return self

    def fit(self, weights=None, epochs=None, score_idx=-1):
        if weights is not None:                                             # after first round
            self.weights = weights
            self.weight_update_func(self.model, weights)
        # res, model = ME.train_and_eval_model(self.df, self.target, 
        #                                         self.x_cols, self.model,
        #                                         k=5, epochs=epochs)
        # self.model = model
        res, _ = ME.eval_model_windowed(self.df[0], self.df[1], 
                                     self.model, epochs=epochs, k=K, 
                                     rseed=42)
        print(res)
        return res.iloc[-1,score_idx]

    def broadcast(self):
        Gnoise = np.random.normal(0,0.1)
        return self.weight_func(self.model) + [Gnoise]

class FederatedModel:
    def __init__(self, num_rounds, agg_func, train_size=1.0,
                 epochs=None, score_idx=-1):
        self.num_nodes = 0
        self.num_rounds = num_rounds
        self.agg_func = agg_func
        self.train_size = train_size
        self.epochs = epochs
        self.score_idx = score_idx
        self.node_weights = []
        self.weights = None
        self.nodes = []
        self.node_agg_weights = []
        self.scores = []

    def check_active(self):
        ready = True
        for i, node in enumerate(self.nodes):
            if not node.active:
                print(f"Node {i} not initialized")
                ready = False
        return ready

    def get_next_node_id(self):
        return self.num_nodes+1

    def add_node(self, node):
        self.nodes.append(node)
        self.node_weights.append(None)
        self.node_agg_weights.append(node.agg_weight)
        self.num_nodes += 1

    def fit(self):
        if self.check_active():
            self.scores = []
            self.weights = None
            for r in range(self.num_rounds):
                self.scores.append([])
                print(f"Round {r+1}")
                self.train_round(r)
                self.aggregate()
        return self.scores

    def predict(self, X, i=0):
        return self.nodes[i].model.predict(X)

    def train_round(self, round):
        if self.train_size < 1.0:
            N = ceil(self.train_size * self.num_nodes)
            train_node_set = sample(self.nodes, N)
        else:
            train_node_set = self.nodes
        for _, node in enumerate(train_node_set):
            print(f"\nTraining node {node.id}")
            s = node.fit(self.weights, self.epochs, self.score_idx)
            self.scores[round].append(s)
            self.node_weights[node.id-1] = node.broadcast()

    def aggregate(self):
        self.weights = self.agg_func(self.nodes, self.node_agg_weights)

def linear_agg(node_weights, agg_weights, p=1.0):
    weights = [n.weights for n in node_weights]
    return list(np.average(weights, weights=agg_weights, axis=0))

def fed_avg(nodes, nk):
    n = sum(nk)
    W0 = nodes[0].model["model"].get_weights()
    weights = []
    for wi in range(len(W0)):
        weights.append(np.zeros(shape=W0[wi].shape))
        for row in range(W0[wi].shape[0]):
            try:
                for col in range(W0[wi].shape[1]):
                    agg_weight = 0
                    for i in range(len(nodes)):
                        if random() >= P:
                            continue
                        x = nodes[i].model["model"].get_weights()[wi][row][col]
                        agg_weight += ((nk[i]/n)*x)
                    weights[wi][row][col] = agg_weight
            except IndexError:
                agg_weight = 0
                for i in range(len(nodes)):
                    x = nodes[i].model["model"].get_weights()[wi][row]
                    agg_weight += ((nk[i]/n)*x)
                weights[wi][row] = agg_weight
    return weights

def power_weighted_mean(nodes, nk, p):
    if p == 1:
        return fed_avg(nodes, nk)
    n = sum(nk)
    W0 = nodes[0].model["model"].get_weights()
    weights = []
    for wi in range(len(W0)):
        weights.append(np.zeros(shape=W0[wi].shape))
        for row in range(W0[wi].shape[0]):
            try:
                for col in range(W0[wi].shape[1]):
                    agg_weight = 0
                    for i in range(len(nodes)):
                        w = (nk[i]/n)
                        x = nodes[i].model["model"].get_weights()[wi][row][col]
                        agg_weight += (-1.0 if x < 0 else 1.0)*(w * pow(abs(x), p))
                    weights[wi][row][col] = (-1.0 if agg_weight < 0 else 1.0)*pow(abs(agg_weight), (1.0/p))
            except IndexError as idx_err:
                agg_weight = 0
                for i in range(len(nodes)):
                    w = (nk[i]/n)
                    x = nodes[i].model["model"].get_weights()[wi][row]
                    agg_weight += (-1.0 if x < 0 else 1.0)*(w * pow(abs(x), p))
                weights[wi][row] = (-1.0 if agg_weight < 0 else 1.0)*pow(abs(agg_weight), (1.0/p))
    return weights

def lin_reg_weight_update(model, weights):
    model["model"].coef_ = weights

def network_weight_update(model, weights):
    model["model"].set_weights(weights)

def get_window_centroid(w):
    return [w.iloc[:,i].mean() for i in range(w.shape[1]-1)]

def get_window_dist(w1, w2):
    return dist(get_window_centroid(w1), get_window_centroid(w2))

def calc_initial_rul(df, u, w=10, g=7, thresh=0.2):
    df = df[df["unit"] == u]
    windows = [df.iloc[i:i+w,:] for i in range(0,(w*g)+1,w)]
    w1 = windows[0]
    for i, wi in enumerate(windows[1:]):
        d = get_window_dist(w1,wi)
        d = d*d
        if d >= thresh:
            return (wi.iloc[-1,-1] - (w*i))
    return None

def window_data(df, w=5):
    n_samples = df.shape[0] - w + 1
    x = np.zeros((n_samples,w,df.shape[1]))
    y = np.zeros((n_samples,))
    for i in range(n_samples):
        n = i+w
        x[i] = df.iloc[i:n,:].to_numpy()
        y[i] = df.iloc[n-1,-1]
    return x, y

def window_data_unitwise(df, w=5):
    x_list = []
    y_list = []
    for _, group in df.groupby(["unit"]):
        xi, yi = window_data(group, w)
        x_list.append(xi)
        y_list.append(yi)
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    return x, y

def run_pipeline(df, cols, window_size=10, 
                filter_size=101, med_window=5, n_comp=4,
                deg_window=12, pre_scaler=None, pca=None,
                g=7, thresh=0.2):
    if filter_size > 0:
        df[cols] = GaussianFilter(cols, filter_size, 3.0).fit_transform(df[cols])
    if med_window > 0:
        df = MedianFilter(df, cols, med_window)
    scaler = None
    if pre_scaler is None:
        scaler = StandardScaler()
        df.loc[:,cols] = scaler.fit_transform(df[cols])
    else:
        df.loc[:,cols] = pre_scaler.transform(df[cols])
    if "unit" in df.columns.values:
        for u in list(df["unit"].unique()):
            irul = calc_initial_rul(df, u, w=deg_window, g=g, thresh=thresh)
            if irul is not None:
                df.loc[df["unit"] == u,"RUL"] = df[df["unit"] == u]["RUL"].mask(df[df["unit"] == u]["RUL"] > irul, irul)
    if pca is None and n_comp > 0:
        pca = PCA(n_components=n_comp, whiten=True)
        pca.fit(df[cols])
    if n_comp > 0:
        new_df = pd.DataFrame(pca.transform(df[cols]), 
                              columns=['PCA%i' % i for i in range(n_comp)], 
                              index=df[cols].index)
        new_df["unit"] = df["unit"]
        new_df["RUL"] = df.iloc[:,-1]
    else:
        new_df = df[cols+["unit","RUL"]].copy()
    # print(new_df)
    x, y = window_data_unitwise(new_df, window_size)
    return x, y, scaler, pca

if __name__ == "__main__":
    cols = ["cycles","hpc_stat_press","burner_fuel_ratio","flow_press_ratio",
            "corr_core_speed","lpt_out_temp","hpc_out_press","bypass_ratio",
            "lpt_bleed","corr_fan_speed"]
    print("Running in test mode")
    N = 4
    R = 30
    W = [5] # [5,10,20,25]
    DW = [12]
    F = [0] # [0,5,9]
    S = [0.2] # [0.1,0.01,0.001]
    M = [0]# [10, 40, 70, 100] # median filter window length
    E = [10]#[1,5,10]
    D = [16]
    L = [0.0001]
    G = [7]
    T = [0.2]
    IMPORT_MODEL = False
    params = product(DW,G,T)
    best_scores = []
    best_score = np.inf
    best_params = {}

    for dw, g, t in params:
        # print(f"params: degradation window = {dw}, gauss std = {s}, epsilon = {e}")
        print(f"deg window = {dw}, g = {g}, thresh = {t}")
        fl_model = FederatedModel(R, fed_avg, epochs=E[-1])
        scalers = []
        pcas = []

        for i in range(N):
            df = pd.read_csv(f"data\\train_FD00{i+1}.csv")
            # df = pad_data(df, time_col="cycles", sample_col="unit", pad_val=0.)
            df = df[cols+["unit","RUL"]]
            x, y, scaler_i, pca_i = run_pipeline(df, cols, window_size=W[-1], 
                                                 filter_size=F[-1], med_window=M[-1],
                                                 deg_window=dw, g=g, thresh=t)
            scalers.append(scaler_i)
            pcas.append(pca_i)
            if IMPORT_MODEL:
                nn = load_model(f"n{i+1}_model.keras")
            else:
                d = D[-1]
                nn = models.Sequential()
                nn.add(Input(shape=(x.shape[1],x.shape[2])))
                nn.add(layers.LSTM(d, return_sequences=True))
                nn.add(layers.GaussianNoise(S[-1]))
                nn.add(layers.Dropout(0.25))
                nn.add(layers.LSTM(d, return_sequences=True))
                nn.add(layers.GaussianNoise(S[-1]))
                nn.add(layers.Dropout(0.25))
                nn.add(layers.LSTM(d, return_sequences=True))
                nn.add(layers.GaussianNoise(S[-1]))
                nn.add(layers.Dropout(0.25))
                nn.add(layers.LSTM(d, return_sequences=True))
                nn.add(layers.GaussianNoise(S[-1]))
                nn.add(layers.Dropout(0.25))
                nn.add(layers.LSTM(d))
                nn.add(layers.Dense(1))
                nn.compile(loss="mean_squared_error", 
                    optimizer=optimizers.Adam(learning_rate=L[-1]))
            pipe = Pipeline([("model", nn)])
            node_id = fl_model.get_next_node_id()
            agg_w = x.shape[0]
            new_node = FederatedNode().initialize((x,y), cols, "RUL", pipe, 
                                                network_weight_update,
                                                lambda x: x["model"].get_weights(), 
                                                node_id, agg_weight=agg_w) 
            fl_model.add_node(new_node)
        # continue
        if IMPORT_MODEL:
            break
        scores = fl_model.fit()
        avg_score = [sum(s)/float(len(s)) for s in scores]
        if avg_score[-1] <= best_score:
            best_score = avg_score[-1]
            best_params = {"dw": dw, "g": g, "t": t}
        # best_scores.append(avg_score[-1])
        r = [i for i in range(R)]
        #plt.plot(r, avg_score, label=f"deg window = {dw}, std = {s}, epochs = {e}")
        plt.plot(r, avg_score, label=f"w = {dw}, g = {g}, t = {t}")
    
    if not IMPORT_MODEL:
        plt.xlabel("Aggregation Rounds")
        plt.ylabel("Node-Averaged Validation RMSE")
        plt.legend()
        plt.grid()
        plt.show()
    
    abs_errors = []
    print(f"Best params = {best_params}")
    for i in range(N):
        fl_model.nodes[i].model["model"].save(f"n{i+1}_model.keras")
        test_set = pd.read_csv(f"data\\test_FD00{i+1}.csv")
        train_set = pd.read_csv(f"data\\train_FD00{i+1}.csv")
        x_test, y_test, _, _ = run_pipeline(test_set[cols+["unit","RUL"]], 
                                        cols, W[-1], F[-1], M[-1], 
                                        4, DW[-1], scalers[i], pcas[i],
                                        g=G[-1], thresh=T[-1])
        x_train, y_train, _, _ = run_pipeline(train_set[cols+["unit","RUL"]], 
                                        cols, W[-1], F[-1], M[-1], 
                                        4, DW[-1], scalers[i], pcas[i],
                                        g=G[-1], thresh=T[-1])
        y_pred = fl_model.predict(x_test, i)
        y_train_pred = fl_model.predict(x_train, i)
        y_pred = y_pred.reshape((y_pred.shape[0],))
        # y_pred = medfilt(y_pred, kernel_size=11)
        y_train_pred = y_train_pred.reshape((y_train_pred.shape[0],))
        # y_train_pred = medfilt(y_train_pred, kernel_size=11)

        # getting error vs RUL trend
        errs = [abs(t-p)/t for t, p in zip(y_test,y_pred)]

        for j, val in enumerate(y_pred):
            abs_errors.append((val,errs[j]))

        print(f"FD00{i+1} Train Set Performance:")
        print(f"RMSE = {mean_squared_error(y_train, y_train_pred, squared=False)}")
        print(f"R^2 = {r2_score(y_train, y_train_pred)}")    
        print(f"FD00{i+1} Test Set Performance:")
        print(f"RMSE = {mean_squared_error(y_test, y_pred, squared=False)}")
        print(f"R^2 = {r2_score(y_test, y_pred)}")
        plt.clf()
        plt.plot(y_pred, label="Pred")
        plt.plot(y_test, label="True")
        plt.xlabel("Elapsed Time (cycles)")
        plt.ylabel("RUL (cycles)")
        plt.legend()
        plt.grid()
        plt.show()
    
    # rul_x = [x[0] for x in abs_errors]
    # err_y = [x[1] for x in abs_errors]
    
    # plt.clf()
    # plt.scatter(rul_x[::10], err_y[::10])
    # plt.xlabel("Predicted RUL (cycles)")
    # plt.ylabel("Absolute Percentage Error")
    # plt.grid()
    # plt.show()
