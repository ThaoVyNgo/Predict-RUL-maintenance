from math import ceil, pow
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
import tensorflow as tf
from itertools import product

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
                                     self.model, epochs=epochs, k=3)
        # print(res)
        return res.iloc[-1,score_idx]

    def broadcast(self):
        return self.weight_func(self.model)

class FederatedModel:
    def __init__(self, num_rounds, agg_func, train_size=1.0,
                 epochs=None, p=1.0, score_idx=-1):
        self.num_nodes = 0
        self.num_rounds = num_rounds
        self.agg_func = agg_func
        self.train_size = train_size
        self.epochs = epochs
        self.p = p
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

    # def predict(self, X):
    #     pred = X @ np.array(self.weights[0])        # first layer prediction
    #     for layer_weights in self.weights[1:]:      # remaining layers
    #         pred = pred @ np.array(layer_weights)
    #     return pred

    def train_round(self, round):
        if self.train_size < 1.0:
            N = ceil(self.train_size * self.num_nodes)
            train_node_set = sample(self.nodes, N)
        else:
            train_node_set = self.nodes
        for _, node in enumerate(train_node_set):
            print(f"Training node {node.id}")
            s = node.fit(self.weights, self.epochs, self.score_idx)
            self.scores[round].append(s)
            self.node_weights[node.id-1] = node.broadcast()

    def aggregate(self):
        self.weights = self.agg_func(self.nodes, self.node_agg_weights, self.p)

def linear_agg(node_weights, agg_weights, p=1.0):
    # n_layers = len(node_weights[0].weights)
    # n_nodes = len(node_weights)
    # n_neurons = len(node_weights[0][0])
    # n_feat = len(node_weights[0][0][0])
    # biases = [node_weights[n][l][1] for n in range(n_nodes) for l in range(0,len(node_weights[0]),2)]
    # print(biases)
    # print(len(biases))
    weights = [n.weights for n in node_weights]
    return list(np.average(weights, weights=agg_weights, axis=0))
    # for l in range(n_layers):
    #     layer_weights = []
    #     for n in range(n_neurons):
    #         neuron_weights = [node_weights[i][l][n] for i in range(n_nodes)]
    #         layer_weights.append(np.average(neuron_weights, weights=agg_weights))
    #     weights.append(layer_weights)
    # return weights

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
                        x = nodes[i].model["model"].get_weights()[wi][row][col]
                        agg_weight += ((nk[i]/n)*x)
                    weights[wi][row][col] = agg_weight
            except IndexError as idx_err:
                # print(idx_err)
                # print(f"Assuming flat array for shape {W0[wi].shape}")
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
                # print(idx_err)
                # print(f"Assuming flat array for shape {W0[wi].shape}")
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

def window_data(df, cols, window_size=10, filter_size=101):
    if filter_size > 0:
        df[cols] = GaussianFilter(cols, filter_size, 3.0).fit_transform(df[cols])
    df[cols] = StandardScaler().fit_transform(df[cols])
    n_samples = df.shape[0] - window_size + 1
    x = np.zeros((n_samples,window_size,df.shape[1]-1))
    y = np.zeros((n_samples,))
    for i in range(n_samples):
        n = i+window_size
        x[i] = df.iloc[i:n,:-1].to_numpy()
        y[i] = df.iloc[n-1,-1]
    return x, y

if __name__ == "__main__":
    # print(tf.config.list_physical_devices('GPU'))
    # quit()
    cols = ["cycles","hpc_stat_press","burner_fuel_ratio","flow_press_ratio",
            "corr_core_speed","lpt_out_temp","hpc_out_press","bypass_ratio",
            "lpt_bleed"]
    # cols = ["lpc_out_temp", "hpc_out_temp", "lpt_out_temp", "hpc_out_press", 
    #         "fan_speed", "core_speed", "hpc_stat_press", "flow_press_ratio", 
    #         "corr_fan_speed", "corr_core_speed", "bypass_ratio", "bleed_enthalpy", 
    #         "hpt_bleed", "lpt_bleed"]
    # cols = ["core_speed","corr_core_speed","lpt_out_temp"]
    # cols = ["hpc_stat_press","fan_speed"]
    # filter_len = {1: 11, 2: 99, 3: 11, 4: 99}
    print("Running in test mode")
    N = 4
    R = 50
    NCOMP = len(cols)
    P = [1.0] #[1.0,2.0,3.0]
    W =  [136]#[i for i in range(1,300,15)]#[10,30,100]
    F = [0]#[0,5,9]
    params = product(W,F)
    best_scores = []

    for w, f in params:
        print(f"params: w = {w}, L = {f}")
        fl_model = FederatedModel(R, power_weighted_mean, epochs=1, p=1.0)

        for i in range(N):
            df = pd.read_csv(f"data\\train_FD00{i+1}.csv")
            # print(df.corr().iloc[:,-1].sort_values())
            # continue
            # df = pad_data(df, time_col="cycles", sample_col="unit", pad_val=0.)
            df = df[cols+["RUL"]]
            x, y = window_data(df, cols, window_size=w, filter_size=f)
            # print("Node " + str(i+1))
            # print(x.shape)
            # print(y.shape)
            # continue
            # print(df.cov().iloc[1:,-1])
            # print(df)
            # continue
            nn = models.Sequential()
            # nn.add(Input(shape=(len(cols),)))
            # nn.add(layers.Dense(4, activation="relu"))
            # nn.add(layers.MaxPool1D(4))
            # nn.add(layers.Dense(16, activation="relu"))
            # nn.add(layers.MaxPool1D(2))
            # nn.add(layers.Dense(2, activation="relu"))
            # nn.add(layers.Dense(4, activation="relu"))
            # nn.add(layers.Masking(0.))
            nn.add(layers.LSTM(20, input_shape=(x.shape[1],x.shape[2]), return_sequences=True))
            nn.add(layers.Dropout(0.1))
            nn.add(layers.LSTM(20, return_sequences=True))
            nn.add(layers.Dropout(0.1))
            # nn.add(layers.LSTM(20, return_sequences=True))
            # nn.add(layers.Dropout(0.1))
            nn.add(layers.LSTM(20))
            # nn.add(layers.Dropout(0.2))
            nn.add(layers.Dense(1))
            nn.compile(loss="mean_squared_error", 
                optimizer=optimizers.Adam(learning_rate=0.001))
            # print(sorted([df[df["unit"] == u].shape[0] for u in list(df["unit"].unique())]))
            # continue
            pipe = Pipeline([# ("filter", GaussianFilter(cols, 101, 3.0)),
                             # ("scaler", StandardScaler()),
                             # ("reshaper", DataShaper()),
                             ("model", nn)])
            node_id = fl_model.get_next_node_id()
            agg_w = x.shape[0] # np.nanmax([abs(c) for c in df.corr().iloc[-1,1:-1].values])
            new_node = FederatedNode().initialize((x,y), cols, "RUL", pipe, 
                                                network_weight_update,
                                                lambda x: x["model"].get_weights(), 
                                                node_id, agg_weight=agg_w) 
            fl_model.add_node(new_node)
        # continue
        scores = fl_model.fit()
        avg_score = [sum(s)/float(len(s)) for s in scores]
        # best_scores.append(avg_score[-1])
        r = [i for i in range(R)]
        plt.plot(r, avg_score, label=f"w = {w}, L = {f}")
    # plt.plot(W, best_scores)
    # plt.xlabel("Window Size")
    # plt.ylabel("Best Validation RMSE (Node-Averaged)")
    # plt.grid()
    # plt.show()
    plt.xlabel("Aggregation Rounds")
    plt.ylabel("Node-Averaged Validation RMSE")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig("latest.png")