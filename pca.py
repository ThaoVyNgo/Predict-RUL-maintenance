from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def run_pca(df, hline=0.99):
    pca = PCA(whiten=True)
    pca.fit(df.drop(["unit","cycles","RUL"], axis=1))
    ncomp = pca.components_.shape[0]
    exp_var = np.cumsum(pca.explained_variance_ratio_)
    x = [i for i in range(ncomp)]
    plt.bar(x, exp_var, label="Cumulative Variance")
    plt.axhline(hline, color="r")
    plt.title("Explained Variance Plot")
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance Ratio")
    plt.show()

if __name__ == "__main__":
    path = input("Enter file path: ")
    df = pd.read_csv(path)
    run_pca(df)