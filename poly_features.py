import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    path = input("Enter file path: ")
    df = pd.read_csv(path)
    ncomps = 4
    pca = PCA(whiten=True, n_components=ncomps)
    new_df = pca.fit_transform(df.drop(["unit","cycles","RUL"], axis=1))
    cols = []
    for i in range(1, ncomps+1):
        cols.append(f"PC{i}")
    new_df = pd.DataFrame(new_df, columns=cols)
    print(new_df)
    poly = PolynomialFeatures(degree=2)
    new_df = poly.fit_transform(new_df)
    names = poly.get_feature_names_out()
    new_df = pd.DataFrame(new_df[:,1:], columns=names[1:])
    new_df["RUL"] = df["RUL"]
    print(new_df)
    sns.heatmap(new_df.corr())
    title = path.split("_")[-1].split(".")[0]
    plt.title(f"{title} PCA+PolyFeatures Heatmap")
    plt.show()
