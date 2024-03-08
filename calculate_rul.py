import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def calc_rul(df):
    df_list = []
    for _, gdf in df.groupby("unit"):
        final_cycle = gdf["cycles"].max()
        gdf["RUL"] = final_cycle - gdf["cycles"]
        df_list.append(gdf)
    return pd.concat(df_list)

def rul_stats(df):
    tbf_list = []
    for _, gdf in df.groupby("unit"):
        tbf_list.append(gdf["RUL"].max())
    tbf_list = np.array(tbf_list)
    print(f"Min = {min(tbf_list)}")
    print(f"Max = {max(tbf_list)}")
    print(f"Mean = {tbf_list.mean()}")
    print(f"St Dev = {tbf_list.std()}")
    iqr = np.quantile(tbf_list, 0.75) - np.quantile(tbf_list, 0.25)
    print(f"IQR = {iqr}, 1.5*IQR = {1.5*iqr}")

def compile_test_rul(df, rul_list):
    rul_index = 0
    df_list = []
    for _, gdf in df.groupby("unit"):
        max_rul = rul_list[rul_index] + gdf["cycles"].max()
        rul_index += 1
        gdf["RUL"] = max_rul - gdf["cycles"]
        df_list.append(gdf)
    return pd.concat(df_list)

if __name__ == "__main__":
    test = int(input("Is this a test file (1:Yes/0:No)? "))
    path = input("Enter file path: ")
    df = pd.read_csv(path)
    if test == 1:
        node = path.split("_")[-1].split(".")[0]
        rul_path = f"data\\RUL_{node}.txt"
        try:
            rul_list = pd.read_csv(rul_path, header=None).iloc[:,0].tolist()
            df = compile_test_rul(df, rul_list)
        except FileNotFoundError:
            print(f"RUL file does not exist for {path}")
            exit()
    else:
        df = calc_rul(df)
        rul_stats(df)
            
    print(df)
    df.to_csv(path, index=False)
