import pandas as pd

def get_numeric_data_report(df):
    num_df = df.select_dtypes("number")
    rows = num_df.shape[0]
    qual_rep = num_df.describe()
    missing = {"n": [], "p": []}
    unique = {"n": [], "p": []}
    for col in num_df.columns.values:
        n_miss = num_df[col].isna().sum()
        p_miss = float(n_miss) / rows
        missing["n"].append(n_miss)
        missing["p"].append(p_miss)
        n_unique = num_df[col].nunique()
        p_unique = float(n_unique) / rows
        unique["n"].append(n_unique)
        unique["p"].append(p_unique)
    qual_rep.loc["n_missing"] = missing["n"]
    qual_rep.loc["missing_pct"] = missing["p"]
    qual_rep.loc["n_unique"] = unique["n"]
    qual_rep.loc["unique_pct"] = unique["p"]
    return qual_rep

if __name__ == "__main__":
    file = input("Please enter the file path: ")
    df = pd.read_csv(file)
    data_name = file.split("\\")[-1]
    data_rep = get_numeric_data_report(df)
    print(data_rep)
    data_rep.to_csv(f"data\\{data_name}_Report.csv")