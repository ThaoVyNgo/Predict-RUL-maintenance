import apply_gaussian_filter as GF
import poly_features as PF
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# class TestTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         print("Fit called")
#         return self
#     def transform(self, X, y=None):
#         print("Transform called")
#         return X

class GaussianFilter(BaseEstimator, TransformerMixin):
    def __init__(self, cols, len=11, std=1.0):
        self.cols = cols
        self.len = len
        self.std = std
    def fit(self, X, y=None):
        self.kernel = GF.compute_weights(self.len, self.std)
        return self
    def transform(self, X, y=None):
        filtered_df = GF.apply_filter(X, self.cols, self.kernel)          
        return filtered_df
    
class DataShaper(BaseEstimator, TransformerMixin):
    def __init__(self, timesteps=1, ci=0, ri=0):
        self.timesteps = timesteps
        self.ci = ci
        self.ri = ri
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X[self.ri:,self.ci:]
        samples = X.shape[0] // self.timesteps
        features = X.shape[1]
        return X.reshape((samples,self.timesteps,features))
    
def generate_padded_row(df, pad_val=0.):
    row = {}
    for col in df.columns.values:
        row[col] = pad_val
    return row

def pad_data(df, time_col="cycles", sample_col="unit", pad_val=0.):
    dfs = []
    sample_size = df[time_col].max()
    print(f"Padding all samples to {sample_size} timesteps")
    row = generate_padded_row(df, pad_val)
    for name, group in df.groupby(sample_col):
        has_rows = group[time_col].max()
        add_rows = sample_size - has_rows
        row[sample_col] = name
        rows = [row] * add_rows
        padded_df = pd.concat([group, pd.DataFrame(rows)])
        dfs.append(padded_df)
    return pd.concat(dfs, ignore_index=True)