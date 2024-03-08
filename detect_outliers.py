import pandas as pd

def detect_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df <(Q1-1.5*IQR)) | (df > (Q3 +1.5*IQR))
    return outliers

train1 = pd.read_csv('data\\train_FD001.csv')
outliers = detect_outliers(train1)
print(outliers)
outliers.to_csv('Outliers_Train1.csv')

train2 = pd.read_csv('data\\train_FD002.csv')
outliers = detect_outliers(train2)
print(outliers)
outliers.to_csv('Outliers_Train2.csv')

train3 = pd.read_csv('data\\train_FD003.csv')
outliers = detect_outliers(train3)
print(outliers)
outliers.to_csv('Outliers_Train3.csv')

train4 = pd.read_csv('data\\train_FD004.csv')
outliers = detect_outliers(train4)
print(outliers)
outliers.to_csv('Outliers_Train4.csv')