import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def MedianFilter(df, cols, kernel=7):
    for col in cols:
        df.loc[:,col] = medfilt(df[col].to_numpy(), kernel_size=kernel)
    return df

if __name__ == "__main__":
    data = pd.read_csv('data\\train_FD001.csv')
    #df = pd.DataFrame(data)
    #print(df.head())
    #extract_data = df[df['unit'] == 1]
    #print(extract_data.head())
    #number_sensors = df.shape[1]
    #number_engines = df['unit'].nunique()
    #number_datapoints = len(df)
    #sensor = extract_data.shape
    #print(sensor)
    #new_shape = (sensor[0], sensor[1], 1)
    #data_array = extract_data.values
    #reshaped_data = data_array.reshape(new_shape)
    #print(reshaped_data.shape)


    window = 80
    filtered_data = MedianFilter(data.values,window)
    #print(filtered_data)
    plt.plot(data['hpc_out_temp'], label='Original Data')
    #filtered = pd.DataFrame(filtered_data)
    #print(filtered)
    #plt.plot(filtered_data[:,7], label='Filtered Data')

    #plt.show()

    def detect_outliers(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df <(Q1-1.5*IQR)) | (df > (Q3 +1.5*IQR))
        return outliers


    outliers = detect_outliers(data)
    #print(outliers)
    count = outliers.sum().sum()
    print(count)


    fildf = pd.DataFrame(filtered_data)
    #print(fildf.head)
    outliersfiltered = detect_outliers(fildf)
    countFil = outliersfiltered.sum().sum()
    print(countFil)
    #print(outliersfiltered)
    #outliersfiltered.to_csv('Filter_Outliers_Train1.csv')


