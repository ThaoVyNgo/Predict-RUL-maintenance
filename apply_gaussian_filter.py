import pandas as pd
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

FILTER_LEN = {"FD001": 11, "FD002": 99, "FD003": 11, "FD004": 99}
FILTER_STD = {"FD001": {'lpc_out_temp': 1.3894736842105266, 
                        'hpc_out_temp': 1.6473684210526318, 
                        'lpt_out_temp': 0.8736842105263158, 
                        'hpc_out_press': 0.8736842105263158, 
                        'fan_speed': 0.6157894736842106, 
                        'core_speed': 0.35789473684210527, 
                        'hpc_stat_press': 0.6157894736842106, 
                        'flow_press_ratio': 0.8736842105263158, 
                        'corr_fan_speed': 0.6157894736842106, 
                        'corr_core_speed': 0.1, 
                        'bypass_ratio': 1.1315789473684212, 
                        'bleed_enthalpy': 1.3894736842105266, 
                        'hpt_bleed': 1.1315789473684212, 
                        'lpt_bleed': 1.1315789473684212},
              "FD002": {'lpc_out_temp': 1.1315789473684212, 
                        'hpc_out_temp': 1.1315789473684212, 
                        'lpt_out_temp': 1.1315789473684212, 
                        'hpc_out_press': 0.1, 
                        'fan_speed': 1.6473684210526318, 
                        'core_speed': 0.8736842105263158, 
                        'hpc_stat_press': 0.8736842105263158, 
                        'flow_press_ratio': 5.0, 
                        'corr_fan_speed': 1.6473684210526318, 
                        'corr_core_speed': 0.6157894736842106, 
                        'bypass_ratio': 1.1315789473684212, 
                        'bleed_enthalpy': 1.1315789473684212, 
                        'hpt_bleed': 0.8736842105263158, 
                        'lpt_bleed': 0.8736842105263158}}

def compute_weights(length=3, sigma=1.0, limit=3.0):
    if (length % 2) == 0:
        print(f"Kernel size must be odd, adjusting to {length+1}")
        length += 1
    x = np.linspace(-limit, limit, length)
    kernel = stats.norm.pdf(x, scale=sigma)
    
    # Return the normalized kernel so that the sum is 1
    return kernel / kernel.sum()

def apply_filter(df, cols, kernel):
    kernel_size = len(kernel)
    for col in cols:
        vals = df[col].values
        L = len(vals)
        pad_size = kernel_size // 2
        frontpad =  np.full(pad_size, vals[0])              # compute the front and back padding vectors
        backpad = np.full(pad_size, vals[-1])
        vals = np.concatenate((frontpad, vals, backpad))
        offset = kernel_size - 1                            # filter window offset
        avg = kernel[-1] * vals[offset:L+offset]            # last term
        
        for i in range(offset):                             # rest of the terms 
            avg += (kernel[i] * vals[i:L+i])

        df.loc[:,col] = avg                                 # replace noisy signal with filtered signal
    return df

def plot_sensor_failure_trend(df, unit, cols, node):
    df = apply_filter(df, cols, 
                      compute_weights(length=101, sigma=2.0))
    udf = df[df["unit"] == unit]
    rul = np.array(udf["RUL"].tolist()) * -1
    for col in cols:
        data = np.array(udf[col].tolist()).reshape((-1,1))
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        plt.plot(rul, data, label=col)
    plt.title(f"Node {node}, Unit {unit} Failure Trend")
    plt.xlabel("-1*RUL")
    plt.ylabel("Standard Deviations")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    path = input("Enter file path: ")
    df = pd.read_csv(path)
    node = path.split("_")[-1].split(".")[0]

    # old names
    # cols = ["SM2", "SM3", "SM4", "SM6", "SM7", "SM8", "SM9", 
    #        "SM11", "SM12", "SM13", "SM14", "SM15", "SM17", 
    #        "SM20", "SM21"]

    cols = ["lpc_out_temp", "hpc_out_temp", "lpt_out_temp", "hpc_out_press", 
            "fan_speed", "core_speed", "hpc_stat_press", "flow_press_ratio", 
            "corr_fan_speed", "corr_core_speed", "bypass_ratio", "bleed_enthalpy", 
            "hpt_bleed", "lpt_bleed"]
    
    cmd = int(input("Generate sensor failure graph (1), signal filter demo (2), or filter sigma sweep (3)? "))
    if cmd == 1:
        skip = 40
        units = list(df["unit"].unique())[::skip]
        for u in units:
            plot_sensor_failure_trend(df, u, cols, node)
    elif cmd == 2:
        N = 1000
        S = 7
        df = df.iloc[:N,:]
        kernel_sizes = [3,11,33,101,1001]
        # sigmas = np.linspace(0.1, 8.0, S)
        # for s in sigmas:
        #     x = [i for i in range(kernel_size)]
        #     plt.plot(x, compute_weights(kernel_size, s), label=f"Sigma = {s}")
        # plt.legend()
        # plt.show()
        print(df.columns.values)
        col = input("Select a column from the choices above: ")
        x_series = [i for i in range(df.shape[0])]
        prows = 2
        pcols = 3
        fig = plt.figure()
        # plt.title(f"Filtering for {node}:{col}, kernel size = {kernel_size}")
        # plt.xlabel("Cycles")
        # plt.ylabel(f"Signal Value")
        ax0 = fig.add_subplot(prows, pcols, 1)
        ax0.plot(x_series, df[col].values)
        ax0.set_title("Unfiltered")
        ax0.set(xlabel="Cycles", ylabel=col)
        ax0.label_outer()
        for i, k in enumerate(kernel_sizes):
            temp_df = df.copy()
            ys = apply_filter(temp_df, [col], compute_weights(k, 3.0))
            rho = ys.corr().loc["RUL",col]
            ax = fig.add_subplot(prows, pcols, i+2)
            ax.plot(x_series, ys[col].values)
            ax.set_title(f"Kernel Size = {k}, rho = {rho:.2f}")
            ax.set(xlabel="Cycles", ylabel=col)
            ax.label_outer()
        plt.show()
    else:
        sigmas = np.linspace(0.1, 5.0, 6)
        kernel_sizes = [3, 11, 99, 999]
        # kernel_sizes = [FILTER_LEN[node]]
        best_sigmas = {}

        for col in cols:
            for kernel_size in kernel_sizes:
                orig_corr = df.corr().loc["RUL",col]
                corrs = [orig_corr]
                x_series = [0] + list(sigmas)
                best_sigma = None
                best_corr = orig_corr
                for s in sigmas:
                    temp_df = df.copy()
                    y = apply_filter(temp_df, [col], compute_weights(kernel_size, s))
                    corr = y.corr().loc["RUL",col]
                    if abs(corr) > abs(best_corr):
                        best_sigma = s
                        best_corr = corr
                    corrs.append(corr)
                best_sigmas[col] = best_sigma
                plt.cla()
                plt.clf()
                plt.title(f"Filtering for {node}:{col}, kernel size = {kernel_size}")
                plt.plot(x_series, corrs)
                plt.text(best_sigma, best_corr, f"Sigma = {best_sigma:.2f}")
                plt.xlabel("Filter Sigma")
                plt.ylabel(f"{col} Correlation to RUL")
                plt.show()
                # plt.savefig(f"{node}_{col}_filter_sweep.png")
        print(best_sigmas)