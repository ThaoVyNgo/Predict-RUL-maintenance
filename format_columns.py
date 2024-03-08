import pandas as pd

def format_cols(df, names):
    df = df.iloc[:,1:-2]
    df.columns = names
    return df

if __name__ == "__main__":
    path = input("Enter file path: ")
    df = pd.read_csv(path)
    names = ["unit", "cycles", "OS1", "OS2", "OS3", "fan_in_temp", "lpc_out_temp", "hpc_out_temp", "lpt_out_temp",
             "fan_in_press", "bypass_press", "hpc_out_press", "fan_speed", "core_speed", "epr", "hpc_stat_press", 
             "flow_press_ratio", "corr_fan_speed", "corr_core_speed", "bypass_ratio", "burner_fuel_ratio", 
             "bleed_enthalpy", "dmd_fan_speed", "dmd_corr_fan_speed", "hpt_bleed", "lpt_bleed", "RUL"]
    
    # these are the old names
    # for i in range(1,4):
    #     names.append(f"OS{i}")
    
    # for i in range(1,22):
    #     names.append(f"SM{i}")

    # df = format_cols(df, names)
    df.columns = names
    print(df.columns.values)
    print(df)
    df.to_csv(path, index=False)

    