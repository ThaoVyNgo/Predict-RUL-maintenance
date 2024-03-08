import pandas as pd

def convert_to_csv(file_path):
    df = pd.read_csv(file_path, sep=" ", header=None)
    new_path = file_path.split(".txt")[0]
    new_path += ".csv"
    df.to_csv(new_path, index=False)

if __name__ == "__main__":
    path = input("Enter file path: ")
    convert_to_csv(path)