import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data\\train_FD004.csv') 
df = df.iloc[:,3:-2]
new_column_names = ['OS1','OS2','OS3','SM1','SM2','SM3','SM4',
                    'SM5', 'SM6','SM7','SM8','SM9','SM10','SM11','SM12','SM13','SM14','SM15',
                    'SM16','SM17','SM18','SM19','SM20','SM21']

df.columns = new_column_names
variables = df.columns

num_plots = len(variables)
num_cols = 4
num_rows = -(-num_plots // num_cols)

fig, axs = plt.subplots(num_rows,num_cols,figsize=(20,15))

for i, variable in enumerate(variables):
    ax = axs[i // num_cols, i%num_cols]
    ax.plot(df[variable])
    #ax.set_title(f'Plot of {variable}')
    #ax.set_xlabel('Index')
    ax.set_ylabel(f'{variable}')
    ax.grid(True)

for i in range(num_plots,num_rows*num_cols):
    fig.delaxes(axs[i//num_cols, i%num_cols])

#plt.ylim(variables.min(),variables.max())
#plt.xlim(0,21000)
plt.tight_layout()
plt.show()
