import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# dropping cols with low or no variance (< 1% uniqueness)
drop_cols = ["unit","cycles","OS3","SM1","SM5","SM10","SM16","SM18","SM19"]

#HEATMAPs
#Train_FD001
train1 = pd.read_csv('data\\train_FD001.csv').drop(drop_cols, axis=1).corr()
#train1 = train1.iloc[:,:-2]
sns.heatmap(train1)

plt.title('Heatmap of Train_FD001')
plt.show()

#Train_FD002
train2 = pd.read_csv('data\\train_FD002.csv').drop(drop_cols, axis=1).corr()
sns.heatmap(train2)

plt.title('Heatmap of Train_FD002')
plt.show()

#Train_FD003
train3 = pd.read_csv('data\\train_FD003.csv').drop(drop_cols, axis=1).corr()
sns.heatmap(train3)

plt.title('Heatmap of Train_FD003')
plt.show()

#Train_FD004
train4 = pd.read_csv('data\\train_FD004.csv').drop(drop_cols, axis=1).corr()
sns.heatmap(train4)

plt.title('Heatmap of Train_FD004')
plt.show()