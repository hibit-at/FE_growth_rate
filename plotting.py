import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import plotting

#オリジナルデータを読み込む
df = pd.read_csv('growth_rate.csv')

#後で性別のカラーラベルに使用するためにbとrに置き換えたものを作成
col_sex = df['sex'].replace('Male','b').replace('Female','r')

plotting.scatter_matrix(df.iloc[:, 3:], figsize=(8, 8), c = col_sex, alpha=.2)
plt.show()