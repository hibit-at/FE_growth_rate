import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA

# オリジナルデータを読み込む
df = pd.read_csv('growth_rate.csv')
df = df[df['series']=='TH']

# 後で性別のカラーラベルに使用するためにbとrに置き換えたものを作成
col_sex = df['sex'].replace('Male', 'b').replace('Female', 'r')

# ここで主成分分析
pca = PCA()
feature = pca.fit(df.iloc[:, 3:])
feature = pca.transform(df.iloc[:, 3:])

# 寄与率
print(pca.explained_variance_ratio_)

# ユニットの名前を散布図にする
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.scatter(feature[:, 0], feature[:, 1], alpha=0)  # 空のマーカー
# 下のイテレーションでテキストラベルを貼り付ける
for i, (name, PC1, PC2, col) in enumerate(zip(df['name'], feature[:, 0], feature[:, 1], col_sex)):
    ax.annotate(name, (PC1, PC2), c=col, fontname="MS Gothic")
plt.show()
