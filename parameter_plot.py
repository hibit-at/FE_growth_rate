import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('growth_rate.csv')
df = df[df['series'] == 'TH']  # シリーズを限定するのをオススメします

Axis1 = list(df['POW']+df['TEC'])
Axis2 = list(df['DEF']+df['SPD'])
col_sex = df['sex'].replace('Male', 'b').replace('Female', 'r')

check = [[0]*201 for _ in range(201)]

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.scatter(Axis1, Axis2, alpha=0)  # 空のマーカー
# 下のイテレーションでテキストラベルを貼り付ける
for i, (name, PC1, PC2, col) in enumerate(zip(df['name'], Axis1, Axis2, col_sex)):
    check[PC1][PC2] += 1
    PC2 -= check[PC1][PC2] - 1
    ax.annotate(name, (PC1, PC2), c=col, fontname="MS Gothic")

plt.show()
