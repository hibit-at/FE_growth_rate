import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import time
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 時間計測
start = time.time()

# データの作成
df = pd.read_csv('growth_rate.csv')
X = df.iloc[:, 3:].values
t = df['sex'].replace('Male', 1).replace('Female', 0)
x_train = X[:400]
t_train = df.sex[:400].replace('Male', 0).replace('Female', 1)
x_test = X[400:]
t_test = df.sex[400:].replace('Male', 0).replace('Female', 1)
print(x_train.shape, t_train.shape)

# ここで線形判別分析
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, t_train)

# 予測精度
print("経過時間: {0} [sec]".format(time.time()-start))
print("予測精度: {0}".format(accuracy_score(t_test, lda.predict(x_test))))

# ヒートマップ
grange = 150
plt.figure(figsize=(10, 10))
pca = PCA(n_components=2)
tx = pca.fit_transform(X)
px, py = np.meshgrid(np.arange(-grange, grange, 1),
                     np.arange(-grange, grange, 1))
x = px.reshape((4*grange*grange, 1))
y = py.reshape((4*grange*grange, 1))
xy = np.concatenate([px, py], 1)
invX = pca.inverse_transform(xy)
z = lda.transform(invX)
z = z.reshape(2*grange, 2*grange)
plt.contour(px, py, z, alpha=1.0, cmap='coolwarm')
Y = lda.predict(X)
col_sex = ['b' if y == 0 else 'r' for y in Y]
plt.scatter(tx[:, 0], tx[:, 1], color=col_sex)
plt.ylim((-grange, grange))
plt.xlim((-grange, grange))
plt.show()
