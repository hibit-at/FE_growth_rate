import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

# データの作成
df = pd.read_csv('growth_rate.csv')
X = df.iloc[:, 3:].values
x_train = X[:400]
t_train = df.sex[:400].replace('Male', 0).replace('Female', 1)
x_test = X[400:]
t_test = df.sex[400:].replace('Male', 0).replace('Female', 1)
print(x_train.shape, t_train.shape)

# ここで線形判別分析
X = df.iloc[:, 3:]
y = df['sex'].replace('Male', 1).replace('Female', 0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# ヒートマップ
grange = 150
plt.figure(figsize=(10, 10))
pca = PCA(n_components=2)
tx = pca.fit_transform(X)
px, py = np.meshgrid(np.arange(-grange, grange, 1),
                     np.arange(-grange, grange, 1))
x = px.reshape((4*grange*grange, 1))
y = py.reshape((4*grange*grange, 1))
xy = np.concatenate([x, y], 1)
invX = pca.inverse_transform(xy)
z = lda.transform(invX)
z = z.reshape(2*grange, 2*grange)
plt.contourf(px, py, z, alpha=1.0, cmap='coolwarm')
Y = lda.predict(X)
col_sex = ['b' if y == 0 else 'r' for y in Y]
plt.scatter(tx[:, 0], tx[:, 1], color=col_sex)
plt.ylim((-grange, grange))
plt.xlim((-grange, grange))
plt.show()
