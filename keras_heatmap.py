import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense
from keras.optimizers import RMSprop, SGD
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# データの作成
df = pd.read_csv('growth_rate.csv')
X = df.iloc[:, 3:].values
x_train = X[:400]
t_train = df.sex[:400].replace('Male', 0).replace('Female', 1)
x_test = X[400:]
t_test = df.sex[400:].replace('Male', 0).replace('Female', 1)
print(x_train.shape, t_train.shape)

# モデルの定義
model = Sequential()
model.add(Dense(activation='sigmoid', input_dim=8, units=128))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=0.05), metrics=['accuracy'])

# 学習開始
epoch_num = 500
history = model.fit(x_train, t_train, batch_size=64,
                    epochs=epoch_num, verbose=1, validation_split=0.1)

# 学習率のグラフ
plt.plot(range(1, epoch_num+1), history.history['loss'], label='training')
plt.plot(range(1, epoch_num+1),
         history.history['val_loss'], label='validation')
plt.show()

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
z = model.predict(invX)
z = z.reshape(2*grange, 2*grange)
plt.contourf(px, py, z, alpha=1.0, cmap='coolwarm')
Y = model.predict(X)
col_sex = ['b' if y < 0.5 else 'r' for y in Y]
plt.scatter(tx[:, 0], tx[:, 1], color=col_sex)
plt.ylim((-grange, grange))
plt.xlim((-grange, grange))
plt.show()
