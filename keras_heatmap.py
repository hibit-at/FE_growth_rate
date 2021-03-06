import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Activation, BatchNormalization, Dense
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop
from sklearn.decomposition import PCA
from tensorflow.python.client import device_lib

# Tensorflow ... GPUが使えるかの確認
print(device_lib.list_local_devices())

# 時間計測
start = time.time()

# データの作成
df = pd.read_csv('growth_rate.csv')
X = df.iloc[:, 3:].values
x_train = X[:400]
t_train = df.sex[:400].replace('Male', 0).replace('Female', 1)
x_test = X[400:]
t_test = df.sex[400:].replace('Male', 0).replace('Female', 1)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

# モデルの定義
model = Sequential()
model.add(Dense(64, input_dim=8))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=0.01), metrics=['accuracy'])


# 学習開始
epoch_num = 1000
history = model.fit(x_train, t_train, batch_size=64,
                    epochs=epoch_num, verbose=1, validation_data=(x_test, t_test))

# 経過時間
print("経過時間: {0} [sec]".format(time.time()-start))

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
col_sex = ['b' if y < 0.4 else 'r' for y in Y]
plt.scatter(tx[:, 0], tx[:, 1], color=col_sex)
plt.ylim((-grange, grange))
plt.xlim((-grange, grange))
plt.show()
