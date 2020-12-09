import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.decomposition import PCA
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# Pytorch ... GPUが使えるならGPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)  # 使えるなら「cuda:0」

# データの作成
df = pd.read_csv('growth_rate.csv')
X = df.iloc[:, 3:].values
x_train = X[:400]
t_train = df.sex[:400].replace('Male', 0).replace('Female', 1)
x_test = X[400:]
t_test = df.sex[400:].replace('Male', 0).replace('Female', 1)

# データ変換　numpy.array -> torch.Tensor
x_train = torch.Tensor(x_train)
t_train = torch.Tensor(t_train.values).reshape(-1, 1)
x_test = torch.Tensor(x_test)
t_test = torch.Tensor(t_test.values).reshape(-1, 1)

# データ変換　CPU -> GPU
x_train = x_train.to(device)
t_train = t_train.to(device)
x_test = x_test.to(device)
t_test = t_test.to(device)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

dataset = TensorDataset(x_train, t_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# モデルの定義
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.BatchNorm1d(64),
    nn.Sigmoid(),
    nn.Linear(64, 1),
    nn.Sigmoid(),
)
model.to(device)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 学習開始
loss_train = []
loss_test = []
epoch_num = 1000
for epoch in tqdm(range(epoch_num)):
    model.train()
    for x, t in loader:
        x.to(device)
        optimizer.zero_grad()
        y = model(x)
        loss = loss_fn(y, t)
        loss.backward()
        optimizer.step()
    loss_train.append(loss_fn(model(x_train), t_train))
    loss_test.append(loss_fn(model(x_test), t_test))

# 学習率のグラフ
x = np.arange(1, epoch_num+1, 1)
plt.plot(x, loss_train)
plt.plot(x, loss_test)
plt.show()

# ヒートマップ描画
plt.figure(figsize=(10, 10))
grange = 150
pca = PCA(n_components=2)
tx = pca.fit_transform(X)
px, py = np.meshgrid(np.arange(-grange, grange, 1),
                     np.arange(-grange, grange, 1))
x = px.reshape((4*grange*grange, 1))
y = py.reshape((4*grange*grange, 1))
xy = np.concatenate([x, y], 1)
invX = pca.inverse_transform(xy)
z = model(torch.Tensor(invX).to(device))  # CPU -> GPU
z = z.reshape(2*grange, 2*grange)
z = z.to('cpu').detach().numpy()  # GPU -> CPU
plt.contourf(px, py, z, alpha=1.0, cmap='coolwarm')
Y = model(torch.Tensor(X).to(device))  # CPU -> GPU
Y = Y.to('cpu').detach().numpy()  # GPU -> CPU
col_sex = ['b' if y < 0.4 else 'r' for y in Y]
plt.scatter(tx[:, 0], tx[:, 1], color=col_sex)
plt.ylim((-grange, grange))
plt.xlim((-grange, grange))
plt.show()
