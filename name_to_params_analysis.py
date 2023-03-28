import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# カタカナ文字セット
katakana = [chr(i) for i in range(ord("ァ"), ord("ヺ")+1)]
katakana.append('ー')


# 文字列をカタカナのインデックスに変換
def name_to_index(name):
    indices = []
    for char in name:
        index = katakana.index(char)
        indices.append(index)
    return indices

# インデックスをワンホットエンコーディングに変換


def one_hot_encoding(indices, max_length=6, num_classes=len(katakana)):
    encoded = np.zeros((max_length, num_classes))
    for i, idx in enumerate(indices):
        if i < max_length:
            encoded[i, idx] = 1.0
    return encoded

# 文字列からテンソルに一気に変換する関数


def name_to_tensor(word):
    vector = one_hot_encoding(name_to_index(word))
    return torch.Tensor(vector)


class MyDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # タイトル行をスキップする
            next(reader)
            for row in reader:
                name = row[0].split('（')[0]
                tensor = name_to_tensor(name)
                params = [min(float(row[i])/100, 1.0)
                          for i in range(3, 11)]  # row[3]からrow[10]までをfloatに変換
                params = torch.tensor(params)  # Tensorに変換
                self.data.append((name, tensor, params))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        name, tensor, params = self.data[index]
        # nameを補助的な情報、tensorを入力、params出力とするタプルを返す
        return name, tensor, params


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.fc1(out[:, -1, :]))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


file_path = 'growth_rate.csv'
input_size = 91  # 名前の最大長
hidden_size = 64  # 隠れ層のユニット数
num_layers = 3  # LSTMのレイヤー数
output_size = 8  # 出力層のユニット数
lr = 0.001  # 学習率

# データセットとデータローダーを作成する
dataset = MyDataset(file_path)
train_size = len(dataset) - 32
test_size = 32
generator = torch.Generator().manual_seed(0)
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size], generator=generator)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
net = LSTMNet(input_size, hidden_size, num_layers, output_size)

FILE_PATH = "./name_to_params.pth"


def train():
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=lr)

    # ロスの履歴を保存するためのリスト
    train_loss_history = []

    # モデルを学習する
    num_epochs = 1000
    for epoch in range(num_epochs):
        for i, (name, inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                log = []
                log.append('Epoch [{}/{}]'.format(epoch+1,num_epochs))
                log.append('Step [{}/{}]'.format(i+1,len(dataset)))
                log.append('Loss : {:.4f}'.format(loss.item()))
                print(', '.join(log))
            # ロスを履歴に追加する
            train_loss_history.append(loss.item())

    torch.save(net.state_dict(), FILE_PATH)

    # ロスのグラフを描画して保存する
    fig, ax = plt.subplots()
    ax.plot(train_loss_history)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title('Training Loss')
    fig.savefig('train_loss_name_to_params.png')


def eval():

    if os.path.exists(FILE_PATH):
        net.load_state_dict(torch.load(FILE_PATH))
        print("Saved weight loaded.")
        for i, (names, inputs, targets) in enumerate(test_dataloader):
            predicts = net(inputs)
            for name, predict in zip(names, predicts):
                predict = predict.tolist()
                predict = {k: int(v*100)
                    for k, v in zip(['HP', 'POW', 'MAG', 'TEC', 'SPD', 'LUC', 'DEF', 'RES'], predict)}
                print(f'{name} {predict}')

    else:
        print("No saved weight.")


def test(name):
    if os.path.exists(FILE_PATH):
        net.load_state_dict(torch.load(FILE_PATH))
        tensor = name_to_tensor(name)
        tensor = tensor.view(1, 6, 91)
        print(tensor.shape)
        predict = net(tensor)
        predict = predict.tolist()[0]
        predict = {k: int(v*100)
            for k, v in zip(['HP', 'POW', 'MAG', 'TEC', 'SPD', 'LUC', 'DEF', 'RES'], predict)}
        print(f'{name} {predict}')
    else:
        print("No saved weight.")


def api_score(name):
    if os.path.exists(FILE_PATH):
        net.load_state_dict(torch.load(FILE_PATH))
        tensor = name_to_tensor(name)
        tensor = tensor.view(1, 6, 91)
        predict = net(tensor)
        predict = predict[0].tolist()
        return  sum(predict)*100
    else:
        print("No saved weight.")
        return 0

def heuristic(base_name,reverse_flag):
    print(base_name, api_score(base_name))
    if len(base_name) == 6:
        return
    candidates = []
    for k in katakana:
        for n in katakana:
            name = base_name + k + n
            score = api_score(name)
            candidates.append((score, k + n))
    candidates.sort(reverse=reverse_flag)
    base_name = base_name + candidates[0][1]
    heuristic(base_name,reverse_flag)


if __name__ == '__main__':
    # train()
    # eval()
    # test('ヒビトス')
    # test('ヒビリア')
    heuristic('',True)
    heuristic('',False)
