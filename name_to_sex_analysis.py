import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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
                # 'Male'の場合は1、それ以外の場合は0をセットする
                sex = int(row[2] == 'Male')
                self.data.append((name, tensor, sex))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        name, tensor, sex = self.data[index]
        # nameを補助的な情報、tensorを入力、sexを出力とするタプルを返す
        return name, tensor, sex


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = torch.sigmoid(out)
        return out


file_path = 'growth_rate.csv'
input_size = 91  # 名前の最大長
hidden_size = 128  # 隠れ層のユニット数
num_layers = 3  # LSTMのレイヤー数
output_size = 1  # 出力層のユニット数
lr = 0.001  # 学習率

# データセットとデータローダーを作成する
dataset = MyDataset(file_path)
train_size = len(dataset) - 32
test_size = 32
generator = torch.Generator().manual_seed(0)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
net = LSTMNet(input_size, hidden_size, num_layers, output_size)

FILE_PATH = "./name_to_sex.pth"

def train():
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # ロスの履歴を保存するためのリスト
    train_loss_history = []

    # モデルを学習する
    num_epochs = 200
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
    fig.savefig('train_loss_name_to_sex.png')


def eval():

    if os.path.exists(FILE_PATH):
        net.load_state_dict(torch.load(FILE_PATH))
        print("Saved weight loaded.")
        for i, (names, inputs, targets) in  enumerate(test_dataloader):
            predicts = net(inputs)
            for name,predict in zip(names,predicts):
                print(f'{name} {predict.item():.4f}')

    else:
        print("No saved weight.")

def test(name):
    if os.path.exists(FILE_PATH):
        net.load_state_dict(torch.load(FILE_PATH))
        tensor = name_to_tensor(name)
        tensor = tensor.view(1,6,91)
        predict = net(tensor)
        print(f'{name} {predict.item():.4f}')
    else:
        print("No saved weight.")
    

if __name__ == '__main__':
    # train()
    # eval()
    test('ファヂァトャ')
    test('ヌッエナスン')
