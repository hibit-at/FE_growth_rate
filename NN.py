import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#オリジナルデータを読み込む
df = pd.read_csv('growth_rate.csv')

#ここでニューラルネットワーク
X = df.iloc[:,3:]
y = df['sex']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state = 0)
mlp = MLPClassifier(hidden_layer_sizes=[10,10,10], random_state=0)
mlp.fit(X_train, y_train)

#判別のスコア
print(mlp.score(X_train, y_train))
print(mlp.score(X_test, y_test))