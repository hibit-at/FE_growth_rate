import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#オリジナルデータを読み込む
df = pd.read_csv('growth_rate.csv')

#後で性別のカラーラベルに使用するためにbとrに置き換えたものを作成
col_sex = df['sex'].replace('Male','b').replace('Female','r')

X = df.iloc[:,3:11]
y = df['sex']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
tree = DecisionTreeClassifier(random_state=0, max_depth=3)
tree.fit(X_train, y_train)

print(tree.score(X_train, y_train))
print(tree.score(X_test, y_test))

from sklearn.tree import export_graphviz

feature_label = df.columns[3:]
export_graphviz(tree, out_file='tree.dot', class_names=["Female","Male"], feature_names=feature_label, filled = True)