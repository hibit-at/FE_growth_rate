import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#オリジナルデータを読み込む
df = pd.read_csv('growth_rate.csv')

#ここで線形判別分析
X = df.iloc[:,3:11]
y = df['sex']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

#判別のスコア
print(lda.score(X_train, y_train))
print(lda.score(X_test, y_test))

#判別関数
print(lda.coef_)

plt.bar(height = lda.coef_)
plt.show()