import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

#オリジナルデータを読み込む
df = pd.read_csv('growth_rate.csv')

#後で性別のカラーラベルに使用するためにbとrに置き換えたものを作成
col_sex = df['sex'].replace('Male','b').replace('Female','r')

#ここで主成分分析
pca = PCA() 
pca.fit(df.iloc[:,3:])
feature = pca.transform(df.iloc[:,3:])

#ここで線形判別分析
X = df.iloc[:,3:]
y = df['sex']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state = 0)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

plt.figure(figsize=(10,10))

#第二主成分をグラフ化
print(pca.components_[1])
plt.subplot(2,1,1)
plt.bar(range(1,9),pca.components_[1],tick_label = df.columns[3:])
plt.title('主成分分析における第2主成分')

#判別関数をグラフ化
print(lda.coef_[0])
plt.subplot(2,1,2)
plt.bar(range(1,9),-lda.coef_[0],tick_label = df.columns[3:])
plt.title('線形判別分析における判別関数')

plt.show()