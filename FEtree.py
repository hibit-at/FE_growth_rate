import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from pandas import plotting
import mglearn

df = pd.read_csv('growth_rate.csv')

col_sex = []
for s in list(df['sex']):
    if s == 'Male':
        col_sex.append('b')
    else:
        col_sex.append('r')

dfs = df.iloc[:,3:].apply(lambda x: (x-x.mean())/x.std(), axis = 0)

pca = PCA()
feature = pca.fit(dfs)
feature = pca.transform(dfs)

fig, ax = plt.subplots(figsize=(10,10), dpi=100)
ax.scatter(feature[:,0],feature[:,1],alpha=0)
for i in range(0,len(df['name'])):
    ax.annotate(df['name'][i],(feature[:,0][i],feature[:,1][i]),c = col_sex[i])
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

X = df.iloc[:,3:11]
y = df['sex']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
forest = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=3)
forest.fit(X_train, y_train)

print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))

#from sklearn.tree import export_graphviz
#feature_label = ['HP','POW','MAG','TEX','SPD','LUC','DEF','RES']
#export_graphviz(forest, out_file='tree3.dot', class_names=["Female","Male"], feature_names=feature_label, filled = True)
