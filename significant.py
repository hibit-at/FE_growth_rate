import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#オリジナルデータを読み込む
df = pd.read_csv('growth_rate.csv')

males = df[df.sex=='Male'].sum(axis=1) #男性ユニットの合計成長率
females = df[df.sex=='Female'].sum(axis=1) #女性ユニットの合計成長率

print(males.mean())
print(females.mean())

#等分散も正規分布も仮定できないのでU検定
print(stats.mannwhitneyu(males, females, alternative='two-sided'))