import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv("iris.csv")

# # 直方图
# df.hist()
# plt.show()
#
# # 盒图
# df.boxplot()
# plt.show()
#
#
# # 五数概括
# def Five(d):
#     Minimum = min(d)
#     Maximum = max(d)
#     Q1 = np.percentile(d, 25)
#     Median = np.median(d)
#     Q3 = np.percentile(d, 75)
#     return Minimum, Q1, Median, Q3, Maximum
#
#
# print("sepal.length", FiveStat(df['sepal.length']))
# print("sepal.width", FiveStat(df['sepal.width']))
# print("petal.length", FiveStat(df['petal.length']))
# print("petal.width", FiveStat(df['petal.width']))

# setosa = df[df.variety == 'Setosa']
# versicolor = df[df.variety == 'Versicolor']
# virginica = df[df.variety == 'Virginica']


def B(a, b):
    df.plot.scatter(x=a, y=b)
    plt.title(a + '-' + b)
    plt.show()
    print("Pearson", np.corrcoef(df[a], df[b]))
    print("Cov", np.cov(df[a], df[b]))


def C(a, b):
    sns.regplot(x=df[a], y=df[b], ci=None, color='b', line_kws={'color': 'r'})
    plt.title(a + '-' + b)
    plt.show()


def D(a, b):
    d = df.drop('variety', axis=1)
    d = d.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    sns.regplot(x=d[a], y=d[b], ci=None, color='b', line_kws={'color': 'r'})
    plt.title(a + '-' + b + 'minmax')
    plt.show()
    d = d.apply(lambda x: (x - x.mean())/math.sqrt(sum((x - x.min())**2 / len(x))))
    sns.regplot(x=d[a], y=d[b], ci=None, color='b', line_kws={'color': 'r'})
    plt.title(a + '-' + b + 'zscore')
    plt.show()


pca = PCA(n_components=2)
d = pca.fit_transform(df.drop('variety', axis=1))


