"""
TODO:   In this file, we'll try to plot the same dataset 'Pokemon.csv' but using another approach.
        This approach is based on the follow article of Medium-->Towards data science:
        <<https://towardsdatascience.com/visualizing-clusters-with-pythons-matplolib-35ae03d87489>>
        The focus of article is not to develop the best cluster for this algorithm, it's just to improve the
        visualization of the clusters and the data.

        ps: this file isn't a notebook because i don't have access to pycharm professional.
"""

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Pokemon.csv')

# do this to decrease the number of classes to 4.
types = df['Type 1'].isin(['Grass', 'Fire', 'Water', 'Electric'])
drop_cols = ['Type 1', 'Type 2', 'Generation', 'Legendary', '#']
df = df[types].drop(columns=drop_cols)
# print(df.head())


# In this case, we already know the number of classes that we are trying to cluster. If we didn't know that, would be
# interesting to use inertia and silhouette algorithms
kmeans = KMeans(n_clusters=4, random_state=0)
df['cluster'] = kmeans.fit_predict(df[['Attack', 'Defense']])
# print(df)

# get centroids
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids]
cen_y = [i[1] for i in centroids]
print(centroids)

# add to df
df['cen_x'] = df.cluster.map({0: cen_x[0], 1: cen_x[1], 2: cen_x[2], 3: cen_x[3]})
df['cen_y'] = df.cluster.map({0: cen_y[0], 1: cen_y[1], 2: cen_y[2], 3: cen_x[3]})

# define and map colors
colors = ['#4F0AF7', '#97F70A', '#D32500', "#000000"]
df['c'] = df.cluster.map({0: colors[0], 1: colors[1], 2: colors[2], 3: colors[3]})

plt.scatter(df.Attack, df.Defense, c=df.c, alpha=0.6, s=10)
plt.show()
