# Initalizing studies in data visualization for clustering
# Data handling
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Data visualization
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px

# Clustering
from sklearn.cluster import KMeans

# Dimensionality reduction
from sklearn.decomposition import PCA


df = pd.read_csv("Pokemon.csv")
types = df['Type 1'].isin(['Grass', 'Fire', 'Water'])
drop_cols = ['Type 1', 'Type 2', 'Generation', 'Legendary', '#', 'Name']
df_types_pokemon = df[types].drop(columns=drop_cols)

print(df_types_pokemon.columns)
print(df_types_pokemon)


scores = [KMeans(n_clusters=i+2).fit(df_types_pokemon).inertia_ for i in range(10)]
sns.lineplot(np.arange(2, 12), scores)
plt.xlabel('Number of clusters')
plt.ylabel("Inertia")
plt.title("Inertia of k-Means versus number of clusters")
plt.show()


normalized_pokemon = preprocessing.normalize(df_types_pokemon)
kmeans = KMeans(n_clusters=3)
kmeans.fit(normalized_pokemon)


def prepare_pca(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = PCA(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i: names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels

    return df_matrix


pca_df = prepare_pca(3, df_types_pokemon, kmeans.labels_)
sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette="Set2")
plt.show()


def plot_3d(data, name='labels'):
    fig = px.scatter_3d(data, x='x', y='y', z='z',
                        color=name, opacity=0.5)

    fig.update_traces(marker=dict(size=3))
    fig.show()


pca_df = prepare_pca(3, df_types_pokemon, kmeans.labels_)
plot_3d(pca_df)
plt.show()
