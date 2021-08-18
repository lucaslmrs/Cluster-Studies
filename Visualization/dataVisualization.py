# Initializing studies in data visualization for clustering
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


# loading and preprocessing tha database POKEMON
def load_database(path_data="Pokemon.csv"):
    df = pd.read_csv(path_data)
    types = df['Type 1'].isin(['Grass', 'Fire', 'Water', 'Electric'])
    labels = extract_labels(list(df[types]['Type 1']))
    drop_cols = ['Type 1', 'Type 2', 'Generation', 'Legendary', '#', 'Name']
    types_pokemon = df[types].drop(columns=drop_cols)
    return types_pokemon, labels


def extract_labels(data_classes):
    classes = list(set(data_classes))
    labels = np.array([])
    for c in data_classes:
        labels = np.append(labels, int(classes.index(c)))
    return labels


df_types_pokemon, labels_types_pokemon = load_database("Pokemon.csv")
df_types_pokemon = df_types_pokemon[["HP", "Attack", "Defense"]]
print(df_types_pokemon.head())


# inertia is used to know how many clusters we must use to represent the data
def inertia(data):
    scores = [KMeans(n_clusters=i+2).fit(data).inertia_ for i in range(10)]
    sns.lineplot(np.arange(2, 12), scores)
    plt.xlabel('Number of clusters')
    plt.ylabel("Inertia")
    plt.title("Inertia of k-Means versus number of clusters")
    plt.show()


# In this analysis, we will choice '4' groups to describe it, according to the result of inertia.
# This was take by the point that change of the curve is made. After this point we don't have
# too much changes in the inertia curve.
inertia(df_types_pokemon)

# At this point, let's make our k-means clustering in the database choosing 'n_clusters=4'.
normalized_pokemon = preprocessing.normalize(df_types_pokemon)
kmeans_result = KMeans(n_clusters=4)
kmeans_result.fit(normalized_pokemon)


# Now, we'll use the PCA to dimensionality reduction and represent the dataset clustered.
def prepare_pca(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = PCA(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i: names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels
    return df_matrix


def plot_2d(data, labels):
    pca_df = prepare_pca(3, data, labels)
    sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette="Set1")
    plt.show()


def plot_3d(data, labels, name='labels'):
    pca_df = prepare_pca(3, data, labels)
    fig = px.scatter_3d(pca_df, x='x', y='y', z='z', color=name, opacity=1, symbol=name)
    fig.update_traces(marker=dict(size=3))
    fig.show()


plot_2d(df_types_pokemon, kmeans_result.labels_)
plot_2d(df_types_pokemon, labels_types_pokemon)

plot_3d(df_types_pokemon, kmeans_result.labels_)
plot_3d(df_types_pokemon, labels_types_pokemon)
