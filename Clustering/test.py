import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

df_mergeData = pd.read_csv('Merge.csv')


# 2020-11-16 ADDED BY JINKYUNG
# ──────────────────────────────────────────
# PREPROCESSING FUNCTIONS
# ──────────────────────────────────────────
# SCAILING
def SCALING(data, scaler):
    df_scaled = scaler.fit_transform(data)
    return df_scaled


# ENCODING
def ENCODING(col, df):
    for c in col:
        encoder = LabelEncoder()
        encoder.fit(df[c])
        df[c] = encoder.transform(df[c])
    return df


# ──────────────────────────────────────────
#          LABEL ENCODING
# ──────────────────────────────────────────

X = df_mergeData.iloc[:, :-1]  # Features
y = df_mergeData.iloc[:, -1]  # Target variable

columnsList = df_mergeData.columns.tolist()
df_mergeData = ENCODING(columnsList, df_mergeData)

# ──────────────────────────────────────────
#          SCALING
# ──────────────────────────────────────────

s_scaler = StandardScaler()
df_mergeData = SCALING(df_mergeData, s_scaler)


# ──────────────────────────────────────────
#          MAKE SCATTER PLOT
# ──────────────────────────────────────────

def scatter_plot(result, data, model_type):
    plt.title(model_type)
    plt.scatter(data[1], data[0], s=100, cmap='rainbow')
    plt.show()


# ──────────────────────────────────────────
#          K-MEANS CLUSTERING
# ──────────────────────────────────────────

from sklearn.cluster import KMeans

# #K-MEANS PARAMETER
n_clusters = [2, 3, 4, 5, 6]
max_iter = [50, 100, 200, 300]


def KMEANS_CLUSTERING(dataset):
    for i in n_clusters:
        for j in max_iter:
            kmeans = KMeans(n_clusters=i, max_iter=j)
            pd_kmeans = kmeans.fit_predict(dataset)
            # ──────────────────────────────────────────
            #  VISUALIZE BEST RESULT AS SCATTER PLOT
            # ──────────────────────────────────────────
            scatter_plot(pd_kmeans, dataset, 'K-Means')


# ──────────────────────────────────────────
#          DBSCAN CLUSTERING
# ──────────────────────────────────────────

# DBSCAN PARAMETER
eps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
min_samples = [3, 5, 10, 15, 20, 30, 50, 100]


# Compute DBSCAN
def DBSCAN_CLUSTERING(dataset):
    for i in eps:
        for j in min_samples:
            dbscan = DBSCAN(eps=i, min_samples=j)
            pd_dbscan = dbscan.fit_predict(dataset)
            # ──────────────────────────────────────────
            #  VISUALIZE BEST RESULT AS SCATTER PLOT
            # ──────────────────────────────────────────
            scatter_plot(pd_dbscan, dataset, 'DBSCAN')


# ──────────────────────────────────────────
#          EM CLUSTERING
# ──────────────────────────────────────────

from sklearn.mixture import GaussianMixture

# EM PARAMETER
n_components = [2, 3, 4, 5, 6]
max_iter = [50, 100, 200, 300]


def EM_CLUSTERING(dataset):
    for i in n_components:
        for j in max_iter:
            em = GaussianMixture(n_components=i, max_iter=j)
            pd_em = em.fit_predict(dataset)
            # ──────────────────────────────────────────
            #  VISUALIZE BEST RESULT AS SCATTER PLOT
            # ──────────────────────────────────────────
            scatter_plot(pd_em, dataset, 'EM')


# ──────────────────────────────────────────
#          RESULT
# ──────────────────────────────────────────

KMEANS_CLUSTERING(df_mergeData)
DBSCAN_CLUSTERING(df_mergeData)
EM_CLUSTERING(df_mergeData)
