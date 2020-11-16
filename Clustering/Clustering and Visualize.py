import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import folium

warnings.filterwarnings('ignore')
country_geo = 'world-countries.json'

df_data= pd.read_csv('Merge.csv')
df_mergeData = df_data[['IncomeGroup', 'Value']]
print(df_data)

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

print(df_mergeData)

# ──────────────────────────────────────────
#          SCALING
# ──────────────────────────────────────────
# scatter출력에서 integer값들만 사용할 수 있어서 일단 scaler를 주석처리 했습니다.
s_scaler = StandardScaler()
m_scaler = MinMaxScaler()
#df_mergeData = SCALING(df_mergeData, m_scaler)


# ──────────────────────────────────────────
#          MAKE SCATTER PLOT
# ──────────────────────────────────────────

def scatter_plot(result, data, model_type):
    plt.title(model_type)
    plt.scatter(data['IncomeGroup'], data['Value'], c=result, alpha=0.5)
    plt.show()

# ──────────────────────────────────────────
#          MAKE Visualize Map
# ──────────────────────────────────────────

def make_Map(data, method):
    plot_data = data[['CountryCode', method]]

    map = folium.Map([20, 10], zoom_start=3)

    map.choropleth(geo_data=country_geo, data=plot_data,
                   columns=['CountryCode', method],
                   key_on='feature.id',
                   fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2, legend_name='Cluster')
    map.save('Map.html')


# ──────────────────────────────────────────
#          K-MEANS CLUSTERING
# ──────────────────────────────────────────

from sklearn.cluster import KMeans

# #K-MEANS PARAMETER
n_clusters = [2, 3, 4, 5, 6]
max_iter = [50, 100, 200, 300]


def KMEANS_CLUSTERING(dataset1, dataset2):
    for i in n_clusters:
        for j in max_iter:
            kmeans = KMeans(n_clusters=i, max_iter=j)
            pd_kmeans = kmeans.fit_predict(dataset1)
            dataset2['KMeans']=pd_kmeans
            # ──────────────────────────────────────────
            #  VISUALIZE BEST RESULT AS SCATTER PLOT
            # ──────────────────────────────────────────
            scatter_plot(pd_kmeans, dataset1, 'K-Means')
            make_Map(dataset2, 'KMeans')


# ──────────────────────────────────────────
#          DBSCAN CLUSTERING
# ──────────────────────────────────────────

# DBSCAN PARAMETER
eps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
min_samples = [3, 5, 10, 15, 20, 30, 50, 100]


# Compute DBSCAN
def DBSCAN_CLUSTERING(dataset1, dataset2):
    for i in eps:
        for j in min_samples:
            dbscan = DBSCAN(eps=i, min_samples=j)
            pd_dbscan = dbscan.fit_predict(dataset1)
            dataset2['DBSCAN'] = pd_dbscan
            # ──────────────────────────────────────────
            #  VISUALIZE BEST RESULT AS SCATTER PLOT
            # ──────────────────────────────────────────
            scatter_plot(pd_dbscan, dataset1, 'DBSCAN')
            make_Map(dataset2, 'DBSCAN')


# ──────────────────────────────────────────
#          EM CLUSTERING
# ──────────────────────────────────────────

from sklearn.mixture import GaussianMixture

# EM PARAMETER
n_components = [2, 3, 4, 5, 6]
max_iter = [50, 100, 200, 300]


def EM_CLUSTERING(dataset1, dataset2):
    for i in n_components:
        for j in max_iter:
            em = GaussianMixture(n_components=i, max_iter=j)
            pd_em = em.fit_predict(dataset1)
            dataset2['EM']=pd_em
            # ──────────────────────────────────────────
            #  VISUALIZE BEST RESULT AS SCATTER PLOT
            # ──────────────────────────────────────────
            scatter_plot(pd_em, dataset1, 'EM')
            make_Map(dataset2, 'EM')


# ──────────────────────────────────────────
#          RESULT
# ──────────────────────────────────────────

KMEANS_CLUSTERING(df_mergeData, df_data)
DBSCAN_CLUSTERING(df_mergeData, df_data)
EM_CLUSTERING(df_mergeData, df_data)
