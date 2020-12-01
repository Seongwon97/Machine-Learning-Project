import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import folium
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
country_geo = 'data_file/world-countries.json'

df_data= pd.read_csv('data_file/Merge.csv')
df_mergeData = df_data[['IncomeGroup', 'Value']]
print(df_data)

# ──────────────────────────────────────────
# PREPROCESSING FUNCTIONS
# ──────────────────────────────────────────
# SCAILING
def SCALING(data, scaler):
    df_scaled = scaler.fit_transform(data)
    return df_scaled

# ENCODING
def ENCODING(df, column):
  encoder = LabelEncoder()
  encoder.fit(df[column])
  df[column] = encoder.transform(df[column])
  return df


def scatter_plot(result, data, model_type):
    plt.title("{0},{1},{2}".format(model_type, i, j))
    plt.scatter(data['IncomeGroup'], data['Value'], c=result, alpha=1.0)
    plt.xlabel("IncomeGroup")
    plt.ylabel("population growth")
    plt.xticks([0, 1, 2, 3, 4])
    # plt.show()
    plt.savefig('Clustering Result-{0},{1},{2}.png'.format(model_type, i, j))
    
# ──────────────────────────────────────────
#          MAKE Visualize Map
# ──────────────────────────────────────────
#일단은 parameter가 바뀔때마다 지도를 업데이트를 시켰는데
#추후에 회의를 통해 best case를 정하고 best case만 따로 저장시키는게 좋을 것 같아요.
def make_Map(data, method, i, j):
    plot_data = data[['CountryCode', method]]

    map = folium.Map([20, 10], zoom_start=3)

    map.choropleth(geo_data=country_geo, data=plot_data,
                   columns=['CountryCode', method],
                   key_on='feature.id',
                   fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2, legend_name='Clustering Result')
    map.save('Clustering Result Map-{0},{1},{2}.html'.format(method, i, j))

    plot_data = data[['CountryCode', 'Value']]

    map = folium.Map([20, 10], zoom_start=3)

    map.choropleth(geo_data=country_geo, data=plot_data,
                   columns=['CountryCode', 'Value'],
                   key_on='feature.id',
                   fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2, legend_name='Population growth (annual %)')
    map.save('Map according to Value value.html')
    
# #K-MEANS PARAMETER


def KMEANS_CLUSTERING(dataset1, dataset2):
  n_clusters = [2, 3, 4, 5, 6]
  max_iter = [50, 100, 200, 300]
  for i in n_clusters:
      for j in max_iter:
          print("n_cluster = {}, max_iter = {}".format(i,j))
          kmeans = KMeans(n_clusters=i, max_iter=j)
          pd_kmeans = kmeans.fit_predict(dataset1)
          dataset2['KMeans']=pd_kmeans
          # ──────────────────────────────────────────
          #  VISUALIZE BEST RESULT AS SCATTER PLOT
          # ──────────────────────────────────────────
          scatter_plot(pd_kmeans, dataset1, 'K-Means', i, j)
          make_Map(dataset2, 'KMeans', i, j)




# Compute DBSCAN
def DBSCAN_CLUSTERING(dataset1, dataset2):

  # DBSCAN PARAMETER
  eps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
  min_samples = [3, 5, 10, 15, 20, 30, 50, 100]
  for i in eps:
      for j in min_samples:
          print("eps = {}, min_samples = {}".format(i,j))
          dbscan = DBSCAN(eps=i, min_samples=j)
          pd_dbscan = dbscan.fit_predict(dataset1)
          dataset2['DBSCAN'] = pd_dbscan
          # ──────────────────────────────────────────
          #  VISUALIZE BEST RESULT AS SCATTER PLOT
          # ──────────────────────────────────────────
          scatter_plot(pd_dbscan, dataset1, 'DBSCAN', i, j)
          make_Map(dataset2, 'DBSCAN', i, j)



def EM_CLUSTERING(dataset1, dataset2):
  # EM PARAMETER
  n_components = [2, 3, 4, 5, 6]
  max_iter = [50, 100, 200, 300]
  for i in n_components:
      for j in max_iter:
          print("n_components = {}, max_iter = {}".format(i,j))
          em = GaussianMixture(n_components=i, max_iter=j)
          pd_em = em.fit_predict(dataset1)
          dataset2['EM']=pd_em
          # ──────────────────────────────────────────
          #  VISUALIZE BEST RESULT AS SCATTER PLOT
          # ──────────────────────────────────────────
          scatter_plot(pd_em, dataset1, 'EM', i, j)
          make_Map(dataset2, 'EM', i, j)


if __name__ == "__main__":


    # ──────────────────────────────────────────
    #          LABEL ENCODING
    # ──────────────────────────────────────────

    df_mergeData = ENCODING(df_mergeData, 'IncomeGroup')  # Label encoding
    df_mergeData.head()

    # ──────────────────────────────────────────
    #          RESULT
    # ──────────────────────────────────────────
    KMEANS_CLUSTERING(df_mergeData, df_data)
    DBSCAN_CLUSTERING(df_mergeData, df_data)
    EM_CLUSTERING(df_mergeData, df_data)
