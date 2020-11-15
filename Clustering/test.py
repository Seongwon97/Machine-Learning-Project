import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import warnings
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# df_wdl = pd.read_csv('Indicators.csv', index_col=[0])
# print(df_wdl.head())
#
# df_population = df_wdl.loc[df_wdl['IndicatorName'].str.contains('Population' or 'population', na=False)]
# df_population.to_csv("populations_indicators.csv")
#
# df_population = pd.read_csv('populations_indicators.csv', index_col=[0])
# print(df_population.head())
#
# df_population_sorted_by_county = df_population.sort_values(by=['CountryName', 'Year'], ascending=True)
# df_population_sorted_by_county.to_csv("populations_indicators_sorted.csv")
# # 여기까진 인구에 관련 된 indicators만..

df_population_sorted_by_county = pd.read_csv('populations_indicators_sorted.csv')
# print(df_population_sorted_by_county.head())

# 여기서부턴 우리가 사용할 indicator..
# 'SP.POP.GROW'  == Population growth (annual %)
hist_indicator = 'SP.POP.GROW'
hist_year = 2014
# df_population_total = df_population_sorted_by_county.loc[
#     df_population_sorted_by_county['IndicatorName'].str.contains('Population, total', na=False)]

mask1 = df_population_sorted_by_county['IndicatorCode'].str.contains(hist_indicator)
mask2 = df_population_sorted_by_county['Year'].isin([hist_year])

# 인구 증가율과 연도(2014)에 관한 부분만 추출
df_population_growth = df_population_sorted_by_county[mask1 & mask2]
print(df_population_growth)

df_country = pd.read_csv('Country.csv')
df_sample = df_country[['CountryCode', 'IncomeGroup']]
# print(df_sample.head(10))

# 인구증가율 데이터셋과 country.csv의 Income정보를 country code를 기준으로 병합
df_mergeData = pd.merge(df_population_growth, df_sample, on='CountryCode', how='inner')
# print(df_mergeData)
# IncomeGroup의 결측값이 있는 나라는 국가가 아니라 연합임으로 행 삭제
df_mergeData.dropna(axis=0, inplace=True)
df_mergeData.reset_index(drop=True, inplace=True)
df_mergeData.to_csv("Merge.csv")

# # 2020.11.03 지도위 그리기 테스트...
# plot_data = df_population_total[['CountryCode', 'Value']]
# print(plot_data.head())
# print(plot_data.describe())
# hist_indicator = df_population_total.iloc[0]['IndicatorName']
# print(hist_indicator)

# country_geo = './world-countries.json'
# map = folium.Map(location=[100, 0], zoom_start=1.5)
# map.choropleth(geo_data=country_geo, data=plot_data,
#                columns=['CountryCode', 'Value'],
#                key_on='feature.id',
#                fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
#                legend_name=hist_indicator)
#
# map.save('plot_data.html')


# 2020-11-16 ADDED BY JINKYUNG
#──────────────────────────────────────────
# PREPROCESSING FUNCTIONS
#──────────────────────────────────────────
#SCAILING
def SCALING(data, scaler):
    df_scaled = scaler.fit_transform(data)
    return df_scaled

#ENCODING
def ENCODING(col,df):
    for c in col:
        encoder = LabelEncoder()
        encoder.fit(df[c])
        df[c] = encoder.transform(df[c])
    return df

#──────────────────────────────────────────
#          LABEL ENCODING
#──────────────────────────────────────────

X = df_mergeData.iloc[:,:-1] # Features
y = df_mergeData.iloc[:,-1] # Target variable


columnsList = df_mergeData.columns.tolist()
df_mergeData = ENCODING(columnsList, df_mergeData)

#──────────────────────────────────────────
#          SCALING
#──────────────────────────────────────────

s_scaler = StandardScaler()
df_mergeData = SCALING(df_mergeData, s_scaler)

#──────────────────────────────────────────
#          MAKE SCATTER PLOT
#──────────────────────────────────────────

def scatter_plot (result , data , model_type):
    plt.title(model_type)
    plt.scatter(data[1],data[0],  s = 100 , cmap= 'rainbow')
    plt.show()
    

#──────────────────────────────────────────
#          K-MEANS CLUSTERING
#──────────────────────────────────────────

from sklearn.cluster import KMeans

# #K-MEANS PARAMETER
n_clusters = [2, 3, 4, 5, 6]
max_iter = [50, 100, 200, 300]

def KMEANS_CLUSTERING(dataset):
    for i in n_clusters:
        for j in max_iter:
            kmeans = KMeans(n_clusters = i, max_iter = j)
            pd_kmeans = kmeans.fit_predict(dataset)
            #──────────────────────────────────────────
            #  VISUALIZE BEST RESULT AS SCATTER PLOT
            #──────────────────────────────────────────
            scatter_plot(pd_kmeans , dataset , 'K-Means')
            
#──────────────────────────────────────────
#          DBSCAN CLUSTERING
#──────────────────────────────────────────

#DBSCAN PARAMETER
eps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
min_samples = [3, 5, 10, 15, 20, 30, 50, 100]

# Compute DBSCAN
def DBSCAN_CLUSTERING(dataset):
    for i in eps:
        for j in min_samples:
            dbscan = DBSCAN(eps = i, min_samples = j)
            pd_dbscan = dbscan.fit_predict(dataset)
            #──────────────────────────────────────────
            #  VISUALIZE BEST RESULT AS SCATTER PLOT
            #──────────────────────────────────────────
            scatter_plot(pd_dbscan, dataset,  'DBSCAN' )
            

#──────────────────────────────────────────
#          EM CLUSTERING
#──────────────────────────────────────────

from sklearn.mixture import GaussianMixture

#EM PARAMETER
n_components=[2, 3, 4, 5, 6] 
max_iter=[50, 100, 200, 300]

def EM_CLUSTERING(dataset):
    for i in n_components:
        for j in max_iter:
            em = GaussianMixture(n_components=i, max_iter=j)
            pd_em = em.fit_predict(dataset)
            #──────────────────────────────────────────
            #  VISUALIZE BEST RESULT AS SCATTER PLOT
            #──────────────────────────────────────────
            scatter_plot(pd_em , dataset , 'EM')
            
#──────────────────────────────────────────
#          RESULT
#──────────────────────────────────────────

KMEANS_CLUSTERING(df_mergeData)
DBSCAN_CLUSTERING(df_mergeData)
EM_CLUSTERING(df_mergeData)
