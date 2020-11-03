import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import warnings

warnings.filterwarnings('ignore')

# df_wdl = pd.read_csv('Indicators.csv', index_col=[0])
# print(df_wdl.head())
#
# df_population = df_wdl.loc[df_wdl['IndicatorName'].str.contains('Population' or 'population', na=False)]
# df_population.to_csv("populations.csv")
#
# df_population = pd.read_csv('populations.csv', index_col=[0])
# print(df_population.head())
#
# df_population_sorted_by_county = df_population.sort_values(by=['CountryName', 'Year'], ascending=True)
# df_population_sorted_by_county.to_csv("populations_sorted.csv")
# # 여기까진 인구에 관련 된 indicators만..

df_population_sorted_by_county = pd.read_csv('populations_sorted.csv', index_col=[0])
# print(df_population_sorted_by_county.head())

# 여기서부턴 우리가 사용할 indicator..
hist_indicator = 'Population, total'
hist_year = 2011
# df_population_total = df_population_sorted_by_county.loc[
#     df_population_sorted_by_county['IndicatorName'].str.contains('Population, total', na=False)]

mask1 = df_population_sorted_by_county['IndicatorName'].str.contains(hist_indicator)
mask2 = df_population_sorted_by_county['Year'].isin([hist_year])

df_population_total = df_population_sorted_by_county[mask1 & mask2]
# print(df_population_total.head())

# 2020.11.03 지도위 그리기 테스트...
plot_data = df_population_total[['CountryCode', 'Value']]
print(plot_data.head())
print(plot_data.describe())
hist_indicator = df_population_total.iloc[0]['IndicatorName']
print(hist_indicator)

country_geo = './world-countries.json'
map = folium.Map(location=[100, 0], zoom_start=1.5)
map.choropleth(geo_data=country_geo, data=plot_data,
               columns=['CountryCode', 'Value'],
               key_on='feature.id',
               fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
               legend_name=hist_indicator)

map.save('plot_data.html')
