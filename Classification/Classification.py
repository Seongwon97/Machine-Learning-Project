import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('bank.csv')


print("\n--------------------------- [ Initial data ] ----------------------------")
print(df)
print("-------------------------------------------------------------------------\n")
# Dirty data count output
print("In Initial data, total dirty data count = ", sum(df.isna().sum()))


# ──────────────────────────────────────────
#          Drop Columns
# ──────────────────────────────────────────
print("\n--------------------------- [ Initial Columns ] ----------------------------")
count = 0
for c in df.columns:
    if (count != 0) & ((count % 7) == 0):
        print(c)
    elif count == len(df.columns):
        print(c)
    else:
        print(c, end=', ')
    count += 1

df2 = df.copy()

#Drop unnecessary columns
df2.drop(['Unnamed: 0', 'marital',  'marital', 'education', 'contact', 'month', 'day_of_week',
          'duration', 'campaign', 'pdays', 'previous', 'poutcome'], axis=1, inplace=True)

print("\n--------------------------- [ After Drop Columns ] ----------------------------")
count = 0
for c in df2.columns:
    if (count != 0) & ((count % 7) == 0):
        print(c)
    elif count == len(df2.columns):
        print(c, end='\n\n')
    else:
        print(c, end=', ')
    count += 1

# ──────────────────────────────────────────
#          LABEL ENCODING
# ──────────────────────────────────────────
df2 = df2.apply(LabelEncoder().fit_transform)


# ──────────────────────────────────────────
#      Show Correlation with Heatmap
# ──────────────────────────────────────────
corrmat = df2.corr()
top_corr = corrmat.index
plt.figure(figsize=(9, 9))
g = sns.heatmap(df2[top_corr].corr(), annot=True, cmap="RdYlGn")
#plt.show()


# ──────────────────────────────────────────
#               Normalization
# ──────────────────────────────────────────
y = df2['y']
X = df2.drop(['y'], axis=1)
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_df, columns=['age', 'job', 'default', 'housing', 'loan', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'])
scaled_df['y'] = y

# Visualize Data Normalization with MinMax Scaling
ig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 7))
ax1.set_title('Before Scaling', size=15)
sns.kdeplot(df2['age'], ax=ax1)
sns.kdeplot(df2['job'], ax=ax1)
sns.kdeplot(df2['default'], ax=ax1)
sns.kdeplot(df2['housing'], ax=ax1)
sns.kdeplot(df2['loan'], ax=ax1)
sns.kdeplot(df2['emp.var.rate'], ax=ax1)
sns.kdeplot(df2['cons.conf.idx'], ax=ax1)
sns.kdeplot(df2['cons.price.idx'], ax=ax1)
sns.kdeplot(df2['euribor3m'], ax=ax1)
sns.kdeplot(df2['nr.employed'], ax=ax1)

ax2.set_title('After MinMax Scaler', size=15)
sns.kdeplot(scaled_df['age'], ax=ax2)
sns.kdeplot(scaled_df['job'], ax=ax2)
sns.kdeplot(scaled_df['default'], ax=ax2)
sns.kdeplot(scaled_df['housing'], ax=ax2)
sns.kdeplot(scaled_df['loan'], ax=ax2)
sns.kdeplot(scaled_df['emp.var.rate'], ax=ax2)
sns.kdeplot(scaled_df['cons.conf.idx'], ax=ax2)
sns.kdeplot(scaled_df['cons.price.idx'], ax=ax2)
sns.kdeplot(scaled_df['euribor3m'], ax=ax2)
sns.kdeplot(scaled_df['nr.employed'], ax=ax2)

plt.show()

