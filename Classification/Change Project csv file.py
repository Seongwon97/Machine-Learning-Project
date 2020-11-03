# Importing Data Analysis Librarys
import pandas as pd
import random
import numpy as np

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('bank-additional-full.csv', sep=';')
df.head()
df.to_csv('bank.csv')

for i in range(421):
    ran=random.randrange(len(df))
    df.loc[ran,'age']=np.nan

for i in range(562):
    ran=random.randrange(len(df))
    df.loc[ran,'job']=np.nan

for i in range(354):
    ran=random.randrange(len(df))
    df.loc[ran,'marital']=np.nan

for i in range(841):
    ran=random.randrange(len(df))
    df.loc[ran,'education']=np.nan

for i in range(712):
    ran=random.randrange(len(df))
    df.loc[ran,'default']=np.nan

for i in range(348):
    ran=random.randrange(len(df))
    df.loc[ran,'housing']=np.nan

for i in range(642):
    ran=random.randrange(len(df))
    df.loc[ran,'loan']=np.nan

for i in range(426):
    ran=random.randrange(len(df))
    df.loc[ran,'contact']=np.nan

for i in range(751):
    ran=random.randrange(len(df))
    df.loc[ran,'month']=np.nan

for i in range(841):
    ran=random.randrange(len(df))
    df.loc[ran,'day_of_week']=np.nan

for i in range(321):
    ran=random.randrange(len(df))
    df.loc[ran,'duration']=np.nan

for i in range(272):
    ran=random.randrange(len(df))
    df.loc[ran,'campaign']=np.nan

for i in range(298):
    ran=random.randrange(len(df))
    df.loc[ran,'pdays']=np.nan

for i in range(124):
    ran=random.randrange(len(df))
    df.loc[ran,'previous']=np.nan

for i in range(348):
    ran=random.randrange(len(df))
    df.loc[ran,'poutcome']=np.nan

for i in range(472):
    ran=random.randrange(len(df))
    df.loc[ran,'emp.var.rate']=np.nan

for i in range(428):
    ran=random.randrange(len(df))
    df.loc[ran,'cons.price.idx']=np.nan

for i in range(231):
    ran=random.randrange(len(df))
    df.loc[ran,'cons.conf.idx']=np.nan

for i in range(294):
    ran=random.randrange(len(df))
    df.loc[ran,'euribor3m']=np.nan

for i in range(327):
    ran=random.randrange(len(df))
    df.loc[ran,'nr.employed']=np.nan

print("Total dirty data count",sum(df.isna().sum()))
df.to_csv('bank_d.csv')
