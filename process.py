import numpy as np
import pandas as pd
import opti

df = pd.read_csv("data/msci.csv")
df.index = pd.to_datetime(df["Date"], format='%m/%d/%Y')
df = df.drop(['Date'], axis=1)

df = df[["France","Japan","Spain"]]

for col in df.columns.values:
    df[col] = np.log(df[col]) - np.log(df[col].shift(1))
data = df.dropna()

sample = data

#plot efficient frontier
opti.plot_eff_frontier(sample)


weights = opti.opti_mkv(sample, target="max_sharpe")
print(weights)


vol = np.sqrt(np.dot(weights.T,np.dot(sample.cov().values,weights)) )*np.sqrt(252)
ret = np.dot(weights.T, [sample[col].mean() for col in sample.columns.values] )*252

print(vol, ret)