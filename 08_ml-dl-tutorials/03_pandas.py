import pandas as pd

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('/content/sbus.csv')
print(df.head())

type(df)

df.head()

df.tail()


df.info()

df.columns

df[['open', 'close']]

df.iloc[0]

df2 = pd.read_csv('sbus.csv', index_col='date')
df2.head()

df2.loc['2013-02-08']

import numpy as np

A = np.arange(10)

# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


A[A % 2 == 0]


df.values

A = df[['open', 'close']].values

type(A)


smalldf = df[['open', 'close']]
smalldf.to_csv('output.csv')

!head output.csv

# date

def date_to_year(row):
  return int(row['date'].split('-')[0])
  # 그 날짜에서 - 빼달라

  df.apply(date_to_year, axis=1)
  # 첫 줄만 적용해달라


