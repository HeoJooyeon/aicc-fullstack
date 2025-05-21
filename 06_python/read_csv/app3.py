import pandas as pd
#pip install pandas
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("weather.csv", encoding='utf-8')
df = df[["Station.City","Data.Temperature.Avg Temp","Data.Temperature.Max Temp","Data.Temperature.Min Temp"]]

# df["Station.City"]
df = df.dropna()

df.columns = ["City", "AvgTemp", "MaxTemp", "MinTemp"]

top10 = df.sort_values(by="AvgTemp", ascending = False).head(10)

plt.figure(figsize=(12,6))
bar_width = 0.25
x = range(len(top10))
x_max = [i + bar_width for i in x]
x_min = [i + bar_width*2 for i in x]

plt.bar(x, top10["AvgTemp"], width=bar_width, label='평균 기온', color='orange')
plt.bar(x_max, top10["MaxTemp"], width=bar_width, label='최고 기온', color='red')
plt.bar(x_min, top10["MinTemp"], width=bar_width, label='최저 기온', color='blue')

x_tick = [i + bar_width for i in x]
plt.xticks(x_tick, top10["City"], rotation=45, fontsize=12)
plt.title("상위 10개 도시의 평균/최고/최저 기온", fontsize=14)
plt.ylabel("기온(F)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()