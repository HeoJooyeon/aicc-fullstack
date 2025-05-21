import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from pathlib import Path
import numpy as np

path = Path("weather.csv")
lines = path.read_text().splitlines()

reader = csv.reader(lines)
header_row = next(reader)

citys, tavg_list, tmax_list, tmin_list= [], [], [], []

for row in reader:
    city = row[5]
    tavg = row[9]
    tmax = row[10]
    tmin = row[11]
    citys.append(city)
    tavg_list.append(float(tavg))
    tmax_list.append(float(tmax))
    tmin_list.append(float(tmin))

data = list(zip(citys, tavg_list, tmax_list, tmin_list))
data_sorted = sorted(data, key=lambda x: float(x[1]), reverse=True)
top_10 = data_sorted[:10]
citys, tavg_list, tmax_list, tmin_list = zip(*top_10)

x = np.arange(len(citys))  
width = 0.25       

fig, ax = plt.subplots(figsize=(18, 6))

ax.bar(x - width, tavg_list, width, label='평균기온', color='skyblue')
ax.bar(x, tmax_list, width, label='최고기온', color='salmon')
ax.bar(x + width, tmin_list, width, label='최저기온', color='lightgreen')

ax.set_xticks(x)
ax.set_xticklabels(citys, fontsize=12)

ax.set_title("상위 10개 도시의 평균/최고/최저 기온", fontsize=14)
ax.set_ylabel("기온(F)", fontsize=12)
fig.autofmt_xdate()
ax.legend()
plt.tight_layout()
plt.show()