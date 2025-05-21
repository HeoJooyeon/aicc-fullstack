from pathlib import Path
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from matplotlib import rcParams

rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


path = Path("weather.csv")
# lines, list
lines = path.read_text().splitlines()

reader = csv.reader(lines)
#header 첫 줄
header_row = next(reader)

temps, dates = [], []

for row in reader:
    temp = row[10]
    #str parsing time
    current_date = datetime.strptime(row[1], "%Y-%m-%d")
    temps.append(temp)
    dates.append(current_date)

# axes
fig, ax = plt.subplots()

ax.plot(dates, temps, color="blue")

ax.set_title("Temp")
ax.set_xlabel("", fontsize = 16)
fig.autofmt_xdate()
ax.set_ylabel("Temperature", fontsize = 16)

plt.show()