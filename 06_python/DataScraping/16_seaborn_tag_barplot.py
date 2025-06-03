import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

df = pd.read_excel("quotes_result.xlsx")
df["tags"] = df["tags"].fillna("").apply(lambda x: [t.strip() for t in x.split(",") if t.strip()])
all_tags = sum(df["tags"], [])

tag_counts = pd.Series(all_tags).value_counts().head(30)

plt.figure(figsize=(14, 6))
sns.barplot(x=tag_counts.index, y=tag_counts.values, palette="viridis")
plt.title("전체 태그 사용 빈도 (상위 30개)")
plt.xlabel("태그")
plt.ylabel("사용 횟수")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()