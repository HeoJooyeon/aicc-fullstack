import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_excel("quotes_result.xlsx")

all_tags = []
for tag_str in df["tags"]:
    split_tags = [tag.strip() for tag in tag_str.split(",") if tag.strip()]
    all_tags.extend(split_tags)

tag_counts = Counter(all_tags)
top_tags = tag_counts.most_common(10)
tags, counts = zip(*top_tags)

plt.figure(figsize=(10, 5))
plt.bar(tags, counts, color='skyblue')
plt.title("Top 10 Most Common Tags")
plt.xlabel("Tags")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()