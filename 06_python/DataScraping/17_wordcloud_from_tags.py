from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

df = pd.read_excel("quotes_result.xlsx")
df["tags"] = df["tags"].fillna("").apply(lambda x: [t.strip() for t in x.split(",") if t.strip()])
all_tags = sum(df["tags"], [])
tag_counts = Counter(all_tags)

wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='viridis')
wordcloud.generate_from_frequencies(tag_counts)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("전체 태그 워드클라우드", fontsize=18)
plt.tight_layout()
plt.show()