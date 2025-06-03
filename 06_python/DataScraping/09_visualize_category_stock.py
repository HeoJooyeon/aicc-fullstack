import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("book_info.xlsx")
df['in_stock'] = df['in_stock'].astype(int)
avg_stock_by_category = df.groupby('category')['in_stock'].mean().sort_values()

plt.figure(figsize=(10, 6))
avg_stock_by_category.plot(kind='bar', color='skyblue')
plt.title("카테고리별 평균 재고 수량")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()