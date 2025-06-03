import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("book_test.xlsx")
rating_counts = df['stars'].value_counts().sort_index()
rating_counts.plot(kind='bar', color='skyblue')
plt.title('Star Rating Distribution')
plt.xlabel('Star Rating')
plt.ylabel('Number of Books')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()