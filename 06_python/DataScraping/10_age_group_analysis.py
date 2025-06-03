import pandas as pd
import matplotlib.pyplot as plt

books_df = pd.read_excel('book_info.xlsx')
buyers_df = pd.read_excel('buyers.xlsx')

buyers_df['age_group'] = (buyers_df['buyer_age'] // 10) * 10
buyers_df['age_group'] = buyers_df['age_group'].astype(str) + "대"
merged_df = pd.merge(buyers_df, books_df[['title', 'category']], left_on='book_purchased', right_on='title', how='left')

count_cate_by_age_group = (
    merged_df.groupby(['age_group', 'category'])
    .size()
    .reset_index(name='count')
    .sort_values(['age_group', 'count'], ascending=[True, False])
)

top3_by_age_group = count_cate_by_age_group.groupby('age_group').head(3)
pivot_df = top3_by_age_group.pivot(index='age_group', columns='category', values='count').fillna(0)

pivot_df.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='Set3')
plt.title("세대별 인기 Top 3 카테고리")
plt.xlabel("연령대")
plt.ylabel("구매 수")
plt.legend(title="카테고리", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()