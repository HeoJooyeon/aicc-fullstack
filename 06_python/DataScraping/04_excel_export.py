import pandas as pd

df_titles = pd.DataFrame({'Title': ['Sample Book']})
df_titles.to_excel("book_titles_page1_to_5.xlsx", index=False)
print("엑셀 저장 완료!")