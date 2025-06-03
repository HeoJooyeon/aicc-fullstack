import pandas as pd

df = pd.read_excel("quotes_result.xlsx")
target_author = "Eleanor Roosevelt"
filtered = df[df["author"] == target_author]
print(filtered)