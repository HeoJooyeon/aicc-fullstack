import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://books.toscrape.com/catalogue/page-1.html'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

books = soup.find_all('article', class_='product_pod')
rating_map = {'One':1, 'Two':2, 'Three':3, 'Four':4, 'Five':5}
book_data = []

for book in books:
    title = book.h3.a['title']
    star_rating = book.p['class'][1]
    star_num = rating_map.get(star_rating, 0)
    price = book.find('p', class_='price_color').text.replace('Â','').strip()
    book_data.append({"Title": title, "Star Rating": star_num, "Price": price})

df = pd.DataFrame(book_data)
print(df)