import requests
from bs4 import BeautifulSoup
import pandas as pd

bread_books = []
for page_num in range(1, 3):
    print(f"[페이지: {page_num}]")
    page_url = f"https://books.toscrape.com/catalogue/page-{page_num}.html"
    base_url = 'https://books.toscrape.com/catalogue/'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(page_url, headers=headers)
    soup = BeautifulSoup(response.text,'html.parser')
    books = soup.find_all('article', class_='product_pod')
    for book in books:
        book_href = book.h3.a['href']
        book_link = base_url + book_href
        res = requests.get(book_link, headers=headers)
        book_soup = BeautifulSoup(res.text,'html.parser')
        title = book_soup.find('div', class_='product_main').h1.text
        category = book_soup.select('ul.breadcrumb li')[2].text.strip()
        bread_books.append({"title": title, "category": category})

df = pd.DataFrame(bread_books)
df.to_excel("bread_books_50_pages.xlsx", index=False)