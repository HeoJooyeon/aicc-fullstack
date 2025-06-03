import requests
from bs4 import BeautifulSoup

for i in range(1, 6):
    url = f'https://books.toscrape.com/catalogue/page-{i}.html'
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')
    books = soup.find_all('article', class_='product_pod')
    print(f"[페이지: {i}]")
    for j, book in enumerate(books, 1):
        print(f"{j}. {book.h3.a['title']}")