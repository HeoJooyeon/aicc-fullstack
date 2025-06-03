import requests
from bs4 import BeautifulSoup

page_url = 'https://books.toscrape.com/catalogue/page-1.html'
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
    print(f"제목: {title} | 장르: {category}")