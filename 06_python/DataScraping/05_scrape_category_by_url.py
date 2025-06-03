import requests
from bs4 import BeautifulSoup
import re

num = 1000
url = "http://books.toscrape.com/catalogue/page-1.html"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
books = soup.find_all('article', class_='product_pod')

for book in books:
    title = book.h3.a['title']
    clean_title = re.sub(r"-+", "-", re.sub(r"[ \[\]#:,)]", "-", re.sub(r"\(.*?\)", "", title))).lower()
    url2 = f"http://books.toscrape.com/catalogue/{clean_title}_{str(num)}/index.html"
    response2 = requests.get(url2)
    soup2 = BeautifulSoup(response2.text, 'html.parser')
    bread = soup2.find('ul', class_='breadcrumb').find_all('li')[2].a.text.strip()
    print(f"제목:{title} | 장르: {bread}")
    num -= 1