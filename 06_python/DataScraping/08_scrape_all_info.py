# 전체 정보 수집 예시
import requests, re, time
from bs4 import BeautifulSoup
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0'}
rating_map = {'One':1, 'Two':2, 'Three':3, 'Four':4, 'Five':5}
book_list = []

for page_no in range(1, 3):
    url = f'https://books.toscrape.com/catalogue/page-{page_no}.html'
    soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
    for book in soup.find_all('article', class_='product_pod'):
        star_rating = rating_map.get(book.p['class'][1], 0)
        price = book.find('p', class_='price_color').text.strip()
        image = 'https://books.toscrape.com/' + book.find('img')['src'].replace('../', '')
        book_href = 'https://books.toscrape.com/catalogue/' + book.h3.a['href']
        detail = BeautifulSoup(requests.get(book_href, headers=headers).text, 'html.parser')
        title = detail.find('div', class_='product_main').h1.text
        category = detail.select('ul.breadcrumb li')[2].text.strip()
        upc = detail.find('th', string='UPC').find_next_sibling().text
        availability = detail.find('th', string='Availability').find_next_sibling().text
        in_stock = int(re.findall(r'\d+', availability)[0])
        book_list.append({
            "title": title, "price": price, "category": category,
            "stars": star_rating, "upc": upc, "availability": availability,
            "in_stock": in_stock, "image_link": image
        })
        time.sleep(0.1)

df = pd.DataFrame(book_list)
df.to_excel("book_info.xlsx", index=False)