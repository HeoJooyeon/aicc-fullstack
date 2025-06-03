from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://quotes.toscrape.com/js/")

all_quotes = []
while True:
    time.sleep(1)
    quotes = driver.find_elements(By.CLASS_NAME, "quote")
    for quote in quotes:
        text = quote.find_element(By.CLASS_NAME, "text").text
        author = quote.find_element(By.CLASS_NAME, "author").text
        tags = [tag.text for tag in quote.find_elements(By.CLASS_NAME, "tag")]
        all_quotes.append({"quote": text, "author": author, "tags": tags})
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, "li.next > a")
        next_button.click()
    except:
        break
driver.quit()

text_data = " ".join([q["quote"] for q in all_quotes])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("명언 텍스트 워드클라우드")
plt.tight_layout()
plt.show()