from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://quotes.toscrape.com/js/")
quotes = driver.find_elements(By.CLASS_NAME, "quote")

for quote in quotes:
    text = quote.find_element(By.CLASS_NAME, "text").text
    author = quote.find_element(By.CLASS_NAME, "author").text
    tags = quote.find_elements(By.CLASS_NAME, "tag")
    tags_text = [tag.text for tag in tags]
    print(f" \"{text}\" - {author} / tags: {', '.join(tags_text)}")