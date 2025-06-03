from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://quotes.toscrape.com/js/")

all_quotes = []

for page_num in range(1, 100):
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "quote"))
    )
    quotes = driver.find_elements(By.CLASS_NAME, "quote")

    for quote in quotes:
        text = quote.find_element(By.CLASS_NAME, "text").text
        author = quote.find_element(By.CLASS_NAME, "author").text
        tags = [tag.text for tag in quote.find_elements(By.CLASS_NAME, "tag")]
        all_quotes.append({"text": text, "author": author, "tags": ", ".join(tags)})

    try:
        next_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "li.next > a"))
        )
        next_button.click()
    except:
        break

driver.quit()

df = pd.DataFrame(all_quotes)
df.to_excel("quotes_result.xlsx", index=False)