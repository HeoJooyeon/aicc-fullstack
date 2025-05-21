import requests
import pymysql

def fetch_books_to_mysql(query, max_results=20):
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}"
    response = requests.get(url)
    items = response.json().get("items", [])
    
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="1111",
        db="bookdb",
        charset="utf8mb4"
    )
    
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bookdb.books (
            id VARCHAR(100) PRIMARY KEY,
            title TEXT,
            authors TEXT,
            publisher TEXT,
            pageCount INT,
            search_keyword VARCHAR(100)
        )
    """)
    
    for item in items:
        info = item.get("volumeInfo", {})
        book_id = item["id"]
        title = info.get("title", "N/A")
        authors = ", ".join(info.get("authors", [] ) )
        publisher = info.get("publisher", "N/A")
        page_count = info.get("pageCount", 0)
    
    
        cur.execute("""
            INSERT IGNORE INTO bookdb.books (id, title, authors, publisher, pageCount, search_keyword)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (book_id, title, authors, publisher, page_count, query))
    
    conn.commit()
    conn.close()
    print(f"'{query} '에 대한 정보를 MySQL에 저장 완료.")
