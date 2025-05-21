import requests
# pip install requests

def fetch_books(query, start_index, max_results=10):
    api_url = f"https://www.googleapis.com/books/v1/volumes?q={query}&startIndex={start_index}&maxResults={max_results}"
    response = requests.get(api_url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"요청 실패: {response.status_code}")
        return None

def display_books(books):
    if not books or "items" not in books:
        print("검색 결과가 없습니다.")
        return
    for item in books["items"]:
        info = item["volumeInfo"]
        title = info.get("title", "제목 없음")
        authors = ", ".join(info.get("authors", ["저자 없음"]))
        publisher = info.get("publisher", "No publisher")
        description = info.get("description", "No description")
        print(f"\nTitle: {title}\nAuthors: {authors}\nPublisher: {publisher}\nDescription: {description}")

def main():
    search_term = input("검색할 책 키워드를 입력하세요 : ")
    start_index = 0
    max_results = 10
    while True:
        books = fetch_books(query=search_term,start_index=start_index,max_results=max_results)
        if not books.get("items"):
            print("더 이상 결과가 없습니다.")
        display_books(books)
        next_step = input("'n'을 누르시면 또다른 결과가 뜹니다. 그외의 키를 누르시면 프로그램이 종료됩니다.")
        if next_step != "n":
            break
        start_index += max_results
        print(f"Pagination 번호: {start_index}")

if __name__ == "__main__":
    main()
    
    