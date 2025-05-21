import matplotlib.pyplot as plt
# pip install matplotlib
from collections import defaultdict
from matplotlib import rcParams
import requests

rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def fetch_todo_data():
    url = "https://jsonplaceholder.typicode.com/todos"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def process_todo_data(data):
    # completed_count_by_user = defaultdict(int)
    completed_count_by_user = {}
    for item in data:
        if item["completed"]:
            user_id = item["userId"]
            if user_id not in completed_count_by_user:
                completed_count_by_user[user_id] = 0
            completed_count_by_user[item["userId"]] += 1
    return completed_count_by_user
    
def plot_completed_tasks(completed_data):
    user_ids = list(completed_data.keys())
    completed_counts = list(completed_data.values())
    plt.figure(figsize=(10,5))
    plt.bar(user_ids, completed_counts, color="lightgreen")
    plt.title("사용자별 완료한 할 일 수")
    plt.xlabel("사용자 ID")
    plt.xticks(user_ids)
    plt.ylabel("완료한 일의 갯수")
    plt.tight_layout()
    plt.show()
    
    # user_id_count = {i: 0 for i in range(1, 11)}
    # for item in data:
    #     if item["completed"]:
    #         user_id = item["userId"]
    #         if user_id in user_id_count:
    #             user_id_count[user_id] += 1
    # # print(user_id_count)
    # days = list(user_id_count.keys())
    # costs = list(user_id_count.values())

def main():
    todo_data = fetch_todo_data()
    if todo_data:
        completed_data = process_todo_data(todo_data)
        if completed_data:
            plot_completed_tasks(completed_data)
        else:
            print("completed_data가 없습니다.")
    else:
        print("todo_data가 없습니다.")

if __name__ == "__main__":
    main()