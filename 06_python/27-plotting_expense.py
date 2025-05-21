import matplotlib.pyplot as plt
# pip install matplotlib

def get_daily_expenses():
    expenses = {}
    print("Enter your daily expenses.Type 'done' when finished.")
    
    while True:
        day = input("Enter the day (e.g. 'Monday'): ")
        if day.lower() == "done":
            break
        try:
            cost = float(input(f"Enter expenses for the {day}: $"))
            expenses[day] = cost
        except ValueError:
            print("Please enter a number.")
    return expenses


def plot_expenses(expenses):
    days = list(expenses.keys())
    costs = list(expenses.values())
    
    plt.figure(figsize=(10,5))
    plt.bar(days, costs, color="skyblue")
    plt.title("Daily Expenses")
    plt.xlabel("Day")
    plt.ylabel("Expenses($)")
    plt.tight_layout()
    plt.show()

def main():
    daily_expenses = get_daily_expenses()
    if daily_expenses:
        plot_expenses(daily_expenses)
    else:
        print("No expenses data")

if __name__ == "__main__":
    main()