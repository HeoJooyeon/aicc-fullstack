questions = {
    "What is the capital of France?": "Paris",
    "What is 2 + 2?": "4",
    "Who wrote 'To Kill a Mockingbird'?": "Harper Lee",
}
score = 0
total_score = len(questions)

print("Welcome to the Quiz Game!")
print("Type 'quit' at any time to exit")

for question, answer in questions.items():
    # while True:
    #     user_input = input(f"{question} ")
    #     if user_input.lower() == answer.lower():
    #         print("Correct!\n")
    #         score += 1
    #         break
    #     else:
    #         print(f"Incorrect The correct answer is {answer}\n")
    #         break
    # if user_input.lower() == "quit":
    #     break
    user_answer = input(question + "")
    
    if user_answer.lower() == "quit":
        break
    elif user_answer.lower() == answer.lower():
        print("Correct!")
        score += 1
    else:
        print(f"Incorrect The correct answer is {answer}")
    
print(f"Quiz completed. Your score is {score}/{total_score}")
