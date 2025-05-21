activities = ["hiking", "swimming", "museum", "picnic"]
group_interests = ["art", "history","swimming"]

# activities에 "museum"이 있고, group_interests에 "art" 또는 "history"가 있다면 "We should visit the museum!"
# "swimming"이 activities와 group_interests에 모두 있다면:
# "Looks like a day at the pool is in order!"
# 위 조건이 둘 다 맞지 않을 경우 "Let's plan a picnic instead."가 출력력되도록 해주세요/

# elif, elif.. else

mus = "museum" in activities
art_his = "art" in group_interests or "history" in group_interests

if mus and art_his:
    print("We should visit the museum!")
elif "swimming" in activities and "swimming" in group_interests:
    print("Looks like a day at the pool is in order!")
else:
    print("Let's plan a picnic instead.")
    
age = 22
toppings = ["olives","tomatoes"]

# 토핑이라는 리스트에 첫번째 item이 "mangoes"가 아니면 print("No mangoes for toppings")라고 출력해주세요.
# 나이가 23세가 아니면 "Not 23"

if toppings[0] != "mangoes":
    print("No mangoes for toppings")
if age != 23:
    print("Not 23")

