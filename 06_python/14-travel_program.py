countrys = {}

print("희망하시는 여행을 계획해보세요")
print("입력이 끝나시면 'done' 이라고 쓰세요.")

while True:
    user_input = input("어떤 나라에 여행하고 싶으세요?")
    if user_input.lower() == "done":
        break
    else:
        user_input2 = input(f"{user_input}에서 특별히 방문해보고 싶은 곳이 있나요? (답하기 싫으시면 enter를 누르시고 skip하실 수 있습니다.")
        countrys[user_input] = user_input2 if user_input2 else "특별한 여행계획이 없습니다."
        print("끝내시려면 'done' 이라고 쓰세요.")
        # countrys.append((user_input, user_input2))
        
print(f"희망 여행 국가:")
print(f"{countrys}")
for country, place in countrys.items():
    print(f"- {country} : {place}")

print("재밌고 안전한 여행이 되시길 바랍니다.")