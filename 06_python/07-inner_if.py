#토요일에만 아침을 먹습니다. 그런데 토요일에도 냉장고와 달걀과 베이컨이 있으면 아침을 먹고, 없으면 씨리얼을 먹습니다.

day = "Sunday"
fridge_contents = ["eggs","bacon"]

if day == "Saturday":
    if "eggs" in fridge_contents and "bacon" in fridge_contents:
        print("아침")
    else:
        print("cereal")
else:
    print("안먹")