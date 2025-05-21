bool = False

# if bool == True:
if bool:
    print("you are tall")
else:
    print("not tall")

# weather = "rainy"라는 것을 만들어서, 비가 오면 "Don't forget your umbrella"가 출력되도록 하시고, 그렇지 않으면 "Grab your sunglasses"가 출력되도록 하세요

weather = "rainy"
if weather == "rainy":
    print("Don't forget your umbrella")
else:
    print("Grab your sunglasses")
    
# 나이가 10살보다 많으면 "You are older than 10", 그렇지 않으면 "You are younger than 10"가 출력하도록 하세요

age = 10
if age > 10:
    print("You are older than 10")
else:
    print("You are younger than 10")
    

temp = 60
if temp < 45:
    print("두꺼운 옷을 가져가세요")
elif temp >= 45 and temp < 60:
    print("가벼운 옷도 괜찮아요")
elif temp >= 60:
    print("아주 따뜻합니다")

participants = ["Bob","Jerry","Adam"]
result = "Bob" in participants
print(result)

partic_result = "hjy" in participants
if partic_result:
    print("registerd")
else:
    print("not registerd")









