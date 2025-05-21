fruits = ["mango","grapes","apple","banana","cherry"]

print(fruits)
print(fruits[0])

# mango -> melon
fruits[0] = "melon"
print(fruits)

# ['melon','apple','banana','cherry']
fruits.remove("grapes")
print(fruits)

for fruit in fruits:
    print(f"I like {fruit}.")
print("we are outside of fruitland ")

# ages라는 리스트를 만드시고 아래와 같은 내용으로 출력하세요
# sort()
# Before Sorting: [34,11,85,24,41]
# After Sorting: [11,24,34,41,85]
ages = [34,11,85,24,41]
print(f"Before Sorting: {ages}")
ages.sort()
print(f"After Sorting: {ages}")

# sorted fruits: ['apple','banana','cherry','melon']
# reversed fruits: ['melon','cherry','banana','apple']
fruits.sort()
print(f"sorted fruits: {fruits}")
fruits.reverse()
print(f"reversed fruits: {fruits}")

list_len = len(ages)
print(list_len)





