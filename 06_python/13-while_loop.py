ingredients = ["avocado", "tomato", "avocado", "apple", "cilantro", "avocado"]
num = 0

print(f"원래 재료 리스트: {ingredients}")
# output = []
# for ingre in ingredients:
#     if ingre != "avocado":
#         output.append(ingre)
#     else:
#         num += 1
        
while "avocado" in ingredients:
    ingredients.remove("avocado")
    num +=1

print(f"avocado는 총 {num}개 있었습니다.")
print(f"avocado 제거 후 리스트: {ingredients}")

ingredients2 = ["avocado", "tomato", "avocado", "apple", "cilantro", "avocado", "tomato"]
print(f"원래 재료 리스트:  {ingredients2}")

unique_ingredients = []
for item in ingredients2:
    if item not in unique_ingredients:
        unique_ingredients.append(item)
        
print(f"중복 없는 재료 리스트: {unique_ingredients}")


# 각 아이템의 갯수 : {}
item_counts = {}

for item in ingredients2:
    if item in item_counts:
        item_counts[item] += 1
    else:
        item_counts[item] = 1
print(f"각 아이템의 갯수: {item_counts}")

