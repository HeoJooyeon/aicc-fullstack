# 0~9까지
list_nums = []
for value in range(0,10):
    list_nums.append(value)
print(list_nums)

# list_len = len(list_nums)
# sum_list = 0
# for value in range(list_len):
#     sum_list += list_nums[value]
# print(f"Sum: {sum_list}")    


# Max: 9
# Min: 0
# Sum: 45
max_num = max(list_nums)
min_num = min(list_nums)
sum_total = sum(list_nums)
    
print(f"Max: {max_num}")
print(f"Min: {min_num}")
print(f"Sum: {sum_total}")

# number = []
# for value in range(0,200):
#     if value % 2 == 0:
#         number.append(value)
# print(f"Number: {number}")

list_num = []
for num in range(0, 200, 2):
    # num = num*2
    list_num.append(num)
print(f"Number: {list_num}")

even_nums = list(range(0,100,2))
print(even_nums)

# list comprehension
ls_sqrs = [value * 2 for value in range(0,10)]
print(ls_sqrs)

