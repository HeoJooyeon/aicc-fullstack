def greet(name, age):
    print(f"Hello,{name} you're {age} years old.")

# for i in range(0,10):
#     greet("name",100)
    
cnt = 0
while cnt < 10:
    greet("name",100)
    cnt += 1


num = int(input("몇 명에게 인사할까요?"))
for i in range(num):
    name = input("이름을 입력하세요: ")
    age = input("나이를 입력하세요:")
    greet(name,age)
    
    
def car_detail(car_type="truck", car_name="bmw"):
    print(f"I have {car_type} named {car_name}.")

car_detail("sedan")

def format_name(first_name, last_name):
    full_name = f"{first_name} {last_name}"
    return full_name.title()


print(format_name("h","jy"))


def cal(num1,num2):
    print(f"{num1} + {num2} = {num1+num2}")
    print(f"{num1} - {num2} = {num1-num2}")
    print(f"{num1} * {num2} = {num1*num2}")
    if num2 == 0:
        print(f"{num1} / {num2} = 0으로 나눌 수 없습니다.")
    else:
        print(f"{num1} / {num2} = {num1/num2}")
        
num1 = input("첫 번째 숫자를 입력하세요 : ")
num2 = input("두 번째 숫자를 입력하세요 : ")
cal(float(num1),float(num2))




def build_profile(fname, lname):
   name = {"first": fname,"last": lname}
   return name
    
user = build_profile("Gina","Kibo")
print(user)

# Bulid a dictionary
"""docstring"""


def sum_numbers(*args):
    print(f"The sum : {sum(args)}")
    
sum_numbers(2,45,1,34,89,10)


def build_profile2(fname, lname,**args):
#    name = {}
#    name.update(args)
#    name.update({"first": fname,"last": lname})
#    return name
   args["first_name"] = fname
   args["last_name"] = lname
   return args
    
print(build_profile2("James","Bond",location="UK",age=57,field="entertainment"))

plants = ["lemon","mango","apple","banana","avocado"]
plants_1 = ["lemon-tree","mango-tree","apple-tree"]

def forPlants(args):
    for value in args:
        print(f"Watering {value}.")

forPlants(plants)
forPlants(args = plants_1)

def build_profile_2(first, last, **details):
    details["first_name"] = first
    details["last_name"] = last
    return details
first = input("Enter first name: ")
last = input("Enter last name: ")

details = {}

print("\nEnter additional resume details (type 'done' as key to finish):")

while True:
    key = input("Field name (e.g. job_title, skills, etc. ): ")
    
    if key.lower() == "done":
        break
    value =  input(f"Value for '{key}': ")
    
    details[key] = value

# a,r,w,d -> "a,r,w,d" -> ['a','r','w','d']

resume = build_profile_2(first,last, **details)
print("\n=== Resume ===")
for key, value in resume.items():
    print(f"{key.capitalize()}: {value}")