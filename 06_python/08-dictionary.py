person = {"name": "Alex", "age": 34, "city": "New York"}


print(person["name"])
# Alex

person["email"] = "alex@gmail.com"
print(person)

person.popitem()
print(person)

person.pop("age")
print(person)

person["is_employed"] = True
person["age"] = 35
print(person)

person.clear()
print(person)