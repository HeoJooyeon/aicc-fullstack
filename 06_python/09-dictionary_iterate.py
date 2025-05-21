person = {"name": "Alex", "age": 34, "city": "New York"}

for key1 in person:
    print(key1)
    
for value in person.values():
    print(value)
    
for key2 in person:
    print(f"{key2} -> {person[key2]}")
    
for key,value in person.items():
    print(f"{key} -> {value}")
    
parent = [{"parent type": "mom", "Name": "MAMA", "Age": 70},
          {"parent type": "dad","Name": "DADDY", "Age": 72}]
for person in parent:
    for key,value in person.items():
       print(f"{key} : {value}")

family = {"mom": {"name": "MAMA", "age": 70},
          "dad": {"name": "DADDY", "age": 72}}
for name, info in family.items():
    print(f"\nParent type: {name}")
    parent_name = f"Name: {info['name']}"
    parent_age = f"Age: {info['age']}"
    print(f"\n{parent_name}")
    print(f"\n{parent_age}")
    
subject = {"student1": ["건축공학","공학설계"],
          "student2": ["사진학","디자인기법"],
          "student3": ["아동심리발달","인지장애","언어치료"]}

for name, info in subject.items():
    print(f"{name} 수강과목:")
    for value in info:
        print(f"- {value}")