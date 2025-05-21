from pathlib import Path
import json

names = ["James", "Ruth", "Mary"]

path = Path("name.json")

with path.open("w") as file:
    contents = json.dumps(names)
    file.write(contents)

with path.open("r") as file:
    contents = json.load(file)
    print(contents)
    



countrys = []

while True:
    print("Enter country names. Type 'quit' to finish")
    country = input("Country : ")
    if country.lower() == "quit":
        break
    countrys.append(country)

path2 = Path("country.json")

with path2.open("w") as file:
    contents = json.dumps(countrys)
    file.write(contents)

with path2.open("r") as file:
    contents = json.load(file)
    print(contents)