from pathlib import Path
import json

def sava_to_json(data, filename="countries.json"):
    path = Path(filename)
    with path.open("w") as f:
        json.dump(data, f)
        
def read_from_json(filename="countries.json"):
    path = Path(filename)
    with path.open("r") as f:
        return json.load(f)

def main():
    countries = []
    print("Enter country names. Type 'quit' to finish")
    
    while True:
        country = input("Country : ")
        if country.lower() == "quit":
            break
        countries.append(country)

        sava_to_json(countries)
        saved_countries = read_from_json()
        print("\nYou've added the following countries : ")
        for country in saved_countries:
            print(country)
        
if __name__ == "__main__":
    main()