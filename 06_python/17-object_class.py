class Book:
    title = "book title"
    author = "hjy"
    genre = "fiction"

    #예약어
    def describe_book(self): 
        print("Describing book...")
        
# 인스턴스를 생성해서 각 메서드를 호출하세요.
mybook = Book()

print(mybook.title)
print(mybook.author)
print(mybook.genre)
print(mybook.describe_book())


class Book2:
    # self: 생성된 인스턴스 자기 자신을 가리킴
    def __init__(self, title, author, genre):
        # 인스턴스 변수
        self.title = title
        self.author = author
        self.genre = genre
    def describe_book(self):
        print(f"{self.title} by {self.author}, genre: {self.genre}")
    # print(book) 할 때 출력됨
    def __str__(self):
        return f"Book: {self.title} / Author: {self.author}"

book2 = Book2("1984", "George Orwell", "Dystopian")
book3 = Book2("Pride and Prejudice", "Jane Austen", "Romance")
book4 = Book2("The Little Prince", "Antoine de Saint-Exupery", "Fable")

book2.describe_book()
book3.describe_book()
book4.describe_book()
print(book2)
print(book3)

# 부모클래스
class Vehicle:
    def __init__(self, name, year):
        self.name = name
        self.year = year
    def __str__(self):
        return f"{self.year} {self.name}"
    def start_engine(self):
        print(f"{self.name}'s engine is now runnning.")
    def stop_engine(self):
        print(f"{self.name}'s engine is now turned off.")
        
# 자식클래스, subclass
class Car(Vehicle):
    def __init__(self, name, year, mileage):
        super().__init__(name, year)
        self.mileage = mileage
    def __str__(self):
        return f"{super().__str__()} with {self.mileage} miles"
    def drive(self, distance):
        self.mileage += distance
        print(f"Driving {distance} miles... New mileage is {self.mileage} miles.")
    def honk(self):
        print(f"{self.name} goes 'Beep beep!'")
        
veh1 = Vehicle("Truck",2015)
car1 = Car("Toyota Camry",2020,30000)

print(veh1)
veh1.start_engine()
veh1.stop_engine()
print()
print(car1)
car1.start_engine()
car1.honk()
car1.drive(120)
car1.stop_engine()


class Human:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def intro(self):
        print(f"Hi, I'm {self.name} and I'm {self.age} years old.")
    
class Student(Human):
    def __init__(self, name, age, sid):
        super().__init__(name, age)
        self.sid = sid
    # 재정의, 오버라이딩
    def intro(self):
        print(f"I'm {self.name}, {self.age} years old, and my student ID is {self.sid}")

hu1 = Human("Alice", 30)
st1 = Student("Bob", 20, "S12345")

hu1.intro()
st1.intro()



