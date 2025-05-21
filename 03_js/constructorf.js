//생성자(constructor) 함수
function Person(name, age){
    this.name = name;
    this.age = age;
}
let person1 = new Person("John", 30);
console.log(person1);

const person3 = new Person("Bob", );
//console.log(person3.name); 
//전역(global)변수가 있는 경우는 불러올 수 있다.

class Person1{
    constructor(name, age){
        this.name = name;
        this.age = age;
    }
}
//위 클래스를 활용하여 이름에에 John을 나이에 80을 입력하고 출력해보세요
const clsPerson1 = new Person1("John1",80);
console.log(clsPerson1.name);
console.log(clsPerson1.age);

//Car 라는 생성자 함수를 만들고 생산회사와 모델명을 입력 받을 때마다 새로운 객체가 생성되도록하여 다음과 같은 내용이 출력되도록 하세요.
//Car { make: 'Toyota', model: 'Camry'}
class Car{
    constructor(make,model){
        this.make = make;
        this.model = model;
    }
}
const clsCar = new Car("Toyota","Camry");
console.log(clsCar);

//생성자 함수 내의 메서드
function Tea(type){
    this.type = type;
    this.describe = function () {
        return `this is a cup of ${this.type}`;
    }
}
const lemonTea = new Tea("lemon tea");
console.log(lemonTea.describe());
console.log(lemonTea);

function Animal(species){
    this.species = species;
}
Animal.prototype.sound = function () {
    return `${this.species} makes a sound`;
};
//위 생성자를 활용하여 Dog makes a sound가 출력되도록 하세요
//cat makes a sound가 출력되도록 하세요
const dogAnimal = new Animal("Dog");
console.log(dogAnimal.sound());
console.log(dogAnimal);
const catAnimal = new Animal("Cat");
console.log(catAnimal.sound());

//에러 처리
function Drink(name){
    if(!new.target) {
        throw new Error("Drink must be called with new keyword");
    }
    this.name = name;
}
//에러 메시지를 확인하세요
//Error: Drink must be called with new keyword
//const cokeDrink = Drink("coke");






