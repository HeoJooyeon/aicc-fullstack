let car = {
    make: "Toyota",
    model: "Camry"
}
console.log(car);
console.log(car.make);

//위 car 객체에 start라는 매소드와 생산년도라는 속성을 만드시고 해당 매소드를 실행하여 아래와 같은 내용이 실행창에 뜨도록 하세요
//Toyota car got started in 2020
let car1 = {
    make: "Toyota",
    model: "Camry",
    start: function () {
        //console.log(`${car1.make} car got started in ${car1.year}`);
        console.log(`${this.make} car got started in ${this.year}`);
        //this는 객체 내부에서 현재 객체(car)를 참조
    },
    year: 2020,
}
car1.start();

/*  setTimeout(() => {
    console.log(`${car1.make} car got started in ${car1.year}`);
},1000), //콜백함수, async*/

//bike라는 객체 안에 제조사 이름과 제조연도를 넣으세요
let bike = {
    make: "Yamaha",
    year: 2022,
};
bike.start = car1.start;
bike.start();

let startFunc = car1.start;
startFunc();

//Animal 객체에 type을 받아와서 type의 값으로 저장하는 함수를 만드세요
//해당 객체에 프로토타입으로 speak라는 매소드를 추가하셔서 해당 객체의 type이 makes a sound 하도록 해 주세요
//type으로 Dog를 받아와서 Dog makes a sound와 같이 출력을 하도록 해 주세요
function Animal(type){
     this.type = type; 
}
Animal.prototype.speak = function () {
    console.log(`${this.type} makes a sound`);
};
const dogAnimal = new Animal("Dog");
dogAnimal.speak();

//모든 배열이 상속받는 프로토타입 객체
Array.prototype.explainArray = function(){
    console.log(`배열입니다`);
}
/* const explain = new Array();
explain.explainArray(); */
let myArray = [1, 2, 3];
myArray.explainArray();

//프로토타입 객체를 활용하여 배열 안에 있는 내용들이 아래와 같이 모두 출력되도록 하세요
//배열의 내용은 1,2,3
Array.prototype.jy = function(){
    console.log(`배열의 내용은 ${this}`);
}
myArray.jy();

//상속
class Vehicle{
    constructor(make, model){
        this.make = make;
        this.model = model;
    }
    start(){
        console.log(`${this.model} is a vehicle from ${this.make}`);
    }
}
class Cars extends Vehicle {
    drive(){
        console.log(`${this.make} : This is an inheritance example`);
    }
}
let cars = new Cars("Toyota1","Camry1");
cars.start();
cars.drive();
let vehi = new Vehicle("Toyota1","Camry1");
vehi.start();

//다형성 Polymorphism : 다양한 형태
class Bird{
    fly(){
        console.log(`Flying...`); 
    }
}
class Penguin extends Bird{
    fly(){ //재정의(오버라이드)
        console.log(`Penguins can't fly`);
    }
}
let bird = new Bird();
let penguin = new Penguin();
bird.fly();
penguin.fly();

//Abstraction 추상화
class CoffeeMachine{
    start(){
       return `Strating the machine...`;
    }
    brewCoffee(){
        return `Brewing coffee`;
    }
    pressStartButton(){
        let msgone = this.start();
        let msgTwo = this.brewCoffee();
        return `${msgone} + ${msgTwo}`;
    }
}
let coffeeMachine = new CoffeeMachine();
console.log(coffeeMachine.start());
console.log(coffeeMachine.brewCoffee());
console.log(coffeeMachine.pressStartButton());

//arrow 함수로 변경하세요
class CoffeeMachine1{
    start = () => {
       return `Strating the machine1...`;
    };
    brewCoffee = () => {
        return `Brewing coffee1`;
    };
    pressStartButton = () => {
        let msgone = this.start();
        let msgTwo = this.brewCoffee();
        return `${msgone} + ${msgTwo}`;
    };
}
let coffeeMachine1 = new CoffeeMachine1();
console.log(coffeeMachine1.start());
console.log(coffeeMachine1.brewCoffee());
console.log(coffeeMachine1.pressStartButton());
