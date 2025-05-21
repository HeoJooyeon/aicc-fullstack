//컴퓨터라는 객체의 오브젝트에 cpu가 12개 담겨 있다는 것을 선언하시고, 컴퓨터의 cpu가 12개로 터미널에 나오도록 하세요
let computer = { cpu : 12 };
console.log(computer.cpu);
console.log(computer["cpu"]);

let lenovo = {
    screen : "HD",
    __proto__ : computer,
}
console.log(lenovo);
console.log(lenovo.cpu);

let lenovo2 = {
    screen : "HD",
    cpu: 20,
    __proto__ : computer,
}
console.log(lenovo2);
console.log(lenovo2.cpu);

let genericCar = { types: 4 };
let tesla = {
    driver: "AI",
};
Object.setPrototypeOf(tesla, genericCar);
//Object.setPrototypeOf(obj, prototype);
console.log(tesla.driver);
console.log(tesla.types);
console.log(tesla);
console.log(tesla.__proto__);
console.log(`tesla`, Object.getPrototypeOf(tesla));

let x;
console.log(x);
function test(){};
console.log(test());
let person = {name: "John"};
console.log(person.age);
let y = null;
console.log(y);
let person1 = {name: "Jane", age: null};
console.log(person1.age);
console.log(typeof undefined);
console.log(typeof null);
console.log(undefined == null);
console.log(undefined === null);

let price1 = 30; 
let quantity1 = 2; 
const calculateTotal1 = (price, quantity) => price * quantity;
let result1 = calculateTotal1(price1, quantity1);
console.log(`가격은 ${price1} 이고 수량은 ${quantity1}입니다. 총 주문액은 ${result1}입니다.`);
const calculateTotal = (price, quantity) => price * quantity;
console.log(calculateTotal(30, 2)); 
let totalCost = calculateTotal(30, 2);
console.log(totalCost); 



