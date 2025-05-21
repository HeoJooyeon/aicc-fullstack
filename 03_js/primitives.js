let balance = 120; //원시 타입 (primitive type)
let anotherBalance = new Number(120); //Number 객체의 인스턴스(instance)
//숫자를 감싼 객체(Wrapper Object), nonprimitive type

console.log(balance);
console.log(anotherBalance);

console.log(typeof balance); //데이터 타입을 찾음
console.log(typeof anotherBalance);
console.log(anotherBalance.valueOf());

//비교연산자를 통해 두개를 비교하시고, 설명하세요
console.log(balance == anotherBalance); //true(값 비교)
console.log(balance === anotherBalance); //false(타입, 값까지 비교)

let myString = "hello";
let myStringOne = "Good Morning";
let username = "jy";

let greeting = myString + " " + username;
console.log(greeting);

let greetMessage = `Hello ${username}, ${myStringOne}!`;
console.log(greetMessage);
//문자열 보간

let num11 = 2;
let num22 = 3;
let demoOne = `Value is ${num11 * num22}`;
console.log(demoOne);

let sm1 = Symbol("jy");
let sm2 = Symbol("jy");
console.log(sm1 === sm2);