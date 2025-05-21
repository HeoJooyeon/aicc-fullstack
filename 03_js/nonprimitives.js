/* Primitive Type 종류
Number  숫자(정수, 실수 포함)   let a = 42;
BigInt  매우 큰 정수를 다룰 때 사용 let bigNum = 12345678901234567890n;
String  문자열 데이터   let name = "John";
Boolean 참/거짓 값  let isAdmin = true;
undefined 값이 할당되지 않은 상태 let x; //undefined
Null 값이 없음(명시적 비어 있음) let empty = null;
Symbol (ES6)   유일한 식별자를 만들 때 사용 let id = Symbol("id"); */


/* Non-Primitive Type 종류
Object 키-값 쌍을 저장하는 데이터 구조 let user = { name: "John", age: 30};
Array   순서가 있는 리스트 형태의 객체 let arr = [1, 2, 3];
Function    함수도 객체의 한 종류 function sayHello() { console.log("Hello"); };
Date    날짜 및 시간을 다루는 객체  let now = new Date();
RegExp  정규 표현식 객체    let regex = /abc/; */

//primitive 새로생성
let a = 10;
let b = a;
a = 20;
console.log(a);
console.log(b);

//non primitive 참조
let obj1 = { name: "Alice" };
let obj2 = obj1; //obj2는 obj1의 참조값을 저장 (주소 공유)
obj1.name = "Bob";
console.log(obj1.name);
console.log(obj2.name);

let str = "Hello";
console.log(str.length);

let temp = new String("Hello");
console.log(temp.length);

//primitive
console.log(typeof 42);
console.log(typeof "Hello");
console.log(typeof true);
console.log(typeof undefined);
console.log(typeof null); // "object" (버그이지만 유지됨)
console.log(typeof Symbol("id"));

//non primitive
console.log(typeof {});
console.log(typeof []);
console.log(typeof function(){});

const username = {
    "first name": "good name",
    isLoggedin: true,
};

console.log(username["first name"]); //대괄호 표기법
//키에 공백(띄어쓰기)이 포함된 경우, 점 표기법이 아닌 대괄호 표기법을 사용

username.firstname = "jy",
username.lastname = "h",
//const도 추가는 가능하다
console.log(username);

let today = new Date();
console.log(today);
console.log(today.getDate());

let year = today.getFullYear();
let month = today.getMonth() + 1;
let day = today.getDate();

let ymd =`${year}-${month}-${day}`;

console.log(ymd);

//Array
let anotherUser = ["jy", true];

console.log(anotherUser[0]);
console.log(anotherUser[1]);

//let이라서 새로생성 가능
anotherUser = "hjy";

console.log(anotherUser);

let isValue = "2abc";

console.log(typeof isValue);
console.log(typeof Number(isValue)); //number
console.log(Number(isValue)); //NaN, not a number
console.log(typeof isValue);
