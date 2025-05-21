let score = 102;
let bonus = 25;

//let totalScore = score + bonus;
//const totalScore = score + bonus;
//console.log(totalScore);

//totalScore = "string";
//console.log(totalScore);

//var, let(변수 값의 변화가 가능함), const(상수)

let addition = 4 + 5;
let subtract = 9 - 3;
let mult = 3 * 5;
let divi = 8 / 2;
let remainder = 9 % 2;
let expo = 2 ** 3;

console.log(remainder);
console.log(expo);

let myscore = 110;
myscore += 1;
//myscore = myscore + 1;
//myscore++;
console.log(myscore);

let credits = 56;
credits--; //credits -=
console.log(credits);

//비교연산자
let num1 = 3;
let num2 = 3;
let num3 = 6;

console.log(num1 == num2); 
console.log(num1 != num2);
console.log(num1 > num2);

//= : 대입
//== : 느슨한 비교, 동등 연산자
//===: 엄격한 비교, 일치 연산자

console.log(5 === "5"); //데이터 타입과 값이 같아야 true
console.log(5 == "5"); //데이터 타입의 값과 같으면 true
console.log(0 == false); //true = 1, false = 0

console.log(null == undefined);
//null은 변수에 값이 명시적으로나 의도적으로 없다. undefined는 할당된 값 자체가 없다
console.log([] == false);

let newScore = (2 * 3 + 2 - 1);
console.log(newScore);



//변환
let gameName = "spiderman";
gameName = "batman";
console.log(gameName);
gameName = 19;
console.log(gameName);

let username = "jy";
username = "hjy";
console.log(username);