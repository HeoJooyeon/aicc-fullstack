/* function makingTea(kindofTea){
    return `I am making ${kindofTea}`;
}
let teaOrder1 = makingTea("Black Tea");
console.log(teaOrder1); */

//위와 같은 함수를 변경하여 console.log라는 매소드 없이 콘솔창에 표기되도록 하세요
function makingTea(kindofTea){
    console.log(`I am making ${kindofTea}`);
}
makingTea("Black Tea");

const makingTea2 = (kindofTea) => { //arrow function
    return `I am making ${kindofTea}`;
}
//실행해서 터미널에 결과가 나오도록 하세요
let teaOrder2 = makingTea2("Black Tea2");
console.log(teaOrder2);

function orderTea(teaType) {
    function confirmOrder(){
        return `Order confirmed for ${teaType}`; //`: backtick, 백틱
    }
    return confirmOrder();
}
//위 함수를 실행해서 Order confirmed for chai 라고 실행창에 나오도록 하세요
console.log(orderTea("chai"));

//가격과 양을 받아 와서 총 주문액을 출력하는 함수를 arrow function 사용하여 구현하시고
//아래와 같이 출력이 되도록하세요
//가격은 30이고 수량은 2입니다 총 주문액은 60입니다
const priceNum = (price, num) => {
    console.log(`가격은 ${price}이고 수량은 ${num}입니다 총 주문액은 ${price*num}입니다`);
}
priceNum(30, 2);

//차 주문 내용을 받는 기능의 클래식 함수를 구현하세요 두개의 함수로 구성하시고 하나는 주문받는 차의 이름을 다른 하나는 주문된 내용을 영수증에 주문내역이라고 써서 결과적으로 아래와 같이 터미널에 찍혀 나오도록 해 주세요
//주문내역 : earl grey
function orderTeaName(teaType){
    return teaType;
}
function orderTeaContent(){
    return `주문내역 : `;
}
console.log(orderTeaContent() + orderTeaName("earl grey"));
//다른 방식
function makeTea(typeOfTea){
    return `주문내역1 : ${typeOfTea}`;
}
function processTeaOrder(teaFunction){
    return teaFunction("earl grey1");
}
let order = processTeaOrder(makeTea);
console.log(order);

//createTeaMaker라는 클래식 함수와 내부 함수를 통해 차를 만드는 사람 이름, 차의 종류를 각각 받아오게 하시고, 함수 내부에 차의 량을을 선언하여 다음과 같이 출력이 되도록 하세요
//Making grean tea by jy about 100 ml
//clousure 함수
function createTeaMaker(name, teaType) {
    function confirmTeaOrder(score){
        console.log(`Making ${teaType} by ${name} about ${score} ml`);
    }
    confirmTeaOrder(100);
}
createTeaMaker("jy", "green tea");
//다른 방식
function createTeaMaker1(name){
    // let score = 101;
    // return function(teaType){
    //     return `Making1 ${teaType} by ${name} about ${score} ml`;
    // }
    return function(teaType){
        let score = 102;
        return `Making2 ${teaType} by ${name} about ${score} ml`;
    }
}
let teaMaker1 = createTeaMaker1("jy1");
let result1 = teaMaker1("green tea1");
console.log(result1);












