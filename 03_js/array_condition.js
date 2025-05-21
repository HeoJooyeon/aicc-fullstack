//1. 좋아하는 차 종류 세가지를 배열에 넣으시고, 그 중 첫번째 차를 출력하세요.
let carList = ["car1", "car2", "car3"];
const firstCar = carList[0];
console.log(firstCar);

//2. 좋아하는 여행지 4곳을 배열에 넣으시고, 그 중 세번째 도시를 출력하세요.
let cityList = ["city1", "city2", "city3", "city4"];
const thirdCity = cityList[2];
console.log(thirdCity);

//3. 좋아하는 차 3가지를 배열에 넣으시고, 첫번째 차를 다른 차로 바꾸세요. 그리고 출력하세요.
let teaList = ["tea1", "tea2", "tea3"];
teaList[0] = "tea4";
console.log(teaList);

//4. push: 배열 추가
let citiesVisited = ["cities1", "cities2"];
citiesVisited.push("cities3");
console.log(citiesVisited);

//5. pop: 배열 빼기
let teaOrders = ["teaOrder1", "teaOrder2", "teaOrder3", "teaOrder4"];
teaOrders.pop();
console.log(teaOrders);

//6. 좋아하는 음식 4가지를 배열에 입력하시고, 맨 마지막 음식을 출력하세요.
let foodList = ["food1", "food2", "food3", "food4"];
const lastFood = foodList.pop();
console.log(lastFood);

//7. 세가지 좋아하는 차의 배열을 만든시고, 다른 변수에 할당하세요. 그리고 두가지 배열 모두 마지막 차가 제거된 채로 출력하세요.
let carcarList = ["carcar1","carcar2","carcar3"];
let carcarListCopy = carcarList;
carcarList.pop();
console.log(carcarList);
console.log(carcarListCopy);
//배열을 다른 변수에 할당하면 값이 아니라 참조가 복사됨

//8. 전개 연산자 사용해서 복사: 주소 참조되지 않음음
let topCities1 = ["Berlin", "Singapore", "New York"];
let copyCities1 = [...topCities1];
console.log(topCities1);
console.log(copyCities1);

//9. 참조 복사를 통해 동일하게 세가지 차 종류를 가진 배열 두개를 만드시고, 하나의 코드를 통해 차 종류 하나를 두 배열 모두의 끝에 추가해서 출력하세요.
let favoriteTeas = ["green tea", "oolong tea", "chai"];
let copyMyTeas = favoriteTeas;
copyMyTeas.push("black tea");
console.log(favoriteTeas);
console.log(copyMyTeas);

//10. concat
//concat() 메서드는 두 개 이상의 배열을 결합하여 새로운 배열을 반환합니다.
//원본 배열은 변경되지 않습니다.
let europeanCities = ["Paris", "Rome"];
let asianCities = ["Tokyo", "Bangkok"];
let concatCities = europeanCities.concat(asianCities);
console.log(europeanCities);
console.log(concatCities);
europeanCities.pop();
console.log(europeanCities);
console.log(concatCities);

//11.length 메서드를 이용하세요. 좋아하는 차 4가지 중 마지막 차를 출력하세요.
let teateaList = ["teatea1", "teatea2", "teatea3", "teatea4"];
let lastTeatea = teateaList[teateaList.length - 1];
console.log(lastTeatea);

//12. slice 
let teaMenus = ["Masala chai", "oolong tea", "green tea", "earl grey", "black tea"];
let sliceTea = teaMenus.slice(1, 4); //1번 인덱스부터 4번째 인덱스 전까지지
console.log(sliceTea);

//두번째 차부터 마지막까지 출력이 되도록 하세요.
let sliceTea2 = teaMenus.slice(1);
console.log(sliceTea2);
let lastTwoTea = teaMenus.slice(-2);
console.log(lastTwoTea); //-1: 맨 마지막, -2: 맨 마지막에서 두번째
let blackTea = teaMenus.slice(-1);
console.log(blackTea);

let cityBucketList = ["Kyoto", "London", "Cape Town", "Vancouver"];
let isLondonInList = cityBucketList.includes("London");
console.log(isLondonInList);

//위 배열에 파리가 있는 여부를 확인할 수 있는 코드를 작성하세요.
let isParisInList = cityBucketList.includes("Paris");
console.log(isParisInList);

let trueSentence = true;
if(!trueSentence){
    console.log("YES");
}else{
    console.log("NO");
}

//배열 cityBuketList 안에 Seoul이 있으면, "저는 서울에 가고 싶어요",
//없으면 "저는 서울에 갈 생각이 없어요"라고 출력이 되도록 하세요.

let isSeoulInList = cityBucketList.includes("Seoul");
if(isSeoulInList){
    console.log("저는 서울에 가고 싶어요");
}else{
    console.log("저는 서울에 갈 생각이 없어요");
}

//items는 배열입니다. items 안에 요소가 있는지 없는지를 확인하는 코드를 작성하세요
let items = ["item1","item2","item3"];
let itemsCheck = items.length;
if(itemsCheck === 0){
    console.log("ITEM NO")
}else{
    console.log("ITEM YES")
}

//수학과 영어 성적이 나왔습니다. 둘 중 어떤 과목의 성적이 높은지를 확인하는 코드를 작성하세요
let math = 23;
let english = 49;
if(math < english){
    console.log("ENGLISH WIN");
}else{
    console.log("MATH WIN")
}

//동점인 경우 추가가
let mathS = 23;
let englishS = 23;
if(mathS < englishS){
    console.log("ENGLISH WIN");
}else if(mathS > englishS){
    console.log("MATH WIN");
}else{
    console.log("SAME");
}

//변수에 string, number 혹은 Boolean을 입력가능합니다. 어떤 타입의 변수인지를 확인하는 코드를 작성하세요
let whatType = true;
if(typeof whatType === "number"){
    console.log("number");
}else if(typeof whatType === "boolean"){
    console.log("boolean");
}else{
    console.log("else");
    console.log(typeof whatType);
}





