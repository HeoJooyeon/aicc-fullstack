/* for(let i=0;i<teaMenus.length;i++){ //let i = 0; 0 < 5
    console.log(teaMenus[i]);
    //i가 0에서 4까지 출력하라는 반복문 : loop
} */

//cities 라는 배열에는 Paris, New York, Tokyo, London라는 요소가 있습니다. for 문을 활용하여 같은 내용이 배열을 cityList라는 변수명에 넣으세요
let cities = ["Paris", "New York", "Tokyo", "London"];
let cityList = [];
for(let i=0;i<cities.length;i++){
    cityList.push(cities[i]);
}
console.log(cityList);

//통장에 돈이 없습니다. 첫달에 1원으로 시작하여 다섯달에 걸쳐 받았는데, 두번째 달부터 1원씩을 증가해서 받았습니다. 지금 총 얼마를 가지고 있는지 확인하는 코드를 작성하세요
let money = 0;
let period = 5;
for(let j=1; j<=period; j++){
    money += j;
}
console.log(money);

let sum = 0;
let m = 1;
while(m <= 5){
    sum += m;
    m++;
}
console.log(sum);

let total = 0;
let k = 1;

do{
    total += k;
    k++;
}while(k <= 5);
console.log(total);

//while 문을 활용하여 5부터 0까지의 숫자를 넣어서 다음과 같이 출력되도록 하세요
//결과는 5,4,3,2,1,0와 같이 나옵니다
let l = 5;
let list = [];
while(l >= 0){
    list.push(l);
    l--;
}
console.log(`${list}`);

//teas라는 배열에는 green tea, black tea, chai, oolong tea가 있습니다
//chai 이전까지의 요소를 selectedTeas 라는 배열에 넣어 출력하세요 for문을 활용하세요
let teas = ["green tea", "black tea", "chai", "oolong tea"];
let selectedTeas = [];
for(let p=0; p<teas.length; p++){
    if(teas[p] === "chai"){
        continue; //break; chai 라는 요소를 만나면 진행하지 마세요
                  //continue; 요건에 맞으면 아래 실행을 하지 말고 건너 뛰세요
    }
    selectedTeas.push(teas[p]);
}
console.log(selectedTeas);

//cities라는 배열에는 London, New York, Paris, Berlin 이 있습니다 그 중 Paris와 London을 제외한 요소들을 visitedCities에 담으세요
let cities1 = ["London","New York", "Paris", "Berlin"]; 
let visitedCities = [];
for(let o=0; o<cities1.length; o++){
    if(cities1[o]=="Paris" || cities1[o]=="London"){
        continue;
    }
    visitedCities.push(cities1[o]);
}
console.log(visitedCities);