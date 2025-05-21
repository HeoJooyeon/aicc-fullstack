/*  const rand = (min, max) => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
} */
import reactImg from "../assets/Penguins.png";

const reactDescriptions =['Fundamental', 'Crucial', 'Core'];
//reactDescriptions[0]: 0~2
//reactDescriptions[getRandomInt(2)]
function getRandomInt(max){
    return Math.floor(Math.random() * (max + 1));
  }
//0~max 까지의 숫자(정수)를 랜덤 추출해주는 함수
//Header() 
//built-in component
//custom component
//명칭 crashing 방지
  export default function Header() { //import Header
    // export function Header ==> import {Header}
    const description = reactDescriptions[getRandomInt(2)];
    return(
      <header>
        <img src={reactImg} width="300px"></img>
        <h1>REACT BASICS</h1>
        <p>{description} React Concepts : 자바스크립트 라이브러리의 하나로서 사용자 인터페이스를 만들기 위한 웹 프레임워크</p>
      </header>
    );
  }

