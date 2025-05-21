const express = require('express'); //routing

const app = express();

app.get('/', (req, res) => {
    res.send('홈페이지');
});


app.get('/about', (req, res) => {
    res.send('소개 페이지');
});

app.listen(3002, () =>{
    console.log("3002에서 서버 실행 중입니다");
});
