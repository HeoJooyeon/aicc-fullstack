const express = require('express');
const bodyParser = require('body-parser');
const app = express(); //익스프레스 어플리케이션 객체 생성

const adminRoutes = require('./routes/admin');
const shopRoutes = require('./routes/shop');

app.use(bodyParser.urlencoded({extended: false}));
//body-parser는 POST로 보낸 데이터를 req.body에 담아서 저장하도록 도와준다.

app.use(adminRoutes);
app.use(shopRoutes);

app.listen(4000, ()=>{
    console.log(`Server is running on port 4000`);
});