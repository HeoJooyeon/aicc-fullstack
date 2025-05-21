const express = require('express');
const router = express.Router();

/* router.get('/', (req, res, next) => {
    res.send(`
        <h1>hello, express</h1>
    `);
}); */

router.get('/add-product', (req, res, next) => {
/*     res.send(`
       <form action='product' method='GET'>
       <input type='text' name='message'></input>
       <button type='submit'>Add Product</button>
       </form>
    `); */
    res.send(`
        <form action='product' method='POST'>
        <input type='text' name='title' />
        <button type='submit'>Add Product</button>
        </form>
     `);
});

router.post('/product', (req, res, next) => {
    /* const message = req.url.split("=")[1];
    res.send(`
        <h1>data, ${decodeURIComponent(message)}</h1>
    `); */
    //console.log(req.body);
    //res.redirect('/');
    const productTilte = req.body.title;
    res.send(`
        <h1>Product Added:: ${decodeURIComponent(productTilte)}</h1>
    `);
});

router.get('/product', (req, res, next) => {
    res.send(`
        <h1>Product Not Found</h1>
    `);
});


module.exports = router;