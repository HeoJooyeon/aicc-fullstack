const express = require('express');
const router = express.Router();

/* router.get('/product', (req, res, next) => {
    const message = req.url.split("=")[1];
    res.send(`
        <h1>data, ${decodeURIComponent(message)}</h1>
    `);
});
 */

router.get('/', (req, res, next) => {
    res.send(`
        <h1>hello, express</h1>
    `);
});

module.exports = router;