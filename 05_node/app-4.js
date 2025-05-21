const http = require('http');

const server = http.createServer((req, res) => {
    const url = req.url;
    if(url === '/'){
        res.setHeader('Content-Type', 'text/html');
        res.write('<html>');
        res.write('<head><title>Enter Message</title></head>');
        res.write(`
            <body>
                <form action='message' method='POST'>
                <input type='text' name='message'></input>
                <button type='submit'>send</button>
                </form>
            </body>
            `);
        res.write('</html>');
        return res.end();
    }
    // else{
    res.setHeader('Content-Type', 'text/html');
    res.write('<html>');
    res.write('<head><title>Message</title></head>');
    res.write(`
        <body>
            <h1>Hello From Node.js Server!</h1>
        </body>
        `);
    res.write('</html>');
    res.end();
    // }
});

server.listen(3003, () => {
    console.log("server start");
});