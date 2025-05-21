const http = require('http');
const fs = require('fs');

const server = http.createServer((req, res) => {
    const url = req.url;
    const method = req.method;
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
    if (url === '/message' && method === "POST"){
        const body = [];
        req.on('data', chunk => {
            console.log(chunk);
            body.push(chunk);
        }); //ASCII
        req.on('end', ()=>{
           const parsedBody = Buffer.concat(body).toString();
           console.log(`parsedBody:: ${parsedBody}`);
           const message = parsedBody.split('=')[1];
           console.log(`message:: ${message}`);
           fs.writeFileSync('./data/msessage-5.txt', message);
        });
        return res.end();
    }
    fs.writeFileSync('./data/msessage.txt','message sent');
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
});

server.listen(3004);