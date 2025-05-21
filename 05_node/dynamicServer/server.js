const http = require('http');
const port = 4000;

const server = http.createServer((req, res) => {
    res.writeHead(200, {"Content-Type": "text/html; charset=utf-8"});
    if(req.url == "/"){
        res.end(`<h1>Welcome to the Home Page</h1>`);
    }else if(req.url === "/time"){
        const currentTime = new Date().toLocaleDateString();
        res.end(`<h1>Current Time:: ${currentTime}</h1>`);
    }else if(req.url.startsWith("/greet?name=")){
        //콘솔에 입력한 이름이 뜨도록 해 보세요.
        /* res.write('<html>');
        res.write('<head><title>Enter Message</title></head>');
        res.write(`
            <body>
                <form action='message' method='POST' onsubmit="console.log(message.value);return false">
                <input type='text' name='message'></input>
                <button type='submit'>send</button>
                </form>
            </body>
            `);
        res.write('</html>');
        res.end(); */
        const name = req.url.split("=")[1];
        console.log(name);
        res.end(`<h1>Hello, ${decodeURIComponent(name)}</h1>`);

        
    }else{
        res.end(`<h1>404 Not Found</h1>`);
    }
});

server.listen(port, ()=> {
    console.log(`Server running at http://localhost:${port}`);
});