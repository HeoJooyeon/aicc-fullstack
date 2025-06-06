const http = require('http');

const server = http.createServer((req, res) => {
    console.log(req.url, req.method, req.headers);
});

server.listen(3000, () => {
    console.log('server is running on http://localhost:3000');
});


// server is running on http://localhost:3000
// / GET {
//   host: 'localhost:3000',
//   connection: 'keep-alive',
//   'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
//   'sec-ch-ua-mobile': '?0',
//   'sec-ch-ua-platform': '"Windows"',
//   'upgrade-insecure-requests': '1',
//   'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
//   accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
//   'sec-fetch-site': 'none',
//   'sec-fetch-mode': 'navigate',
//   'sec-fetch-user': '?1',
//   'sec-fetch-dest': 'document',
//   'accept-encoding': 'gzip, deflate, br, zstd',
//   'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
// }