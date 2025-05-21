const http = require('http');
const fs = require('fs');
const path = require('path');
const port = 4000;

const server = http.createServer((req,res)=>{
    /* const url = req.url;
    if(url === '/' || url ==='/index.html'){
        const dataBuffer = fs.readFileSync("./index.html");
        const html = dataBuffer.toString();
        res.write(html);
        return res.end();
    }else if(url === '/about.html'){
        const dataBuffer = fs.readFileSync("./about.html");
        const html = dataBuffer.toString();
        res.write(html);
        return res.end();
    }else if(url === '/contact.html'){
        const dataBuffer = fs.readFileSync("./contact.html");
        const html = dataBuffer.toString();
        res.write(html);
        return res.end();
    } */
   const filePath = path.join(
        __dirname,
        req.url === "/" ? "index.html" : req.url
   );

   console.log(`dirname:: ${__dirname}`);
   console.log(`filePath:: ${filePath}`);

   const extName = path.extname(filePath).toLowerCase();

   let mimeType = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "text/javascript",
    ".gif": "image/gif",
    ".png": "image/png",
    ".mp4": "video/mp4"
   };

   const contentType = mimeType[extName] || "application/octet-stream";

   fs.readFile(filePath, (err, content)=>{
        if(err){
            if(err.code === "ENOENT"){
                fs.readFile(path.join(__dirname, "404.html"), (err, errorContent) => {
                    res.writeHead(404, {"Content-Type": "text/html"});
                    // res.end("404: File not found");
                    res.end(errorContent);
                });
            }else{
                res.writeHead(500);
                res.end(`Server Error: ${err.code}`);
            }
        }else{
            res.writeHead(200, {"Content-Type": contentType});
            res.end(content);
        }
   });
});
server.listen(port, () => {
    console.log(`Server is listening on port ${port}`);
});