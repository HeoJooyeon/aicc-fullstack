CREATE DATABASE soap_store; -- 데이터베이스 생성
DROP DATABASE soap_store;
SHOW databases;
CREATE DATABASE pets;
USE pets;
SELECT database();
CREATE DATABASE pet_shop;
USE pet_shop;
CREATE TABLE cats (name VARCHAR(50), age INT );
SHOW tables;
SHOW COLUMNS FROM cats;
DESC cats; -- describe 묘사하다
CREATE TABLE dogs (name VARCHAR(50), breed VARCHAR(50), age INT );
SHOW tables;
DESC dogs;
DROP TABLE cats;
CREATE TABLE pastries (name VARCHAR(50), quantity INT );
DESC pastries;
CREATE TABLE cats (name VARCHAR(50), age INT );
DESC cats;

show tables;
desc pastries;
drop table pastries;
insert into cats (name, age) VALUES ('Blue Steel',5);
select * from cats;
insert into cats (name, age) VALUES ('Jenkins',7), ('Beth',2), ('Meatball',5), ('Turkey',1), ('Potato Face',15);
create table people (first_name varchar(20), last_name varchar(20), age int);
insert into people (first_name, last_name, age) VALUES ('Tina','Belcher',13), ('Bob','Belcher',42), ('Linda','Belcher',45), ('Phillip','Frond',38), ('Calvin','Fischoeder',70);
select * from people;
INSERT INTO cats (name) VALUES ('Tods');
CREATE TABLE cat2 (name varchar(100) not null, age int not null);
desc cat2;
insert into cat2 (name) values ('Bildo');
create table cats3 (name varchar(20) default 'no name provided', age int default 99);
insert into cats3(age) values (13);
insert into cats3(name) values ('naaammmee');
select * from cats3;
create table cats4 (name varchar(20) not null default 'unnamed', age int not null default 99);
insert into cats4 (name, age) values (NULL, NULL);
create table unique_cats (cat_id int primary key, name varchar(100) not null, age int not null);
desc unique_cats;
insert into unique_cats (cat_id, name, age) values (1, 'bingo', 2);
insert into unique_cats (cat_id, name, age) values (1, 'mondo', 1); -- Error Code: 1062. Duplicate entry '1' for key 'unique_cats.PRIMARY'
insert into unique_cats (cat_id, name, age) values (2, 'bingo', 2), (3, 'bingo', 2), (4, 'bingo', 2), (999, 'bingo', 2);
select database(); 
select * from unique_cats;
create table unique_cats3 (cat_id int auto_increment, name varchar(100) not null, age int not null, primary key (cat_id));
desc unique_cats3;
insert into unique_cats3 (name, age) values ('Bingo', 3);
select * from unique_cats3;
create table employees (id int auto_increment, first_name varchar(100), last_name varchar(100), middle_name varchar(100), age int, current_status varchar(100) default 'employed', primary key (id));
insert into employees (first_name, last_name, age) values ('Dora', 'Smith', 58);
select * from employees;
create table employees2 (id int auto_increment primary key, first_name varchar(255) not null, last_name varchar(255) not null, middle_name varchar(255), age int not null, current_status varchar(255) not null default 'employed');
desc employees2;
drop table cats;
create table cats (cat_id int auto_increment primary key, name varchar(255), breed varchar(255), age int);
insert into cats(name, breed, age) values ('Ringo','Tabby',4),('Cindy','Maine Coon',10),('Dumbledore','Maine Coon',11),('Egg','Persian',4),('Misty','Tabby',13),('George Michael','Ragdoll',9),('Jackson','Sphynx',7);
select * from cats;
-- CRUD : Create / Read / Update / Delete
select * from cats;
select age from cats;
select name, breed from cats;
select * from cats where age = 4;
select * from cats where name ='Egg';
select cat_id from cats;
select name, breed from cats;
select name, age from cats where breed='Tabby';
select cat_id, age from cats where cat_id=age;
select * from cats where cat_id=age;
use pet_shop;
-- UPDATE 테이블명 SET WHERE
update cats set breed = 'Shorthair' where breed = 'Tabby';
select * from cats;
update cats set age = 14 where name = 'Misty';
select * from cats where name = 'Jackson';
select cat_id, name from cats;
update cats set name = 'Jack' where name = 'Jackson';
select * from cats where name = 'Jack';
update cats set breed = 'Tabby' where name = 'Ringo';
select * from cats where name = 'Ringo';
update cats set breed = 'British Shorthair' where name = 'Ringo';
select * from cats where name = 'Ringo';
select * from cats where breed = 'Maine Coon';
update cats set age = 12 where breed = 'Maine Coon';
select * from cats where breed = 'Maine Coon';
delete from cats where name = 'Egg';
delete from cats;
desc cats;
insert into cats(name, breed, age) values ('Ringo','Tabby',4),('Cindy','Maine Coon',10),('Dumbledore','Maine Coon',11),('Egg','Persian',4),('Misty','Tabby',13),('George Michael','Ragdoll',9),('Jackson','Sphynx',7);
delete from cats where age = 4;
delete from cats where cat_id = age;
select * from cats;
create database shirts_db;
use shirts_db;
create table shirts (shirt_id int not null auto_increment primary key, article varchar(50), color varchar(50), shirt_size varchar(5), last_worn int);
desc shirts;
insert into shirts (article, color, shirt_size, last_worn) values ('t-shirt', 'white', 'S', 10),('t-shirt', 'green', 'S', 200),('polo shirt', 'black', 'M', 10),('tank top', 'blue', 'S', 50),('t-shirt', 'pink', 'S', 0),('polo shirt', 'red', 'M', 5),('tank top', 'white', 'S', 200),('tank top', 'blue', 'M', 15),('polo shirt', 'purple', 'M', 50);
select * from shirts;
select article, color from shirts;
select * from shirts where shirt_size = 'M';
select article, color, shirt_size, last_worn from shirts where shirt_size = 'M';
update shirts set shirt_size = 'L' where article = 'polo shirt';
select article, shirt_size from shirts;
update shirts set last_worn = 0 where last_worn = 15;
select article, last_worn from shirts;
update shirts set shirt_size = 'XS', color = 'off white' where color = 'white';
select article, color, shirt_size from shirts;
delete from shirts where last_worn = 200; -- last_worn이 200인 데이터를 삭제하세요
delete from shirts where article = 'tank top'; -- tank top 데이터를 삭제하세요
select * from shirts;
create table books (book_id int not null auto_increment primary key, title varchar(100), author_fname varchar(100), author_lname varchar(100), released_year int, stock_quantity int, pages int);
desc books;
INSERT INTO books (title, author_fname, author_lname, released_year, stock_quantity, pages)
VALUES
('The Namesake', 'Jhumpa', 'Lahiri', 2003, 32, 291),
('Norse Mythology', 'Neil', 'Gaiman',2016, 43, 304),
('American Gods', 'Neil', 'Gaiman', 2001, 12, 465),
('Interpreter of Maladies', 'Jhumpa', 'Lahiri', 1996, 97, 198),
('A Hologram for the King: A Novel', 'Dave', 'Eggers', 2012, 154, 352),
('The Circle', 'Dave', 'Eggers', 2013, 26, 504),
('The Amazing Adventures of Kavalier & Clay', 'Michael', 'Chabon', 2000, 68, 634),
('Just Kids', 'Patti', 'Smith', 2010, 55, 304),
('A Heartbreaking Work of Staggering Genius', 'Dave', 'Eggers', 2001, 104, 437),
('Coraline', 'Neil', 'Gaiman', 2003, 100, 208),
('What We Talk About When We Talk About Love: Stories', 'Raymond', 'Carver', 1981, 23, 176),
("Where I'm Calling From: Selected Stories", 'Raymond', 'Carver', 1989, 12, 526),
('White Noise', 'Don', 'DeLillo', 1985, 49, 320),
('Cannery Row', 'John', 'Steinbeck', 1945, 95, 181),
('Oblivion: Stories', 'David', 'Foster Wallace', 2004, 172, 329),
('Consider the Lobster', 'David', 'Foster Wallace', 2005, 92, 343);
select * from books;
select CONCAT('pi', 'ckle');
select CONCAT(author_fname,' ', author_lname) as author_name from books;
select concat_ws('*', 'Math', 'Teacher');
select concat_ws('-',title,author_fname,author_lname) from books;
select substring('Hello World', 1, 4); -- 맨마지막: length
select substring('Hello World', 7);
select substring('Hello World', -3);
select CONCAT(substring(title, 1, 10), '...') as 'short title' from books;
-- replace
select replace('Hello World', 'Hell','%$#@');
select replace('Hello World', 'o', '7');
select replace('cheese bread coffee milk', ' ', ' and ');
select replace(title, 'e ', '3') from books;
select replace(title, ' ', '-') from books;
select reverse('Hello World');
select concat(author_fname, reverse(author_fname)) from books;
select char_length('Hello World');
select char_length(title) as 'length', title from books;
select author_lname, char_length(author_lname) as 'length' from books;
select concat(author_lname, ' is ', char_length(author_lname), ' characters long') from books;
select upper('Hello World');
select lower('Hello World');
select concat(upper('my favorite is '), upper(title)) from books;
select concat(upper('my favorite is '), lower(title)) from books;
select insert('Hello Bobby', 6, 0, 'There');
select left('omghahalol!',3);
select right('omghahalol!',4);
select repeat('ha', 4);
select trim('    pickle   ');
select replace(title, ' ', '->') as title from books;
select author_lname as 'forwards', reverse(author_lname) as 'backwards' from books;
select concat(upper(author_fname), ' ', upper(author_lname)) as 'full name in caps' from books;
select concat(title, ' was released in ', released_year) as 'YEAR' from books;
select title, char_length(title) as 'character_count' from books;
select concat(left(title, 10),'...') as 'short_title', concat(author_lname, ' , ', author_fname) as 'author', concat(stock_quantity, ' in stock') as 'quantity' from books;
INSERT INTO books (title, author_fname, author_lname, released_year, stock_quantity, pages)
VALUES 
('10% Happier', 'Dan', 'Harris', 2014, 29, 256), 
('fake_book', 'Freida', 'Harris', 2001, 287, 428),
('Lincoln In The Bardo', 'George', 'Saunders', 2017, 1000, 367);
select * from books;
select distinct author_lname from books;
select distinct concat(author_fname, ' ', author_lname) from books;

use shirts_db;
select * from books order by author_lname asc;
select * from books order by author_lname desc; -- Ascending: 오름차순 vs. descending: 역순, 내림차순
select * from books order by released_year;
select book_id, author_fname, author_lname, pages from books order by 2 desc;
select book_id, author_fname, author_lname, pages from books order by author_lname, author_fname;
select title from books limit 3; -- 상위 데이터에서 3개 노출
select * from books order by released_year desc limit 0, 5;
select * from books where author_fname like '%da%'; -- DA, da, wild card
select * from books where title like '%:%';
-- 책 제목 어디든지 %가 포함된 것들을 뽑아 오세요
select * from books where title like '%\%%';
-- 책 제목 어디든지 _가 포함된 것들을 뽑아 오세요
select * from books where title like '%\_%';
-- 책 제목 어디든지 stories 가 포함된 title을 뽑아 오세요
select * from books where title like '%stories%';
-- 내용이 가장 긴 책의 제목을 뽑아 오세요
select * from books order by pages desc limit 0, 1;
select concat(title, ' - ',released_year) as summary from books order by released_year desc limit 0, 3;
-- 작가의 lname 중에 어디든지 빈칸이 하나있는 경우에만 tilte과 lname을 출력하세요
select title, author_lname from books where author_lname like '% %';
-- 재고량이 가장 적은 순서대로 3권의 책의 제목, 출간년도, 재고량을 출력하세요
select title, released_year, stock_quantity from books order by stock_quantity limit 3;
-- title과 lname을 lname기준으로 오름차순으로 정리하시고, 같은 순위는 title을
select title, author_lname from books order by 2,1;
-- from books order by author_lname, title;
-- 작가의 lname을 기준으로 출력한 내용입니다. sql을 작성하세요.
select concat('MY FAVORITE AUTHOR IS ', upper(author_fname), ' ', upper(author_lname), '!') as 'yell' from books order by author_lname;
select count(*) from books;
-- lname의 갯수를 세 주세요
select count(author_lname) from books;
-- lname의 갯수에서 중복된 것을 빼고 세 주세요
select count(distinct author_lname) from books;
-- title에 어디든지 the가 들어간 것들의 수를 세 주세요
select count(title) from books where title like '%the%';
select author_lname, count(*) from books group by author_lname;

SELECT 
    author_lname, COUNT(*) AS books_written
FROM
    books
GROUP BY author_lname
ORDER BY books_written DESC;

SELECT MAX(pages) FROM books;
SELECT MAX(pages) FROM books;

-- lname 중 알파벳 순서로 가장 앞에 나오는 것을 출력하세요. 

SELECT MIN(author_lname) FROM books;

-- 페이지 수가 가장 많은 책의 이름을 위의 내용을 활용하여 출력하세요. (단톡참고)
SELECT title, pages FROM books WHERE pages = (SELECT MAX(pages) FROM books); 

-- 가장 오래 전에 출간된 책의 이름을 위의 내용을 활용하여 출력하세요. (단톡참고)
SELECT title, released_year FROM books WHERE released_year = (SELECT MIN(released_year) FROM books); 

SELECT author_fname, author_lname, COUNT(*) 
FROM books 
GROUP BY author_lname, author_fname;


SELECT CONCAT(author_fname, ' ', author_lname) AS author,  COUNT(*)
FROM books
GROUP BY author;

-- 작가(lname)들의 가장 최근 출간연도과 가장 오래전 출간연도를 출력하세요. (단톡참고) 

SELECT author_lname, MAX(released_year), MIN(released_year) FROM books GROUP BY author_lname;

-- 작가(lname)들의 총 출간저서 권수, 최근 발간연도, 가장 오래된 발간연도와 저서들 중 가장 많은 페이지수를 출력하세요. 
-- (단톡참고)  


SELECT 
   author_lname, 
    COUNT(*) as books_written, 
    MAX(released_year) AS latest_release,
    MIN(released_year)  AS earliest_release,
    MAX(pages) AS longest_page_count
FROM books GROUP BY author_lname; 
    

SELECT SUM(pages) FROM books;


SELECT author_lname, COUNT(*), SUM(pages) FROM books GROUP BY author_lname;

SELECT AVG(pages) FROM books;


SELECT 
    released_year, 
    AVG(stock_quantity), 
    COUNT(*) FROM books
GROUP BY released_year;

SELECT CONCAT(author_fname, ' ', author_lname) AS author, pages FROM books WHERE pages = (SELECT MAX(pages) FROM books);

SELECT CONCAT(author_fname, ' ', author_lname) AS author, pages FROM books ORDER BY pages DESC LIMIT 1;


SELECT 
    released_year AS year,
    COUNT(*) AS '# books',
    AVG(pages) AS 'avg pages'
FROM books
GROUP BY released_year
ORDER BY released_year;

SELECT * FROM books
WHERE released_year != 2017;

SELECT * FROM books
WHERE title NOT LIKE '%e%';


SELECT * FROM books
WHERE released_year > 2005;

SELECT * FROM books
WHERE pages > 500;

SELECT * FROM books
WHERE released_year <= 1985;


-- 2011년 이후에 출간된 책 중에 lname이 Eggers이고 title에 어느 부분이던지 novel이 포함된 경우, title, lname, released_year를 출력해주세요
SELECT title, author_lname, released_year FROM books 
WHERE released_year > 2010
AND author_lname = 'Eggers'
AND title LIKE '%novel%'; 


-- 페이지 수가 500을 초과하고, 제목이 30자를 초과하는 책제목과 페이지수를 출력하세요.
SELECT title, pages FROM books 
WHERE CHAR_LENGTH(title) > 30
AND pages > 500;

-- lname이 Eggers 이거나 출간연도가 2011년 이상인 책제목과 lname과 출간연도를 출력하세요
SELECT title, author_lname, released_year FROM books
WHERE author_lname='Eggers'
OR released_year > 2010;



SELECT title, released_year,
CASE
   WHEN released_year >= 2000 THEN 'modern lit'
    WHEN released_year < 2000 AND released_year > 1900 THEN 'mid_modern lit'
    ELSE '20th century lit' 
END AS genre
FROM books;


SELECT 
    title,
    stock_quantity,
    CASE
    WHEN stock_quantity >= 0 AND stock_quantity <= 40 THEN '*'
    WHEN stock_quantity BETWEEN 41 AND 70 THEN '**'
   WHEN stock_quantity BETWEEN 71 AND 100 THEN '***'
    WHEN stock_quantity BETWEEN 101 AND 140 THEN '****'
    ELSE '*****'
END AS stock
FROM
    books;





