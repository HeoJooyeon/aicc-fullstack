use shirts_db;
create table contacts (name varchar(100) not null, phone varchar(15) primary key);
desc contacts;
insert into contacts values ('billybob', '878123455');
select * from contacts;
create table contacts2 (name varchar(100) not null, phone varchar(15) not null unique);
desc contacts2;
insert into contacts2 values ('billybob', '878123455');
select * from contacts2;
create table user (user varchar(20) not null, age int check (age > 0));
desc user;
create table user2 (username varchar(20) not null, age int, constraint age_not_negative check (age >= 0));
desc user2;
alter table user2 drop constraint age_not_negative;
insert into user2 values('bily',-1);
select * from user2;
create table companies (name varchar(255) not null, address varchar(255) not null, constraint name_address unique (name, address));
create table houses (purchase_price int not null, sale_price int not null, constraint sprice_gt_pprice check(sale_price >= purchase_price));
desc houses;
alter table companies add column phone varchar(15);
alter table companies add column employee_count int not null default 1;
alter table companies drop column phone;
rename table companies to suppliers;
desc suppliers;
alter table suppliers rename to companies;
desc companies;
-- purchase_price는 0보다 크거나 같아야 하는 조건을 추가해주세요.
alter table houses add constraint pur_pri check(purchase_price >=0 );
alter table houses drop constraint pur_pri;
desc houses;
select * from houses;
create table customers (id int primary key auto_increment, first_name varchar(50), last_name varchar(50), email varchar(50));
desc customers;
create table orders ( id int primary key auto_increment, order_date Date, amount decimal(8,2), customer_id int, foreign key(customer_id) references customers(id));
desc orders;
INSERT INTO customers (first_name, last_name, email) 
VALUES ('Boy', 'George', 'george@gmail.com'),
       ('George', 'Michael', 'gm@gmail.com'),
       ('David', 'Bowie', 'david@gmail.com'),
       ('Blue', 'Steele', 'blue@gmail.com'),
       ('Bette', 'Davis', 'bette@aol.com');
INSERT INTO orders (order_date, amount, customer_id)
VALUES ('2016-02-10', 99.99, 1),
       ('2017-11-11', 35.50, 1),
       ('2014-12-12', 800.67, 2),
       ('2015-01-03', 12.50, 2),
       ('1999-04-11', 450.25, 5);
select * from customers;
select * from orders;
-- customer 1번이 구매한 내역 전체를 가져오세요.
select * from orders where customer_id = 1;
-- Daivs라는 last name을 가진 사람의 주문 데이터를 불러오세요.
select * from orders where customer_id = (select id from customers where last_name = 'Davis');
-- inner join
select * from customers join orders on orders.customer_id = customers.id;
select first_name, last_name, order_date, amount from customers join orders on orders.customer_id = customers.id;
select * from orders join customers on customers.id = orders.customer_id;
select first_name, last_name, sum(amount) as total 
from customers join orders on orders.customer_id = customers.id 
group by first_name, last_name 
order by total; 


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
select database();
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
replace
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
select author_lname, count(title) as books_written from books group by author_lname order by books_written desc;



use shirts_db;
create table contacts (name varchar(100) not null, phone varchar(15) primary key);
desc contacts;
insert into contacts values ('billybob', '878123455');
select * from contacts;
create table contacts2 (name varchar(100) not null, phone varchar(15) not null unique);
desc contacts2;
insert into contacts2 values ('billybob', '878123455');
select * from contacts2;
create table user (user varchar(20) not null, age int check (age > 0));
desc user;
create table user2 (username varchar(20) not null, age int, constraint age_not_negative check (age >= 0));
desc user2;
alter table user2 drop constraint age_not_negative;
insert into user2 values('bily',-1);
select * from user2;
create table companies (name varchar(255) not null, address varchar(255) not null, constraint name_address unique (name, address));
create table houses (purchase_price int not null, sale_price int not null, constraint sprice_gt_pprice check(sale_price >= purchase_price));
desc houses;
alter table companies add column phone varchar(15);
alter table companies add column employee_count int not null default 1;
alter table companies drop column phone;
rename table companies to suppliers;
desc suppliers;
alter table suppliers rename to companies;
desc companies;
-- purchase_price는 0보다 크거나 같아야 하는 조건을 추가해주세요.
alter table houses add constraint pur_pri check(purchase_price >=0 );
alter table houses drop constraint pur_pri;
desc houses;
select * from houses;
create table customers (id int primary key auto_increment, first_name varchar(50), last_name varchar(50), email varchar(50));
desc customers;
create table orders ( id int primary key auto_increment, order_date Date, amount decimal(8,2), customer_id int, foreign key(customer_id) references customers(id));
desc orders;
INSERT INTO customers (first_name, last_name, email) 
VALUES ('Boy', 'George', 'george@gmail.com'),
       ('George', 'Michael', 'gm@gmail.com'),
       ('David', 'Bowie', 'david@gmail.com'),
       ('Blue', 'Steele', 'blue@gmail.com'),
       ('Bette', 'Davis', 'bette@aol.com');
INSERT INTO orders (order_date, amount, customer_id)
VALUES ('2016-02-10', 99.99, 1),
       ('2017-11-11', 35.50, 1),
       ('2014-12-12', 800.67, 2),
       ('2015-01-03', 12.50, 2),
       ('1999-04-11', 450.25, 5);
select * from customers;
select * from orders;
-- customer 1번이 구매한 내역 전체를 가져오세요.
select * from orders where customer_id = 1;
-- Daivs라는 last name을 가진 사람의 주문 데이터를 불러오세요.
select * from orders where customer_id = (select id from customers where last_name = 'Davis');
-- inner join
select * from customers join orders on orders.customer_id = customers.id;
select first_name, last_name, order_date, amount from customers join orders on orders.customer_id = customers.id;
select * from orders join customers on customers.id = orders.customer_id;
select first_name, last_name, sum(amount) as total 
from customers join orders on orders.customer_id = customers.id 
group by first_name, last_name 
order by total; 















