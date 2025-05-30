DROP DATABASE IF EXISTS ig_clone;

CREATE DATABASE ig_clone;

USE ig_clone;

--data.sql 참조

-- 먼저 가입한 5명의 사용자를 조회
SELECT * FROM users
ORDER BY created_at
LIMIT 5;

-- 사용자별 사진 개수
SELECT user_id, COUNT(*) AS photo_count
FROM photos
GROUP BY user_id;

-- 사진별 댓글 수
SELECT photo_id, COUNT(*) AS comment_count
FROM comments
GROUP BY photo_id;

-- 사용자 가입일을 기준으로 요일별 가입자 수를 집계하는 쿼리
SELECT
DAYNAME(created_at) AS day,
COUNT(*) AS total
FROM users
GROUP BY day
ORDER BY total DESC
LIMIT 10;

-- 사용자와 그가 올린 사진의 URL을 함께 출력하세요.
SELECT users.username, photos. image_url
FROM users
JOIN photos ON users.id = photos.user_id;

-- 사용자와 그가 올린 사진, 그리고 URL을 함께 조회하세요.
SELECT users.username, photos.id AS photo_id, photos. image_url
FROM users
JOIN photos ON users.id = photos.user_id;

-- 사용자 전체와 해당 사용자가 올린 사진이 있다면 URL도 함께 보여주세요. (사진 없는 사용자도 포함)
SELECT users.username, photos. image_url
FROM users
LEFT JOIN photos ON users. id = photos.user_id;

-- 사진을 올리지 않은 사용자의 이름을 모두 가져오세요
SELECT users.username
FROM users
LEFT JOIN photos ON users. id = photos.user_id
WHERE photos.id IS NULL;

-- 좋아요(likes)를 누든 사용자 이름과 해당 사진 ID를 출력하시오.
SELECT users.username, likes.photo_id
FROM likes
JOIN users ON likes.user_id = users.id;

SELECT users.username, likes.photo_id
FROM users
JOIN likes ON likes.user_id = users.id;

-- 사용자 목록 전체와 그들 중 댓글을 단 경우가 있다면 댓글을 함께 출력하세요.
SELECT users.username, comments.comment text
FROM users
LEFT JOIN comments ON users.id = comments.user_id;

-- 좋아요를 받은 적이 없는 사진의 ID와 URL을 출력하세요.
SELECT photos.id, photos.image_url
FROM photos
LEFT JOIN likes ON photos.id = likes.photo_id
WHERE likes.photo_id IS NULL;

-- 좋아요를 두번 이상 받은 적이 있는 사진의 ID와 URL을 출력하세요.
SELECT photos.id AS photo_id, photos. image_url
FROM photos
JOIN likes ON photos.id = likes.photo_id
GROUP BY photos.id, photos.image_url
HAVING COUNT(*) >= 2;

-- 사용자 1인당 평균 사진 업로드 수를 계산해서 가져오세요
SELECT (SELECT Count(*) FROM photos) / (SELECT Count(*) FROM users) AS avg;

-- 2016년에 가입한 사용자 수
SELECT COUNT(*) AS count_2016
FROM users
WHERE YEAR(created_at) = 2016;

-- 가장 좋아요 많은 사진 TOP 3
SELECT photo_id, COUNT(*) AS like_count
FROM likes
GROUP BY photo_id
ORDER BY like_count DESC
LIMIT 3;

-- 사용자별 사진 개수를 출력하세요.
SELECT user_id, COUNT(*) AS photo_count
FROM photos
GROUP BY user_id;

-- 사진별 댓글 수를 출력하세요
SELECT photo_id, COUNT(*) AS comment_count
FROM comments
GROUP BY photo_id;

-- 댓글을 단 사용자와 댓글 내용을 출력하세요.
SELECT users.username, comments.comment_text
FROM users
JOIN comments ON users.id = comments.user_id;

-- 사진과 그 사진이 받은 댓글 내용을 출력하세요.
SELECT photos.id AS photo_id, comments.comment_text
FROM photos
JOIN comments ON photos.id = comments.photo_id;

-- 모든 사용자이름과 사용자들이 사용한 태그 목록을 출력하되, 사진을 올리지 않았거나 태그가 없는 사용자도 포함하세요.
SELECT users.username, tags.tag_name
FROM users
LEFT JOIN photos ON users.id = photos.user_id
LEFT JOIN photo_tags ON photos.id = photo_tags.photo_id
LEFT JOIN tags ON photo_tags.tag_id = tags.id;

-- 사용자별로 누른 좋아요 개수를 출력하세요.
SELECT users.username, COUNT(*) AS total_likes
FROM users
JOIN likes ON users.id = likes.user_id
GROUP BY users.id;

-- 좋아요를 한번도 누르지 않은 사용자를 출력해주세요
SELECT users.username
FROM users
LEFT JOIN likes ON users.id = likes.user_id
WHERE likes.user_id IS NULL;

-- 사진을 3장 이상 올린 사용자
SELECT users.username, COUNT(*) AS photos_count
FROM users
LEFT JOIN photos ON users.id = photos.user_id
GROUP BY users.id
HAVING COUNT(*) >= 3;

-- 사진을 3장 이상 올린 사용자
SELECT user_id, COUNT(*) AS photo_count
FROM photos
GROUP BY user_id
HAVING photo_count >= 3;

-- 가장 좋아요를 많이 받은 순서대로 사진 id 5개와 해당 사진을 올린 사용자 이름, 이미지 url, 종 좋아요 숫자를 줄력해주세요.
SELECT username, photos.id, photos. image_url, COUNT(*) AS total
FROM photos
INNER JOIN likes ON likes.photo_id = photos.id
INNER JOIN users ON photos.user_id = users.id
GROUP BY photos.id
ORDER BY total DESC
LIMIT 5;

-- 자기 사진에 스스로 좋아요 누른 경우를 출력해주세요
SELECT users.username, photos.id AS photo_id
FROM photos
JOIN likes ON photos.id = likes.photo_id
JOIN users ON photos.user_id = users.id
WHERE photos.user_id = likes.user_id;