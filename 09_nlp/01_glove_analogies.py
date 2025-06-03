import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import os

def dist1(a,b): # 유클리드 거리: 직선 거리
    return np.linalg.norm(a - b) # linear algebra norm (두 점의 직선 거리 구하는 공식)
# a(4,6), b(1,2) => ()

def dist2(a,b): # 코사인 거리
    return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


#| 각도(θ) | 코사인 유사도 (cos θ) | 코사인 거리 (1 - cos θ) | 해석                |
#| ----- | --------------- | ------------------ | ----------------- |
#| 0°    | 1               | 0                  | 완전 동일 방향, 완전 가까움  |
#| 90°   | 0               | 1                  | 서로 직각, 완전 다름(독립적) |
#| 180°  | -1              | 2                  | 완전 반대 방향, 완전 다름   |

# a(1,0), b(10,0), c(0,1) 
# 거리는 0~2 / 유사도는 방향
# 90도 각도 => 코사인 거리 1 (완전 다름), 코사인 유사도 0 (직각)
# 0도 각도 => 코사인 거리 0 (같은 방향), 코사인 유사도 1 (같은 방향)

dist, metric = dist2, "cosine" # 관계도 혹은 단어간의 상관 관계를 나타낼 때 사용 -> 절대 거리 대신 코사인 거리 사용

def find_analogies(w1, w2, w3): # (king, man, woman)
    for w in (w1, w2, w3):
        if w not in word2vec:
            print(f"{w} not in dictionary")
            return
    
    v0 = word2vec[w1] - word2vec[w2] + word2vec[w3]
    # king - man + woman = queen 왕에서 남성적인 특징을 제외하고 여성적인 특징을 더하면 여왕이 된다
    # king - man = queen - woman 왕에서 남성적인 특징을 제외하면 여왕에서 여성적인 특징을 제외한 것과 같다 
    # South Korea - Seoul + Tokyo vs. Japan 한국에서 서울을 제외하고 도쿄를 더하면 일본이 된다
    # South Korea - Seoul vs. Japan - Tokyo
    
    distances = pairwise_distances(v0.reshape(1, D), embedding, metric=metric) # 단어 2차원으로 변환, 벡터로 변환, 변수로 선언한 거리로 계산
    idxs = distances.argsort()[0] # 거리가 가까운 순으로 정렬
    
    for idx in idxs:
        word = idx2word[idx]
        if word not in (w1, w2, w3): # word2vec에 없는 단어 즉, v0에 담긴 단어
            print(f"{w1} - {w2} = {word} - {w3}")
            break

def nearest_neighbors(w, n=5):
    if w not in word2vec:
        print(f"{w} not in dictionary")
        return

    v = word2vec[w]
    distances = pairwise_distances(v.reshape(1, D), embedding, metric=metric)
    idxs = distances.argsort()[0]  # 가까운 순으로 정렬된 인덱스

    print(f"Top {n} nearest neighbors for '{w}':")
    count = 0
    for idx in idxs:
        neighbor = idx2word[idx]
        if neighbor != w:
            print(f"{neighbor} ({distances[0][idx]:.4f})")
            count += 1
        if count >= n:
            break

print("Loading word vectors...")
word2vec = {}
embedding = [] # 벡터 값의 list
idx2word = []

glove_path = os.path.join(os.path.dirname(__file__), "glove.6B.50d.txt")

with open(glove_path, encoding="utf-8") as f:
    for line in f: # file의 한줄씩
        values = line.split()
        word = values[0] # line의 맨 앞 즉, 단어
        vec = np.asarray(values[1:], dtype="float32") # 벡터 값을 float32로 변환 -> 엔진 소모 방지 및 속도 증가 목적 (굳이 안해도 됨)
        word2vec[word] = vec 
        embedding.append(vec)
        idx2word.append(word)
        
embedding = np.array(embedding)
V, D = embedding.shape # V: 단어 40만 개, D: 50차원
print(f"Found {V} word vectors.")

find_analogies("king", "man", "woman")
find_analogies("france", "paris", "rome")
find_analogies("china", "rice", "bread")
find_analogies("man", "woman", "mother")

nearest_neighbors("king")
nearest_neighbors("france")
nearest_neighbors("rome")