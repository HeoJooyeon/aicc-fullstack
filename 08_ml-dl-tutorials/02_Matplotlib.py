import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ----------------------------------
# 기본적인 함수 시각화
# ----------------------------------

x = np.linspace(0, 20, 1000)  # 0~20 사이를 1000등분한 x 값
print(x)

# y = x^2 그래프
y = x ** 2
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# y = x 직선 그래프
y = x
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# y = sin(x) 그래프
y = np.sin(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# y = sin(x) + 0.2x (선형 성분 추가)
y = np.sin(x) + x * 0.2
plt.plot(x, y)
plt.title("my plot")
plt.xlabel("input")
plt.ylabel("output")
plt.show()

# ----------------------------------
# 히스토그램 (정규분포 난수)
# ----------------------------------

x = np.random.randn(100, 2)  # 100개, 2차원 정규분포 데이터
print(x)

t = np.random.randn(10000)   # 1차원 정규분포 데이터 10000개
print(t)

# 밀도 기반 히스토그램
plt.hist(t, bins=50, density=True)
plt.title('Histogram of Random Normal Data')
plt.xlabel('Value')
plt.ylabel('Density')  # 밀도 기반
plt.grid(True)
plt.show()

# 일반 히스토그램 (빈도 기반)
plt.hist(t, bins=50)
plt.title('Histogram of Random Normal Data')
plt.xlabel('Value')
plt.ylabel('Frequency')  # 빈도 기반
plt.grid(True)
plt.show()

# ----------------------------------
# 산점도 (Scatter Plot)
# ----------------------------------

X = np.random.randn(100, 2)  # 100개의 2차원 정규분포
print(X)

# x축: 첫 번째 열, y축: 두 번째 열
plt.scatter(X[:, 0], X[:, 1])
plt.title("Scatter Plot of Random Points")
plt.show()

# 두 그룹으로 나눠서 다른 색상으로 시각화
X1 = X[:50]  # 앞 50개
X2 = X[50:]  # 뒤 50개
plt.scatter(X1[:, 0], X1[:, 1], color='yellow')
plt.scatter(X2[:, 0], X2[:, 1], color='purple')
plt.title("Two Colored Groups")
plt.show()

# 조건에 따라 색상을 나눔
X = np.random.randn(100, 2)
colors = []
for x, y in X:
    if x < 0 and y < 0:
        colors.append('purple')
    else:
        colors.append('yellow')

plt.scatter(X[:, 0], X[:, 1], color=colors)
plt.title("Conditional Colored Scatter")
plt.show()

# ----------------------------------
# 라벨에 따른 색상 시각화
# ----------------------------------

X = np.random.randn(200, 2)  # 2차원 정규분포 200개
X[:50] += 3  # 앞 50개 데이터를 (3, 3)만큼 이동
Y = np.zeros(200)  # 라벨 벡터 초기화
Y[:50] = 1  # 앞 50개만 라벨을 1로 설정

plt.scatter(X[:, 0], X[:, 1], c=Y)  # 라벨별로 색상 다르게 표시
plt.title("Clustered Labeled Points")
plt.show()

print(Y)
print(X)

# ----------------------------------
# 이미지 처리 (외부 이미지 다운로드 필요)
# ----------------------------------
# 터미널에서 수동으로 이미지 다운로드 필요
# 예: wget https://... 또는 브라우저로 직접 다운로드

# 이미지 열기
im = Image.open('image-26.png')
print(im)
print(type(im))

# 이미지 → numpy 배열로 변환
arr = np.array(im)
print(arr.shape)

# 또 다른 이미지 예시
im2 = Image.open('cotton-tulear-2422612_1280.jpg')
arr2 = np.array(im2)
print(arr2.shape)

# 이미지 출력
plt.imshow(arr2)
plt.title("Original Image")
plt.show()

# 이미지 평균값으로 흑백 변환
gray = arr2.mean(axis=2)  # RGB 평균 → 흑백 이미지
print(gray.shape)

# 흑백 이미지 출력
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.show()

# ----------------------------------
# 현재 디렉토리 내 파일 목록 출력
# ----------------------------------
files = os.listdir('.')
print("Current directory files:", files)
