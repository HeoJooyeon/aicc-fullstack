import numpy as np

# 일반 리스트와 넘파이 배열 비교
L = [1, 2, 3]
A = np.array([1, 2, 3])

# 리스트 요소 출력
for e in L:
    print(e)

# 리스트에 값 추가
L.append(4)

# 리스트에 새 리스트 더하기 (원본 리스트는 변하지 않음)
L + [5]  # 결과: [1, 2, 3, 4, 5] (출력 안 함)

# 넘파이 배열 연산
print(A + np.array([4]))         # 브로드캐스팅: [1+4, 2+4, 3+4]
print(A + np.array([4, 5, 6]))   # 동일 위치끼리 연산

# 정수 연산
print(2 + 4)         # 6
print(2 + A)         # 각 요소에 2 더함: [3, 4, 5]
print(2 * A)         # 각 요소에 2 곱함: [2, 4, 6]
print(2 * L)         # 리스트 복제: [1, 2, 3, 4, 1, 2, 3, 4]
print(L + L)         # 리스트 연결: [1, 2, 3, 4, 1, 2, 3, 4]

# 리스트 컴프리헨션 없이 새 리스트 만들기
L2 = []
for e in L:
    L2.append(e + 3)
print(L2)

# 리스트는 변하지 않음
print(L)

# 리스트 컴프리헨션
L3 = [i + 3 for i in L]
print(L3)

# 각 요소 제곱
L4 = [x ** 2 for x in L]
print(L4)

# [1, 4, 9, 16] 출력되도록 for문 사용
L4 = []
for i in L:
    L4.append(i ** 2)
print(L4)

# 넘파이 배열의 제곱
print(A ** 2)

# 원본 배열 확인
print(A)

# 제곱근, 로그, 지수, 하이퍼볼릭 탄젠트
print(np.sqrt(A))  # 제곱근
print(np.log(A))   # 자연 로그
print(np.exp(A))   # 지수
print(np.tanh(A))  # 하이퍼볼릭 탄젠트

# 2차원 리스트와 넘파이 배열 비교
L = [[1, 2], [3, 4]]
print(L)
print(L[0])        # 첫 번째 행
print(L[0][1])     # 첫 번째 행의 두 번째 열

# 2차원 넘파이 배열
A = np.array([[1, 2], [3, 4]])
print(A[0][1])     # 첫 번째 행, 두 번째 열
print(A[0, 1])     # 위와 동일한 결과
print(A[:, 0])     # 첫 번째 열 전체: [1, 3]

# 전치행렬 (Transpose): 행과 열 교환
print(A.T)
