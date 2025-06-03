import tensorflow as tf
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
type(data)

data.data.shape

data.target

data.target.shape

# 데이터를 테스트용 33%와 나머지는 트레인용으로 나누세요.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
N, D = X_train.shape # (학습 데이터의 샘플 수, 항목(컬럼))

from sklearn.preprocessing import StandardScaler
# 정규화, 전처리

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


