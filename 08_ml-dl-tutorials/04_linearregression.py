import tensorflow as tf
print(tf.__version__)

# 2.18.0

from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('moore.csv', header=None).values

data


plt.scatter(data[:,0], data[:,1])

X = data[:,0]
Y = data[:,1]
plt.scatter(X, Y)

Y = np.log(Y)
plt.scatter(X, Y)
# 한 번만 더 하면 로그 함수가 나옴

# keras : 텐서플로우를 엔진으로 하는 딥러닝 신경망 API
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)), # 연도 입력 (1개의 숫자)
    tf.keras.layers.Dense(1) # 한 개의 값이 출력
    ])

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')
# compile 학습 시작 전 모델 설정
# optimizer 가중치 업데이트 공식
# SGD : 선형 모델 포함 다양한 모델 최적화에 사용되는 기본 알고리즘 Stochastic Gradient Descent
# learning_rate = 0.001 조금씩 가중치를 업데이트. 0.01 (빠름) / 0.00001 (느림)
# momentum = 0.9 기존에 있었던 정보를 90% 신뢰를 하며 이 다음에 계산될 내용을 예측해가겠다
# loss : 오차 범위 (처음에는 상당히 크다)
# Mean Squared Error : 손실 함수

def schedule(epoch,lr):
  if epoch >= 50: # epoch 학습하는 횟수 (한 바퀴 도는 느낌)
    return 0.0001
  return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

r = model.fit(X, Y, epochs=200, callbacks=[scheduler]) # 실행문

# 이걸 먼저 붙이면,,
X = X - X.mean()

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
    ])

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')

def schedule(epoch,lr):
  if epoch >= 50:
    return 0.0001
  return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

r = model.fit(X, Y, epochs=200, callbacks=[scheduler])

# epoch가 최소 25번 / 긍께 25번은 돌아야 거의 정확해진다
plt.plot(r.history['loss'], label='loss') # 예측이 정확해짐

a = model.layers[0].get_weights()[0][0,0]

# np.float32(0.33221465)

Yhat = model.predict(X).flatten()
plt.scatter(X, Y)
plt.plot(X, Yhat)


