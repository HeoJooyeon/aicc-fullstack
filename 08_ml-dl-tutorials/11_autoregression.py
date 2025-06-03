import tensorflow as tf
print(tf.__version__)

  # regression 회귀 : 방의 면적, 개수, 부동산 위치에 따른 집값, 월세, 전세 등
  # 이런 건 일반 회귀인 regression 이론으로 추출
  # 오늘의 날씨를 하려고 하는데 1년 전 오늘, 2년 전 오늘, 10년 전 오늘 .. 15년치 데이터를 모아놓고 예측하는 건
  # 이런 건 RNN 기반으로 분석
  # RNN (Recurrent Neural Network)
  # 두 가지 접근 방식이 다름 => 시간의 흐름이 있느냐 없느냐
  # 전자는 시간에 따르지 않고 언제 어디서나 output값이 튀어나와야 한다
  # RNN 기반 회기는 주식 데이터가 주로 그럼. 10년치를 모아놨어 다음주거를 예측해봐야지 한 달치 기상 정보를 가지고 다음 주 기상 예보. 시간의 흐름 기억
  # 시간의 흐름이 전혀 반영되지 않은 걸 데이터.. 일반적인 회귀
  # 시간의 흐름을 기반으로.. RNN..

  # 미묘한 차이들이 있움~
  # regression 회귀
  # Auto regression 회귀 + 시계열 분석 : 통계적 선형 모델 => 단기 시계열 예측에 적합
  # RNN  => 인공 신경망을 통한 비선형 모델 -> 장기, 복잡한 시게열 데이터 분석에 적합
  # = > 신경망 구조. 바로 직전의 데이터 분석 결과를 얘가 갖고 있음


  # 얘네 사이에 Auto regreesion
  # 시간의 흐름에 따른 데이터 흐름 분석을 중간에서 해줌~


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

series = np.sin(0.1*np.arange(200))
plt.plot(series)
plt.show()

# e.g. [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5] => 여기 10개를 받아와서 그 다음을 예측하게 하는 거

T = 10 # 10개의 데이터의 입력값
X = [] # 그 입력값에 X가 있으면,
Y = [] # Y가
for t in range(len(series) - T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1, 1)
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)


