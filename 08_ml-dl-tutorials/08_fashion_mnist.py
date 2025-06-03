import tensorflow as tf
print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# 데이터를 나눠서 픽셀값을 0~1로 변환 후에 클래스 이름을 다음과 같이 정의하시고, 이미지를 출력해주세요.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(12, 6))
for i in range(15) :
  plt.subplot(3, 5, i + 1)
  plt.imshow(x_train[i], cmap='gray')
  plt.title(class_names[y_train[i]])
  plt.axis('off')

plt.tight_layout()
plt.show()

print('x_train.shape):', x_train.shape)

x_train.size

x_test.size

class_names

x_test.shape

# number of classes

print("number of classes:", len(class_names))

K = len(set(y_train))
print(K)

# Keras API 를 통해서 CNN (Convolutinal Neural Network) 모델을 만듦
# 합성곱 신경망
# 입력 -> 합성곱 계층 3개 -> Flatten -> Dense -> Droptout(과적합 방지) -> 출력 (softmax : 각각의 확률값)

x_train = np.expand_dims(x_train, -1) # 28×28 => 2D, 28×28×1 => 3D, (28, 28, 1) / 2차원을 3차원으로 ~
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)
# (60000, 28, 28, 1)

# 입력 데이터
i = Input(shape = x_train[0].shape)

# 합성곱 계층 3개 │ node가 많으면 많을수록 더 자세하게 나옴
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i) # convolutional layer 1번째
# 32개의 node(filter)가 사용. 각각의 node는 3×3 / stirdes : ndoe가 2개씩 움직임 / 비선형 데이터에 가장 많이 쓰이는 relu
# 그림을 인식해서 그 그림의 형태를 잡아가는 거
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
# activationn 함수를 통해서 더 자세하게 훑고 지나가겠구나

x = Flatten()(x) # 1D 벡터화
x = Dropout(0.2)(x) # 과적합 방지
x = Dense(512, activation='relu')(x) # 가장 많은 필터로 쭉 둘러보는 것이다..
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

i

I

# loss, accuracy 그래프를 그리세요
# 3차원 이미지 입력 -> 특징 추출(3개의 convolutional layer) -> 차원 축소 -> 과적합 방지 2번 -> 출력 : 다중 클래스 분류 (softmax)

import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['accuracy'], label='accuracy')
plt.legend()

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()

from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

def plot_confusiont_matrix(cm, classes,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
  print('Confusion matrix, without normalization')
  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
  plt.show()

p_test = model.predict(x_test).argmax(axis=1) # [0.9, 0.2, ....] -> 0
cm = confusion_matrix(y_test, p_test)
plot_confusiont_matrix(cm, list(range(10)))


def plot_confusiont_matrix(cm, classes,
                           normalize='False',
                           title='Fashion Confusion Matrix',
                           cmap=plt.cm.Blues):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = 'd'
  thresh = cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
  plt.show()

p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusiont_matrix(cm, list(range(10)))

print(cm)

misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i].reshape(28,28), cmap='gray')
plt.title("True label: %s Predicted: %s" % (labels[y_test[i]], labels[p_test[i]]))

labels = '''T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot'''.split("\n")
print(labels)


