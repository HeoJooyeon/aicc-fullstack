# cifar10 이라는 데이터셋이 있움 # 소형 컬러이미지데이터셋 32*32 RGB 60000
# test data와

# class_names = [
#     'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
#     'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
# ]

# 첫 번째 ~ 12개의 이미지를 아래처럼 출력해주세요

import tensorflow as tf
print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

import numpy as np
import matplotlib.pyplot as plt

cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

train_size = train_images.shape
test_size = test_images.shape
print("Train Data Size:", train_size)
print("Test Data Size", test_size)


plt.figure(figsize=(12, 8))
for i in range(12) :
  plt.subplot(3, 4, i+1)
  plt.imshow(train_images[i])
  plt.title(class_names[int(train_labels[i])])
  plt.axis('off')

plt.suptitle('CIFAR-10 Sample Images')
plt.show()

cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# y_train = np.expand_dims(y_train, -1)
# y_test = np.expand_dims(y_test, -1)

class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]


x_train, x_test = x_train / 255.0, x_test / 255.0

y_train

plt.figure(figsize=(12, 6))
for i in range(12) :
  plt.subplot(3, 4, i + 1)
  plt.imshow(x_train[i])
  plt.title(class_names[int(y_train[i])])
  plt.axis('off')

plt.tight_layout()
plt.show()

# 합성곡 신경망(CNN)을 구축하여 모델을 트레이닝하고, loss와 accurac를 그래프로 나타세요

import numpy as np
import matplotlib.pyplot as plt

cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

train_size = train_images.shape
test_size = test_images.shape
print("Train Data Size:", train_size)
print("Test Data Size", test_size)


plt.figure(figsize=(12, 8))
for i in range(12) :
  plt.subplot(3, 4, i+1)
  plt.imshow(train_images[i])
  plt.title(class_names[int(train_labels[i])])
  plt.axis('off')

plt.suptitle('CIFAR-10 Sample Images')
plt.show()

print(train_images.shape)
print(train_labels[:,0])
print(train_labels.shape)

i = Input(shape = train_images[0].shape)

type(i)

type(train_images)

type(train_images.shape)

train_images[0].shape

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0 # 정규화 : 0 ~ 255 --> 0 ~ 1 흑백
# relu : relu(x) = max(0, x) : 0보다 크면 x값 그대로 추출, 0보다 작으면 0으로 출력
y_train, y_test = y_train.flatten(), y_test.flatten() # 2차원 (50000, 1) => 1차원 벡터로 변경 (50000,)
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)

print(len(set(y_train)))

K = len(set(y_train))

i = Input(shape = x_train[0].shape) # 32×32×3
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)

x = Flatten()(x)
x = Dropout(0.5)(x) # 이렇게 해야 가장 정확하다
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

# loss, accuracy 그래프화
# confusion matrix 시각화

import numpy as np
import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='loss') # 트레이닝 한 거
plt.plot(r.history['val_loss'], label='val_loss') # 검증해가면서 한 것
plt.legend()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()

from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
  plt.show()

p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))

labels = '''airplane,
automobile,
bird,
cat,
deer,
dog,
frog,
horse,
ship,
truck.
'''.split()
print(labels)

misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i])
plt.title("True label: %s Predicted: %s" % (labels[y_test[i]], labels[p_test[i]]))


