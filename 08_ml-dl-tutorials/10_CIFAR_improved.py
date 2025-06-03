import tensorflow as tf
print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)

K = len(set(y_train))
print("number of classes:", K)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"), # 사진의 좌우를 바꿔버린다.
        tf.keras.layers.RandomRotation(0.1), # 사진을 10% 변경
        tf.keras.layers.RandomZoom(0.1), # zoom in, out (10%를 가까이서 보거나 멀리서 보거나)
    ]
)

data_augmentation

print(x_test.shape)
print(y_test.shape)

sample = x_train[1]

fig, axes = plt.subplots(1, 6, figsize=(15, 3))
axes[0].imshow(sample)
axes[0].set_title("Original")
axes[0].axis("off")

for i in range(1, 6):
  aug_img = data_augmentation(tf.expand_dims(sample, 0))[0].numpy() # (32, 32, 3) -> (1, 32, 32, 3)
  axes[i].imshow(aug_img)
  axes[i].set_title(f"Aug {i}")
  axes[i].axis("off")

plt.tight_layout()
plt.show()

i = Input(shape = x_train[0].shape)
x = data_augmentation(i)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x) # training의 안정화와 빠른 수렴을 도움
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x) # 특정 맵의 크기를 2분의 1로 줄임

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)

model.evaluate(x_test, y_test)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(
    buffer_size=10000).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

r = model.fit(train_dataset, validation_data=test_dataset, epochs=10)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()

# 열 가지 데이터의 오브멘트,,? 프리딕트하는지 테스트

import matplotlib.pyplot as plt

preds = model.predict(x_test)
pred_labels = preds.argmax(axis=1)

class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    true = class_names[y_test[i]]
    pred = class_names[pred_labels[i]]
    plt.title(f"True: {true}\nPred: {pred}")
    plt.axis('off')
plt.tight_layout()
plt.show()


