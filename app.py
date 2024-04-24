import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils

(x_train, y_train),( x_test, y_test) = fashion_mnist.load_data()

class_names = ["T-short/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker","Bag","Ankle boot"]

#Просмотр Картинки
'''
plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show() 
'''
# Нормализация Картинки
x_train = x_train / 255
x_test = x_test / 255
'''
plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show() 
'''
# Просмотр нескольких картинок
'''
plt.figure(figsize=(10,10))
for i in range (25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i]])
plt.show()
'''

#Создание модели 
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
'''
#Компиляция модели 
'''
model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()
'''
# Обучение модели

'''
history = model.fit(x_train, y_train, epochs=10)

model.save("my_model.keras")
'''
# проверка модели
'''
test_loss, test_acc = model.evaluate(x_test, y_test) 
print('Test accuracy:', test_acc)
'''

# Предсказываем картирнку
index = 5

model = keras.models.load_model("my_model.keras")
predictions = model.predict(x_train)
predicted_class_index = np.argmax(predictions[index])

print("Класс который был:",class_names[y_train[index]])
print("Предсказанный класс для первого элемента:", class_names[predicted_class_index])
