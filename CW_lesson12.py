from pickletools import optimize

import pandas as pd # математичні операції, таблиці
import numpy as np
import tensorflow as tf # для нейронок
from tensorflow import keras # бібліотека ТС
from tensorflow.keras import layers # додавання шарів
from sklearn.preprocessing import LabelEncoder # переведення назв в числа
import matplotlib.pyplot as plt # побудова графіків
# робота з csv файлом
df = pd.read_csv('data/figures.csv')
encoder = LabelEncoder
df['label_enc'] = encoder.fit_transform(df['label'])

# 3. обрання елементів для навчання
X = df[['area', 'perimeter', 'corners']]
y = df["label_enc"]

# 4. створення моделі
model = keras.Sequential([layers.Dense(8, activation = 'relu', input_shape = (3,)),
                          layers.Dense(8, activation = 'relu'),
                          layers.Dense(8, activation = 'softmax')])
# 5. навчання
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(X, y, epochs = 200, verbose = 0)

# 6. візуалізація навчання, графік
plt.plot(history.history['loss'], label = 'Loss') # втрати, помилки
plt.plot(history.history['accuracy'], label = 'Accuracy') # точність
plt.xlabel('Epoch')
plt.ylabel('Значення')
plt.title('Process of learning')
plt.legend()
plt.show()

# 7. тестування
test = np.array([18, 16, 0])
pred = model.predict(test)
print(f'Ймовірність по кожному класу: {pred}')
print(f'Модель визначила: {encoder.inverse_transform([np.argmax(pred)])}')