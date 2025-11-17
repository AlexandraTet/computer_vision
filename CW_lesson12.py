import pandas as pd # математичні операції, таблиці
import numpy as np
import tensorflow as tf # для нейронок
from tensorflow import keras # бібліотека ТС
from tensorflow.keras import layers # додавання шарів
from sklearn.preprocessing import LabelEncoder # переведення назв в числа
import matplotlib.pyplot as plt # побудова графіків

# робота з csv файлом
df = pd.read_csv('data/figures.csv')
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

# 3. обрання елементів для навчання
X = df[['area', 'perimeter', 'corners']]
y = df["label_enc"]

# Визначення кількості класів та перетворення міток у формат OHE
num_classes = len(encoder.classes_)
y_ohe = keras.utils.to_categorical(y, num_classes=num_classes)

# 4. створення моделі
model = keras.Sequential([layers.Dense(8, activation = 'relu', input_shape = (3,)),
                          layers.Dense(8, activation = 'relu'),
                          # Розмір вихідного шару має дорівнювати кількості класів
                          layers.Dense(num_classes, activation = 'softmax')])

# 5. навчання
# Для softmax і OHE-міток використовуємо 'categorical_crossentropy'
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X, y_ohe, epochs = 200, verbose = 0) # Використовуємо OHE мітки

# 6. візуалізація навчання, графік
plt.plot(history.history['loss'], label = 'Loss') # втрати, помилки
plt.plot(history.history['accuracy'], label = 'Accuracy') # точність
plt.xlabel('Epoch')
plt.ylabel('Значення')
plt.title('Process of learning')
plt.legend()
plt.show()

# 7. тестування
# Вхідні дані для predict мають бути 2D-масивом
test = np.array([[18, 16, 0]])
pred = model.predict(test, verbose=0)
pred_class_index = np.argmax(pred)

print(f'Ймовірність по кожному класу: {pred}')
print(f'Модель визначила: {encoder.inverse_transform([pred_class_index])[0]}')
