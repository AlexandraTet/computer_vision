import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.python.layers.normalization import normalization

train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train', image_size=(128, 128), batch_size=30, label_mode='categorical')
test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test', image_size=(128, 128), batch_size=30, label_mode='categorical')

# image normalize from 0 to 1
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# model creating, building. the first layer, import data
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3))) # tensorflow standard; the simplest oznaky
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3))) # tensorflow standard; the simplest oznaky
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3))) # tensorflow standard; the simplest oznaky
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

# second layer, test and train
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=15, validation_data=test_ds)

# ocinka
test_lost, test_accuracy = model.evaluate(test_ds)
print('Test loss:', test_lost)
print('Test accuracy:', test_accuracy)

class_name = ['cars', 'cats', 'dogs']
img = image.load_img('images/cat.jpg', target_size=(128, 128))
img_array = image.img_to_array(img)

# normalization
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])

# show results
print(f'Ймовірність по класам: {predictions[0]}')
print(f'Модель визначила: {class_name[predicted_index]}')