import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print('train_images.shape', train_images.shape)
print('len train labels', len(train_labels))
print('test_images.shape', test_images.shape)
print('len test labels', len(test_labels))

print('reshaping BECAUSE OF CNN INPUT!')
train_images = train_images.reshape(-1,28, 28, 1)   #Reshape for CNN -  should work!!
test_images = test_images.reshape(-1,28, 28, 1)

print('train_images.shape', train_images.shape)
print('len train labels', len(train_labels))
print('test_images.shape', test_images.shape)
print('len test labels', len(test_labels))

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(filters=10, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print('model fitting')
model.fit(train_images, train_labels, epochs=10)

print('model evaluation')
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# predictions = model.predict(test_images)
#
# print('first prediction' , predictions[0])
# print('label of prediction', np.argmax(predictions[0]))
# print('test label 0', test_labels[0])
