import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = tf.keras.datasets.mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = tf.keras.utils.normalize(xtrain, axis=1)
xtest = tf.keras.utils.normalize(xtest, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

model.fit(xtrain, ytrain, epochs=2)

loss, accuracy = model.evaluate(xtest, ytest)
print(accuracy)
print(loss)

model.save('digits.model')

for x in range(2, 3):
    img = cv.imread(str(x) + '.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print("the result is probably :", np.argmax(prediction))
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
