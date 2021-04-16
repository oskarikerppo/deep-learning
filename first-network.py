#Remember to activate deep-learning-env

import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import tensorflow as tf
from tensorflow import keras
# Just comment these off if you don't want to see them
print("TF:    ", tf.__version__)
print("Keras: ", keras.__version__)
print(tf.config.list_physical_devices("GPU"))
print("Uses cuda: ", tf.test.is_built_with_cuda())

np.random.seed(42)
tf.random.set_seed(42)

fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print('Training data shape:', X_train_full.shape)
print('Training data dtype:', X_train_full.dtype)

print('Training label shape:', y_train_full.shape)
print('Training label dtype:', y_train_full.dtype)

print(y_train_full)

class_names = ["T-shirt/top", "Trousers", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print('Class of the first image is "', class_names[y_train_full[0]], '"')

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.

y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# remember to scale all data in a similar way
X_test = X_test / 255.

print('Training data shape  :', X_train.shape)
print('Validation data shape:', X_valid.shape)
print('Test data shape      :', X_test.shape)

plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()

n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))

for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]], fontsize=12)

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()

model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape = (28, 28)))

model.add(keras.layers.Dense(300, activation = keras.activations.relu))

# use your own naming
my_activation = keras.activations.relu
model.add(keras.layers.Dense(100, activation = my_activation))

model.add(keras.layers.Dense(10, activation = keras.activations.softmax))

# model.add(keras.layers.Dense(10, activation = "softmax"))

# This is useful when you create and throw away lots of models (consumes memory)
# See https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session

def free_memory():
  keras.backend.clear_session()

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(300, activation = keras.activations.relu),
    keras.layers.Dense(100, activation = keras.activations.relu),
    keras.layers.Dense(10,  activation = keras.activations.softmax)
])

def create_MLP():
  model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(300, activation = keras.activations.relu),
    keras.layers.Dense(100, activation = keras.activations.relu),
    keras.layers.Dense(10,  activation = keras.activations.softmax)
  ])
  return model

free_memory()
model = create_MLP()

model.summary()

keras.utils.plot_model(model, show_shapes=True)

type(model.layers)

print(*model.layers, sep='\n')

hidden1 = model.layers[1]
print(hidden1.name)

named_layer = model.get_layer(hidden1.name)
# test that we got the same layer as when indexing with 1
print(named_layer is hidden1)

h1_params = hidden1.get_weights()
print(type(h1_params))
print(len(h1_params))

# weights are the first element
h1_weights = h1_params[0]
print(type(h1_weights))
print('Weight matrix shape: ', h1_weights.shape, '\n')
print(h1_weights, '\n')
print('Weight at 0,0:', h1_weights[0,0])

# and biases are the second one
h1_biases = h1_params[1]
print(type(h1_biases))
print('Bias vector shape: ', h1_biases.shape, '\n')
print(h1_biases, '\n')
print('Bias at 0:', h1_biases[0])

# Links to API documentation
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD
# https://www.tensorflow.org/api_docs/python/tf/keras/metrics/sparse_categorical_accuracy

# Also these have their string names but not using them
my_loss = tf.keras.losses.sparse_categorical_crossentropy
my_sgd  = optimizer=keras.optimizers.SGD()
my_acc_metric = keras.metrics.sparse_categorical_accuracy

model.compile(loss = my_loss,
              optimizer = my_sgd,
              metrics=[my_acc_metric])

history = model.fit(X_train, y_train, epochs = 30,
                    validation_data=(X_valid, y_valid))