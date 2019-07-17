from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from II import IImodels

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


print(tf.VERSION)
print(tf.keras.__version__)



def airy_data_err(shape):
    runs, dx = shape
    err = np.random.random(runs)
    airx = np.linspace(0,200,dx)
    airxs = np.full(shape, airx)
    airy = IImodels.airy1D(airxs, 60)
    airy[np.isnan(airy)] = 1

    mod_errs = np.array([np.random.normal(0,scale=er, size=dx) for er in err])
    realistic_airy = airy + mod_errs
    labels = np.zeros(runs)
    labels[err < .5] = 1
    return realistic_airy, labels, err

data_shape = (5000,1000)

classifications = ["BadData","GoodData"]


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def trainor(dshape, model, epoch):
    td, tr, err = airy_data_err(dshape)
    model.fit(td, tr, epochs=epoch)

for i in range(0,50):
    trainor(data_shape, model, 2)
training_data, training_labels, err = airy_data_err(data_shape)
model.fit(training_data, training_labels, epochs=2)




test_data, test_labels, test_err = airy_data_err(data_shape)

test_loss, test_acc = model.evaluate(test_data, test_labels)
predictions = model.predict(test_data)
test_classifcations = np.argmax(predictions,axis=1)
asdf=234