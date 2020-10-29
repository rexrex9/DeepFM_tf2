import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import readData
import sys

class WideLayer(layers.Layer):
    def __init__(self):
        super(WideLayer, self).__init__()

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[1], 1),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)
        super(WideLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs,self.weight)


class DeepPart(tf.keras.Model):
    def __init__(self):
        super(DeepPart, self).__init__(name='DeepPart')
        self.hidden1 = layers.Dense(128, activation='relu')
        self.hidden2 = layers.Dense(128, activation='relu')

    def call(self, inputs):
        h1 = self.hidden1(inputs)
        h2 = self.hidden2(h1)
        return h2

class WideAndDeepModel(tf.keras.Model):
    def __init__(self):
        super(WideAndDeepModel, self).__init__(name='WideAndDeepModel')
        self.wideLayer = WideLayer()
        self.deepPart = DeepPart()
        self.out_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        wide = self.wideLayer(inputs)
        deep = self.deepPart(inputs)
        out = self.out_layer(tf.concat([wide, deep], 1))
        return out



model = WideAndDeepModel()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['binary_crossentropy'])

X,Y = readData.readData()

model.fit(X, Y, batch_size=16, epochs=5)