import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import readData
from tqdm import tqdm

batch_size=32

class WideLayer(layers.Layer):
    def __init__(self):
        super(WideLayer, self).__init__()

    def build(self, input_shape):
        super(WideLayer, self).build(input_shape)

        self.weight = self.add_weight(shape=(int((input_shape[1] * (input_shape[1] - 1)) / 2), 1),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)

    def call(self, inputs):
        t=[]
        for ii in tqdm(range(batch_size)):
            X=inputs[ii]
            i=0
            outputs = []
            for j1 in range(X.shape[0]):
                for j2 in range(j1+1,X.shape[0]):
                    out=(self.weight[i]*X[j1]*X[j2])[0]
                    outputs.append(out)
                    i+=1
            t.append(tf.cast(outputs,tf.float32))
        return tf.cast(t,tf.float32)


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
model.fit(X, Y, batch_size=batch_size, epochs=5)


