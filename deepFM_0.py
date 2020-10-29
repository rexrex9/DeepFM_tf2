import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import readData
from tqdm import tqdm

batch_size=32
k=4

class FMLayer(layers.Layer):
    def __init__(self):
        super(FMLayer, self).__init__()

    def build(self, input_shape):
        super(FMLayer, self).build(input_shape)

        self.weight = self.add_weight(shape=(input_shape[1], k),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)

    def call(self, inputs):
        t=[]
        for ii in tqdm(range(batch_size)):
            X=inputs[ii]
            outputs = []
            for j1 in range(X.shape[0]):
                for j2 in range(j1+1,X.shape[0]):
                    w1=tf.cast([self.weight[j1]],tf.float32)
                    w2=tf.cast([self.weight[j2]],tf.float32)
                    w2=tf.transpose(w2)
                    out=(tf.matmul(w1,w2)*X[j1]*X[j2])[0][0]
                    outputs.append(out)
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

class DeepFMModel(tf.keras.Model):
    def __init__(self):
        super(DeepFMModel, self).__init__(name='DeepFMModel')
        self.FMLayer = FMLayer()
        self.deepPart = DeepPart()
        self.out_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        FM = self.FMLayer(inputs)
        deep = self.deepPart(inputs)
        out = self.out_layer(tf.concat([FM, deep], 1))
        return out



model = DeepFMModel()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['binary_crossentropy'])

X,Y = readData.readData()
model.fit(X, Y, batch_size=batch_size, epochs=5)


