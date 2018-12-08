import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# def conv1d(x, W, b, strides=1):

from timefreq.generator import SingleFreq

FS = 8000
SEED = 0
BATCH_SIZE = 64
N_SAMPLES = 1024

data_generator = SingleFreq(
    freq_range=(100, 3500),
    magnitude_range=(1.0, 1.0),
    random_state=SEED
)

data_input = []
data_output = []
for i in range(2000):
    signal, signal_params = data_generator.generate(N_SAMPLES, FS)
    freq, magnitude = signal_params

    data_input.append(signal)
    data_output.append(freq[0])

data_input = np.vstack(data_input).astype(np.float32)
data_output = np.vstack(data_output).astype(np.float32)

# X = tf.placeholder(tf.float32, shape=(BATCH_SIZE, N_SAMPLES))
data_input = tf.Variable(data_input)
data_output = tf.Variable(data_output)

data_input = tf.reshape(data_input, shape=(data_input.shape[0], data_input.shape[1], 1))
data_output = tf.reshape(data_output, shape=(data_output.shape[0], data_output.shape[1], 1))


conv1 = tf.layers.conv1d(data_input, 64, 50, 1, padding="same")#, activation=tf.nn.relu)
conv2 = tf.layers.conv1d(conv1, 32, 5, 1, padding="same")#, activation=tf.nn.relu)
conv3 = tf.layers.conv1d(conv2, 16, 5, 1, padding="same")#, activation=tf.nn.relu)
freq_computed = tf.layers.dense(conv3, 1)
# magnitude_computed = tf.layers.conv1d(conv3, 1, 25, 1)

true_freq = data_output  #tf.placeholder(tf.float32, shape=(BATCH_SIZE, N_SAMPLES))
error = tf.reduce_mean(tf.square(true_freq - freq_computed))


learning_rate = 0.0001
epochs = 3000
points = [[], []]

init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
tf.truncated_normal
with tf.Session() as sess:
    sess.run(init)

    points[0].append(0)
    points[1].append(sess.run(error))
    for i_epoch in range(epochs):
        sess.run(optimizer)

        print("Cost of epoch {}: {}".format(i_epoch, sess.run(error)))
        points[0].append(i_epoch+1)
        points[1].append(sess.run(error))
        if np.abs(1.0 - (points[1][-2] / points[1][-1])) > 0.03:
            print("decreasing learning rate : loss {} --> {}".format(points[1][-1], points[1][-2]))
            learning_rate /= 5
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
        if i_epoch > 100:
            if np.abs(1.0 - (points[1][-2] / points[1][-1])) < 0.0001:
                print("increasing learning rate : loss {} --> {}".format(points[1][-2], points[1][-1]))
                learning_rate *= 5
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
