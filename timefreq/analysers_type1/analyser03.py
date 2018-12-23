# network computing regression, calculating frequency of a sine with 0 phase shift, using fft

import numpy as np
import tensorflow as tf

SEED = 0
TRAIN_SIZE = 20000
DEV_SIZE = 5000
TEST_SIZE = 10000

DURATION = 0.125    # [s]
FS = 8000           # [Hz]
N_SAMPLES = int(DURATION * FS) + 1
MIN_FREQ = 20       # [Hz]
MAX_FREQ = 3700     # [Hz]
TWO_PI = 2 * np.pi

# network params
N_HIDDEN1 = 4096
N_HIDDEN2 = 1
N_HIDDEN3 = 1
N_CLASSES = 1
LEARNING_RATE = 0.0005
N_TRAINING_EPOCHS = 300
BATCH_SIZE = 50


def create_test_sample(random_state=None):
    freq = MIN_FREQ + (MAX_FREQ - MIN_FREQ) * random_state.rand()
    start_phase = random_state.rand() * TWO_PI
    signal_x = np.linspace(0, DURATION, N_SAMPLES)
    signal_y = np.sin(TWO_PI * freq * signal_x + start_phase)

    return signal_y, freq


if __name__ == '__main__':
    # CREATE DATABASE
    train_in = [];  train_out = [];  dev_in = [];  dev_out = [];  test_in = [];  test_out = []
    random_state = np.random.RandomState(SEED)

    for _ in range(TRAIN_SIZE):
        x, y = create_test_sample(random_state)
        train_in.append(x)
        train_out.append(y)

    for _ in range(DEV_SIZE):
        x, y = create_test_sample(random_state)
        dev_in.append(x)
        dev_out.append(y)

    for _ in range(TEST_SIZE):
        x, y = create_test_sample(random_state)
        test_in.append(x)
        test_out.append(y)

    # PREPARE NETWORK
    x_input_ph = tf.placeholder(tf.float32, [None, N_SAMPLES])
    y_output_ph = tf.placeholder(tf.float32, [None, 1])

    weights = {
        'w1': tf.Variable(tf.random_normal([N_SAMPLES, N_HIDDEN1], 0, 0.1)),
        'w2': tf.Variable(tf.random_normal([N_HIDDEN1, N_HIDDEN2], 0, 0.1)),
        'w3': tf.Variable(tf.random_normal([N_HIDDEN2, N_HIDDEN3], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([N_HIDDEN3, N_CLASSES], 0, 0.1)),
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([N_HIDDEN1], 0, 0.1)),
        'b2': tf.Variable(tf.random_normal([N_HIDDEN2], 0, 0.1)),
        'b3': tf.Variable(tf.random_normal([N_HIDDEN3], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([N_CLASSES], 0, 0.1)),
    }

    # pre-processing step calculating fft and concatenating it to signal
    fft = tf.abs(tf.fft(tf.cast(x_input_ph, tf.complex64))) / tf.constant(N_SAMPLES, dtype=tf.float32)

    fft_and_signal = tf.concat([x_input_ph, fft], -1)

    # first layer
    model = tf.add(tf.matmul(x_input_ph, weights['w1']), biases['b1'])
    model = tf.nn.relu(model)

    # second layer
    model = tf.add(tf.matmul(model, weights['w2']), biases['b2'])
    model = tf.nn.relu(model)

    # third layer
    model = tf.add(tf.matmul(model, weights['w3']), biases['b3'])
    model = tf.nn.relu(model)

    # output layer with no activation
    prediction = tf.abs(tf.add(tf.matmul(model, weights['out']), biases['out']))

    # cost and optimizer
    cost = tf.reduce_mean(tf.abs(tf.log(prediction/y_output_ph)))
    # cost = tf.reduce_mean(tf.abs(prediction - y_output_ph))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # TRAIN AND TEST NETWORK
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(N_TRAINING_EPOCHS):

            import time
            start_time = time.time()
            train_in = []
            train_out = []
            for _ in range(TRAIN_SIZE):
                x, y = create_test_sample(random_state)
                train_in.append(x)
                train_out.append(y)
            # print("PREPARING DATA TIME: {}".format(time.time() - start_time))

            start_time = time.time()
            cum_cost = 0.
            n_batches = int(TRAIN_SIZE / BATCH_SIZE)
            for i in range(n_batches):
                batch_in = train_in[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                batch_out = np.array(train_out[i*BATCH_SIZE:(i+1)*BATCH_SIZE]).reshape(-1,1)

                _, c, p = sess.run([optimizer, cost, prediction],
                                   feed_dict={
                                       x_input_ph: batch_in,
                                       y_output_ph: batch_out,
                                   })
                batch_error = c / BATCH_SIZE
                comparator = np.hstack([np.array(batch_out).reshape(-1, 1), p])
                cum_cost += c

            avg_cost = cum_cost / n_batches

            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))
            # compute of cost on development set
            if epoch % 10 == 0:
                batch_dev_in = dev_in
                batch_dev_out = np.array(dev_out).reshape(-1, 1)
                validation_cost, p, fft_and_signal_result = sess.run([cost, prediction, fft_and_signal],
                                              feed_dict={
                                                  x_input_ph: batch_dev_in,
                                                  y_output_ph: batch_dev_out
                                              })
                avg_cost = validation_cost
                print("Validation epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.9f}".format(avg_cost))
            stop = 1


            # print("EPOCH TIME: {}".format(time.time() - start_time))

        batch_dev_in = test_in
        batch_dev_out = np.array(test_out).reshape(-1, 1)
        test_cost, p = sess.run([cost, prediction],
                                feed_dict={
                                    x_input_ph: batch_dev_in,
                                    y_output_ph: batch_dev_out
                                })
        avg_cost = test_cost
        print("Test results:", '%04d' % (epoch + 1), "cost=", \
              "{:.9f}".format(avg_cost))