# distinguish between 100 and 200 Hz


import numpy as np
import tensorflow as tf

SEED = 0
TRAIN_SIZE = 10000
DEV_SIZE = 1000
TEST_SIZE = 10000

DURATION = 1.0        # [s]
FS = 8000           # [Hz]
N_SAMPLES = int(DURATION * FS) + 1
MIN_FREQ = 20       # [Hz]
MAX_FREQ = 3700     # [Hz]
TWO_PI = 2 * np.pi;

# network params
N_HIDDEN1 = 1024
N_HIDDEN2 = 1024
N_HIDDEN3 = 1024
N_CLASSES = 1
LEARNING_RATE = 0.000001
N_TRAINING_EPOCHS = 100
BATCH_SIZE = 1000

def create_test_sample(random_state=None):
    # freq = MIN_FREQ + (MAX_FREQ - MIN_FREQ) * random_state.rand()
    if random_state.rand() > 0.5:
        freq = 100
    else:
        freq = 200
    signal_x = np.linspace(0, DURATION, N_SAMPLES)
    signal_y = np.sin(TWO_PI * freq * signal_x)
    return signal_y, freq


if __name__ == '__main__':
    # CREATE DATABASE
    train_in = [];  train_out = [];  dev_out = [];  dev_out = [];  test_in = [];  test_out = []
    random_state = np.random.RandomState(SEED)

    for _ in range(TRAIN_SIZE):
        x, y = create_test_sample(random_state)
        train_in.append(x)
        train_out.append(y)

    for _ in range(TRAIN_SIZE):
        x, y = create_test_sample(random_state)
        dev_out.append(x)
        dev_out.append(y)

    for _ in range(TRAIN_SIZE):
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
    # cost = tf.reduce_mean(tf.abs(tf.log(prediction/y_output_ph)))
    cost = tf.reduce_mean(tf.abs(prediction - y_output_ph))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # TRAIN AND TEST NETWORK
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(N_TRAINING_EPOCHS):
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
                comparator = np.hstack([np.array(batch_out).reshape(-1,1), p])
                cum_cost += c

            avg_cost = cum_cost / (n_batches * BATCH_SIZE)

            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))