# network computing regression, calculating frequency of a sine with 0 phase shift, using fft
# regression a la classification

import numpy as np
import tensorflow as tf

NOISE_RATIO = 0.001

DURATION = 0.125   # [s]
FS = 8000           # [Hz]
N_SAMPLES = int(DURATION * FS) + 1
MIN_FREQ = 20       # [Hz]
MAX_FREQ = 3700     # [Hz]
TWO_PI = 2 * np.pi


A4_FREQ = 440
log_min_interval_1_32_semitone = 1.0 / (12 * 64)  # fixme  4 -> 32
log2_a4_freq = np.log2(A4_FREQ)
log2_min_freq = np.log2(MIN_FREQ)
log2_max_freq = np.log2(MAX_FREQ)
min_log_freq = log2_a4_freq \
               - int((log2_a4_freq - log2_min_freq) / log_min_interval_1_32_semitone) * log_min_interval_1_32_semitone
max_log_freq = log2_a4_freq \
               + int((log2_max_freq - log2_a4_freq) / log_min_interval_1_32_semitone) * log_min_interval_1_32_semitone
n_tones = (max_log_freq - min_log_freq) / log_min_interval_1_32_semitone
n_tones = int(n_tones) + 1
FREQS = np.power(2.0, np.linspace(min_log_freq, max_log_freq, n_tones))


SEED = 0
TRAIN_SIZE = 20000
DEV_SIZE = 20000
TEST_SIZE = 10000


# network params
INC = 0.125#1.0
N_HIDDEN1_A = int(10 * INC)
N_HIDDEN2_A = int(10 * INC)
N_HIDDEN3_A = int(10 * INC)
N_HIDDEN1_B = int(1024 * INC)
N_HIDDEN2_B = int(256 * INC)
N_HIDDEN3_B = int(256 * INC)
N_HIDDEN4 = int(1024 * INC)
N_HIDDEN5 = int(4096 * INC)

N_CLASSES = len(FREQS)

LEARNING_RATE = 0.001
N_TRAINING_EPOCHS = 300
BATCH_SIZE = 50

stop = 1


def create_test_sample(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    min_log_freq = np.log2(MIN_FREQ + 1)
    max_log_freq = np.log2(MAX_FREQ - 100)
    log_freq = min_log_freq + (max_log_freq - min_log_freq) * random_state.rand()
    freq = np.power(2.0, log_freq)

    start_phase = random_state.rand() * TWO_PI
    signal_x = np.linspace(0, DURATION, N_SAMPLES)
    signal_y = np.sin(TWO_PI * freq * signal_x + start_phase)

    return signal_y, freq


def create_train_sample(random_state=None, noise_ratio=None):
    if random_state is None:
        random_state = np.random.RandomState()

    freq_idx = np.random.randint(2, N_CLASSES - 2)
    freq = FREQS[freq_idx]
    if noise_ratio is not None:
        freq = freq - freq * noise_ratio / 2.0 + random_state.rand() * freq * noise_ratio

    start_phase = random_state.rand() * TWO_PI
    signal_x = np.linspace(0, DURATION, N_SAMPLES)
    signal_y = np.sin(TWO_PI * freq * signal_x + start_phase)

    return signal_y, freq, freq_idx


if __name__ == '__main__':
    # CREATE DATABASE
    random_state = np.random.RandomState(SEED)

    dev_in = []; dev_true_freq = []
    for _ in range(DEV_SIZE):
        signal_y, freq = create_test_sample(random_state)
        dev_in.append(signal_y)
        dev_true_freq.append(freq)

    # test_in = [];  test_out = []; test_true_freq = []
    # for _ in range(TEST_SIZE):
    #     signal_y, freq, net_result = create_train_sample(random_state)
    #     test_in.append(signal_y)
    #     test_out.append(net_result)
    #     test_true_freq.append(freq)

    # PREPARE NETWORK
    x_input_ph = tf.placeholder(tf.float32, [None, N_SAMPLES])
    # y_output_ph = tf.placeholder(tf.float32, [None, 1])

    weights = {
        'w1A': tf.Variable(tf.random_normal([N_SAMPLES, N_HIDDEN1_A], 0, 0.1)),
        'w2A': tf.Variable(tf.random_normal([N_HIDDEN1_A, N_HIDDEN2_A], 0, 0.1)),
        'w3A': tf.Variable(tf.random_normal([N_HIDDEN2_A, N_HIDDEN3_A], 0, 0.1)),
        'w1B': tf.Variable(tf.random_normal([N_SAMPLES, N_HIDDEN1_B], 0, 0.1)),
        'w2B': tf.Variable(tf.random_normal([N_HIDDEN1_B, N_HIDDEN2_B], 0, 0.1)),
        'w3B': tf.Variable(tf.random_normal([N_HIDDEN2_B, N_HIDDEN3_B], 0, 0.1)),

        'w4': tf.Variable(tf.random_normal([N_HIDDEN3_A + N_HIDDEN3_B, N_HIDDEN4], 0, 0.1)),
        'w5': tf.Variable(tf.random_normal([N_HIDDEN4, N_HIDDEN5], 0, 0.1)),

        'out': tf.Variable(tf.random_normal([N_HIDDEN5, N_CLASSES], 0, 0.1)),
    }
    biases = {
        'b1A': tf.Variable(tf.random_normal([N_HIDDEN1_A], 0, 0.1)),
        'b2A': tf.Variable(tf.random_normal([N_HIDDEN2_A], 0, 0.1)),
        'b3A': tf.Variable(tf.random_normal([N_HIDDEN3_A], 0, 0.1)),
        'b1B': tf.Variable(tf.random_normal([N_HIDDEN1_B], 0, 0.1)),
        'b2B': tf.Variable(tf.random_normal([N_HIDDEN2_B], 0, 0.1)),
        'b3B': tf.Variable(tf.random_normal([N_HIDDEN3_B], 0, 0.1)),
        'b4': tf.Variable(tf.random_normal([N_HIDDEN4], 0, 0.1)),
        'b5': tf.Variable(tf.random_normal([N_HIDDEN5], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([N_CLASSES], 0, 0.1)),
    }

    # pre-processing step calculating fft and concatenating it to signal
    fft = tf.abs(tf.fft(tf.cast(x_input_ph, tf.complex64)))# / tf.constant(N_SAMPLES / 10, dtype=tf.float32)

    # FFT PART
    # first layer
    model_A = tf.add(tf.matmul(fft, weights['w1A']), biases['b1A'])
    model_A = tf.nn.relu(model_A)

    # second layer
    model_A = tf.add(tf.matmul(model_A, weights['w2A']), biases['b2A'])
    model_A = tf.nn.relu(model_A)

    # third layer
    model_A = tf.add(tf.matmul(model_A, weights['w3A']), biases['b3A'])
    model_A = tf.nn.relu(model_A)

    # SIGNAL PART
    # first layer
    model_B = tf.add(tf.matmul(fft, weights['w1B']), biases['b1B'])
    model_B = tf.nn.relu(model_B)

    # second layer
    model_B = tf.add(tf.matmul(model_B, weights['w2B']), biases['b2B'])
    model_B = tf.nn.relu(model_B)

    # third layer
    model_B = tf.add(tf.matmul(model_B, weights['w3B']), biases['b3B'])
    model_B = tf.nn.relu(model_B)

    # concatenating both parts
    model = tf.concat([model_A, model_B], -1)

    # fourth layer
    model = tf.add(tf.matmul(model, weights['w4']), biases['b4'])
    model = tf.nn.relu(model)

    # fifth layer
    model = tf.add(tf.matmul(model, weights['w5']), biases['b5'])
    model = tf.nn.relu(model)

    # output layer
    model = tf.abs(tf.add(tf.matmul(model, weights['out']), biases['out']))

    # cost and optimizer
    labels = tf.placeholder(tf.int32, [None])
    logits = model
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    # cost = tf.reduce_mean(tf.abs(prediction - y_output_ph))
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # TRAIN AND TEST NETWORK
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(N_TRAINING_EPOCHS):

            import time
            start_time = time.time()
            train_in = []
            train_out = []
            true_freq = []
            for _ in range(TRAIN_SIZE):

                signal_y, freq, net_result = create_train_sample(random_state, NOISE_RATIO)
                train_in.append(signal_y)
                train_out.append(net_result)
                true_freq.append(freq)

            # print("PREPARING DATA TIME: {}".format(time.time() - start_time))

            start_time = time.time()
            cum_loss = 0.
            cum_log_ratio_cost = 0.
            n_batches = int(TRAIN_SIZE / BATCH_SIZE)
            for i in range(n_batches):
                batch_in = train_in[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                batch_out = np.array(train_out[i*BATCH_SIZE:(i+1)*BATCH_SIZE])

                _, loss_results, logits_results = sess.run(
                        [optimizer, loss, logits],
                        feed_dict={
                            x_input_ph: batch_in,
                            labels: batch_out,
                            learning_rate: LEARNING_RATE
                        })
                batch_mean_loss = np.mean(loss_results)
                cum_loss += batch_mean_loss

                batch_true_freq = np.array(true_freq[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
                predicted_freqs = np.array(
                        [FREQS[idx] for idx in np.argmax(logits_results, axis=-1)]
                )

                log_ratio_costs = np.abs(np.log(batch_true_freq/predicted_freqs))
                cum_log_ratio_cost += np.mean(log_ratio_costs)

            avg_cost = cum_loss / n_batches
            avg_log_ratio_cost = cum_log_ratio_cost / n_batches

            print("Epoch: {:04d}, loss={:.9f}, log_ratio_cost={:.7f}"
                  "".format((epoch + 1), avg_cost, avg_log_ratio_cost))

            # compute of cost on development set
            if epoch % 1 == 0:
                batch_dev_in = dev_in
                # batch_dev_out = np.array(dev_out)
                logits_results = sess.run(
                        [logits],
                        feed_dict={
                            x_input_ph: batch_dev_in,
                            # labels: batch_dev_out,
                        })

                dev_true_freqs = np.array(dev_true_freq)
                predicted_freqs = np.array(
                        [FREQS[idx] for idx in np.argmax(logits_results, axis=-1)]
                )

                log_ratio_cost = np.abs(np.log(dev_true_freqs/predicted_freqs))

                # avg_cost = np.mean(validation_loss)
                avg_log_ratio_cost = np.mean(log_ratio_cost)

                print("Validation Epoch: {:04d}, loss={:.9f}, log_ratio_cost={:.7f}"
                      "".format((epoch + 1), 0, avg_log_ratio_cost))

                watches = np.vstack([dev_true_freqs, predicted_freqs]).T
                stop = 1
            stop = 1

            LEARNING_RATE *= 0.99
            BATCH_SIZE += 3