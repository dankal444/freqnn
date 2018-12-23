# 2) sin + harm (1D) --> freq (0D)
# network computing regression, calculating frequency of a sine with harmonics, each with some random phase shift
# regression a la classification
# same as 07, making signal generation on GPU

import numpy as np
import tensorflow as tf
from timefreq import tools

NOISE_RATIO = 0.001   # fixme 0.0005

DURATION = 0.125   # [s]
FS = 8000           # [Hz]
N_SAMPLES = int(DURATION * FS) + 1
MIN_FREQ = 40       # [Hz]
MAX_FREQ = 600     # [Hz]
MIN_FREQ_TEST = MIN_FREQ + 1
MAX_FREQ_TEST = MAX_FREQ - 10
MIN_ANALYSIS_FREQ = 20
MAX_ANALYSIS_FREQ = 3700

MIN_HARMONICS = 5
TWO_PI = 2 * np.pi

A4_FREQ = 440
log_min_interval_1_32_semitone = 1.0 / (12 * 32)  # fixme  4 -> 32
log2_a4_freq = np.log2(A4_FREQ)
log2_min_freq = np.log2(MIN_FREQ)
log2_max_freq = np.log2(MAX_FREQ)
min_log_freq = log2_a4_freq \
               - int((log2_a4_freq - log2_min_freq) / log_min_interval_1_32_semitone) * log_min_interval_1_32_semitone
max_log_freq = log2_a4_freq \
               + int((log2_max_freq - log2_a4_freq) / log_min_interval_1_32_semitone) * log_min_interval_1_32_semitone
n_tones = (max_log_freq - min_log_freq) / log_min_interval_1_32_semitone
n_tones = int(n_tones) + 1
FUNDAMENTAL_FREQS = np.power(2.0, np.linspace(min_log_freq, max_log_freq, n_tones))


log_min_interval_1_32_semitone = 1.0 / (12 * 4)  # fixme  4 -> 32
log2_min_freq = np.log2(MIN_ANALYSIS_FREQ)
log2_max_freq = np.log2(MAX_ANALYSIS_FREQ)
min_log_freq = log2_a4_freq \
               - int((log2_a4_freq - log2_min_freq) / log_min_interval_1_32_semitone) * log_min_interval_1_32_semitone
max_log_freq = log2_a4_freq \
               + int((log2_max_freq - log2_a4_freq) / log_min_interval_1_32_semitone) * log_min_interval_1_32_semitone
n_tones = (max_log_freq - min_log_freq) / log_min_interval_1_32_semitone
n_tones = int(n_tones) + 1
ANALYSIS_FREQS = np.power(2.0, np.linspace(min_log_freq, max_log_freq, n_tones))


SEED = 0
TRAIN_SIZE = 20000
DEV_SIZE = 20000
TEST_SIZE = 10000


# network params
INC = 0.5
N_HIDDEN1 = int(4048 * INC)
N_HIDDEN2 = int(4048 * INC)
N_HIDDEN3 = int(4048 * INC)

N_CLASSES = len(FUNDAMENTAL_FREQS)

LEARNING_RATE = 0.0001
N_TRAINING_EPOCHS = 300
BATCH_SIZE = 30


def generate_signal(freq, random_state, noise_ratio=None):
    signal_x = np.linspace(0, DURATION, N_SAMPLES)

    current_harmonic = freq
    possible_harmonics = []
    while current_harmonic < FS / 2:
        possible_harmonics.append(current_harmonic)
        current_harmonic += freq

    n_harmonics = random_state.randint(MIN_HARMONICS, len(possible_harmonics))
    chosen_harmonics = random_state.choice(possible_harmonics, n_harmonics, replace=False)

    signal_y = np.zeros(len(signal_x))
    for current_freq in chosen_harmonics:
        if noise_ratio is not None:
            current_freq = current_freq \
                           - current_freq * noise_ratio / 2.0 \
                           + random_state.rand() * current_freq * noise_ratio

        start_phase = random_state.rand() * TWO_PI
        signal_y += np.sin((TWO_PI * current_freq) * signal_x + start_phase)

    return signal_y


def create_test_sample(random_state=None, noise_ratio=None):
    if random_state is None:
        random_state = np.random.RandomState()
    min_log_freq = np.log2(MIN_FREQ_TEST)
    max_log_freq = np.log2(MAX_FREQ_TEST)
    log_freq = min_log_freq + (max_log_freq - min_log_freq) * random_state.rand()
    freq = np.power(2.0, log_freq)

    signal_y = generate_signal(freq, random_state, noise_ratio)

    return signal_y, freq


# @tools.do_profile(follow=[generate_signal])
def create_train_sample(random_state=None, noise_ratio=None):
    if random_state is None:
        random_state = np.random.RandomState()

    freq_idx = np.random.randint(2, N_CLASSES - 2)
    freq = FUNDAMENTAL_FREQS[freq_idx]

    signal_y = generate_signal(freq, random_state, noise_ratio)

    return signal_y, freq, freq_idx


if __name__ == '__main__':
    # CREATE DATABASE
    random_state = np.random.RandomState(SEED)

    dev_in = []; dev_true_freq = []
    for _ in range(DEV_SIZE):
        signal_y, freq = create_test_sample(random_state, NOISE_RATIO)
        dev_in.append(signal_y)
        dev_true_freq.append(freq)

    # GENERATE SIGNAL
    x_input_ph = tf.placeholder(tf.float32, [None, N_SAMPLES])

    x_domain = np.linspace(0, DURATION, N_SAMPLES)
    # x_domain = tf.constant(), dtype=tf.float32)

    # PREPARE NETWORK
    # first, prepare FFT-matrices
    sine_sdt_matrix = []
    cosine_sdt_matrix = []
    for freq in ANALYSIS_FREQS:
        sine_sdt_matrix.append(
            np.sin(TWO_PI * freq * x_domain)
        )
        cosine_sdt_matrix.append(
            np.cos(TWO_PI * freq * x_domain)
        )
    sine_sdt_matrix = np.vstack(sine_sdt_matrix)
    cosine_sdt_matrix = np.vstack(cosine_sdt_matrix)


    # similar matrix, but multiplied by hamming window
    hamming_window = np.hamming(N_SAMPLES)
    windowed_sine_sdt_matrix = sine_sdt_matrix * hamming_window
    windowed_cosine_sdt_matrix = cosine_sdt_matrix * hamming_window

    # rest of weights

    weights = {
        'wHamming': tf.constant(hamming_window, dtype=tf.float32),
        'wSine': tf.constant(sine_sdt_matrix.T, dtype=tf.float32),
        'wCosine': tf.constant(cosine_sdt_matrix.T, dtype=tf.float32),
        'wWindowedSine': tf.constant(windowed_sine_sdt_matrix.T, dtype=tf.float32),
        'wWindowedCosine': tf.constant(windowed_cosine_sdt_matrix.T, dtype=tf.float32),

        'w1': tf.Variable(tf.random_normal([len(sine_sdt_matrix) * 1, N_HIDDEN1], 0, 0.0001)),
        'w2': tf.Variable(tf.random_normal([N_HIDDEN1, N_HIDDEN2], 0, 0.001)),
        'w3': tf.Variable(tf.random_normal([N_HIDDEN2, N_HIDDEN3], 0, 0.001)),
        'out': tf.Variable(tf.random_normal([N_HIDDEN1, N_CLASSES], 0, 0.001)),
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([N_HIDDEN1], 0, 0.0001)),
        'b2': tf.Variable(tf.random_normal([N_HIDDEN2], 0, 0.001)),
        'b3': tf.Variable(tf.random_normal([N_HIDDEN3], 0, 0.001)),
        'out': tf.Variable(tf.random_normal([N_CLASSES], 0, 0.001)),
    }

    # pre-processing step,
    # fft = tf.fft(tf.cast(x_input_ph, tf.complex64))# / tf.constant(N_SAMPLES / 10, dtype=tf.float32)
    signal = x_input_ph
    signal_windowed = tf.multiply(signal, weights['wHamming'])

    # signal = tf.expand_dims(signal, 1)
    # signal_windowed = tf.expand_dims(signal_windowed,1)

    # FFT PART
    sine_sdt = tf.matmul(signal, weights['wSine'])
    cosine_sdt = tf.matmul(signal, weights['wCosine'])
    magnitude_sdt = tf.sqrt(tf.add(
        tf.multiply(sine_sdt, sine_sdt),
        tf.multiply(cosine_sdt, cosine_sdt)
    ))

    windowed_sine_sdt = tf.matmul(signal_windowed, weights['wWindowedSine'])
    windowed_cosine_sdt = tf.matmul(signal_windowed, weights['wWindowedCosine'])
    magnitude_windowed_sdt = tf.sqrt(tf.add(
        tf.multiply(windowed_sine_sdt, windowed_sine_sdt),
        tf.multiply(windowed_cosine_sdt, windowed_cosine_sdt)
    ))

    # concatenated_sdt = tf.concat([magnitude_sdt, magnitude_windowed_sdt], axis=-1)

    # first layer
    model = tf.add(tf.matmul(magnitude_windowed_sdt, weights['w1']), biases['b1'])
    model = tf.nn.relu(model)

    # second layer
    # model = tf.add(tf.matmul(model, weights['w2']), biases['b2'])
    # model = tf.nn.relu(model)
    #
    # # third layer
    # model = tf.add(tf.matmul(model, weights['w3']), biases['b3'])
    # model = tf.nn.relu(model)

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

        for i_epoch in range(N_TRAINING_EPOCHS):

            import time
            start_time = time.time()
            if i_epoch % 1 == 0:
                import os
                train_data_directory_path = '/home/dankal/bazy/audio/fake/f0analysis_ver2'
                file_path = os.path.join(train_data_directory_path, 'file{0:03d}'.format(i_epoch % 100))
                loaded_data = tools.load_pickled_data(file_path)
                train_in = loaded_data['train_in']
                train_out = loaded_data['train_out']
                true_freq = loaded_data['true_freq']

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
                        [FUNDAMENTAL_FREQS[idx] for idx in np.argmax(logits_results, axis=-1)]
                )

                log_ratio_costs = np.abs(np.log(batch_true_freq/predicted_freqs))
                cum_log_ratio_cost += np.mean(log_ratio_costs)

            avg_cost = cum_loss / n_batches
            avg_log_ratio_cost = cum_log_ratio_cost / n_batches

            # print("TRAINING EPOCH TIME: {}".format(time.time() - start_time))

            print("Epoch: {:04d}, loss={:.9f}, log_ratio_cost={:.7f}"
                  "".format((i_epoch + 1), avg_cost, avg_log_ratio_cost))

            # compute of cost on development set
            if i_epoch % 1 == 0:
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
                        [FUNDAMENTAL_FREQS[idx] for idx in np.argmax(logits_results, axis=-1)]
                )

                log_ratio_cost = np.abs(np.log(dev_true_freqs/predicted_freqs))

                # avg_cost = np.mean(validation_loss)
                avg_log_ratio_cost = np.mean(log_ratio_cost)

                print("Validation Epoch: {:04d}, loss={:.9f}, log_ratio_cost={:.7f}"
                      "".format((i_epoch + 1), 0, avg_log_ratio_cost))

                watches = np.vstack([dev_true_freqs, predicted_freqs]).T
                if i_epoch % 50 == 0:
                    stop = 1
            stop = 1

            LEARNING_RATE *= 0.99
            # BATCH_SIZE += 3