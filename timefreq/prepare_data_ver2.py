import os
import numpy as np
from timefreq import tools


NOISE_RATIO = 0.0005   # fixme 0.0005

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

N_CLASSES = len(FUNDAMENTAL_FREQS)


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
    if noise_ratio is not None:
        freq = freq \
               - freq * noise_ratio / 2.0 \
               + random_state.rand() * freq * noise_ratio

    signal_y = generate_signal(freq, random_state, noise_ratio)

    return signal_y, freq, freq_idx


if __name__ == '__main__':
    output_directory_path = '/home/dankal/bazy/audio/fake/f0analysis_ver2_noise0_0005'

    for i_file in range(100):
        random_state = np.random.RandomState(SEED)
        train_in = []
        train_out = []
        true_freq = []
        for _ in range(TRAIN_SIZE):
            signal_y, freq, net_result = create_train_sample(random_state, NOISE_RATIO)
            train_in.append(signal_y.astype(np.float32))
            train_out.append(net_result)
            true_freq.append(freq)

        file_path = os.path.join(output_directory_path, 'file{0:03d}'.format(i_file))
        tools.save_pickled_data(
            {
                'train_in': train_in,
                'train_out': train_out,
                'true_freq': true_freq,
            },
            file_path,
            make_sure_dir_exist=True
        )

