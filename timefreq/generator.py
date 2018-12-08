import numpy as np


class Signal(object):
    def __init__(self):
        pass

    # 0) empty signal
    def generate(self, n_samples, fs):
        signal = np.zeros(n_samples)
        signal_parameters = None #(np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples))   # freq, phase, magnitude
        return signal, signal_parameters


class SingleFreq(object):
    def __init__(self, freq_range=None, magnitude_range=None, random_state=None):
        if freq_range is None:
            freq_range = (0, np.Inf)
        if magnitude_range is None:
            magnitude_range = (0.001, 1000)  # that is 120dB range, more than enough
        self.freq_range = freq_range
        self.magnitude_range = magnitude_range
        self.random_state = np.random.RandomState(seed=random_state)

    def generate(self, n_samples, fs, signal_type='constant_freq'):
        if signal_type == 'constant_freq':
            return self._generate_sinus_signal(n_samples, fs)

        elif signal_type == 'magnitude_changes':
            return self._generate_am_signal(n_samples, fs)

        elif signal_type == 'magnitude_frequency_changes':
            return self._generate_constant_freq(n_samples, fs)

        else:
            raise ValueError("Wrong signal_type")

    # 1) single freq, no changes
    def _generate_sinus_signal(self, n_samples, fs):
        min_freq = self.freq_range[0]
        max_freq = min(self.freq_range[1], fs / 2.0)

        # todo uniformity of random in log-freq domain
        phase = 2 * np.pi * self.random_state.rand()
        freq = min_freq + (max_freq - min_freq) * self.random_state.rand()

        # magnitude distribution is uniform in logarithmic scale
        magnitude = np.exp(np.log(self.magnitude_range[0])
                           + (self.magnitude_range[1] - self.magnitude_range[0]) * self.random_state.rand())

        time_between_samples = 1.0 / fs
        x_domain = np.linspace(0, (n_samples-1) * time_between_samples, n_samples)
        signal = magnitude * np.sin(2 * np.pi * freq * x_domain + phase)
        signal_parameters = (np.zeros(n_samples) + freq, np.zeros(n_samples) + magnitude)  # freq, magnitude

        # todo: add phase to parameters
        return signal, signal_parameters

# 2) single freq, amplitude changes. Types of changes
#    - increasing/decreasing linearly
#    - increasing/decreasing with some polynomials
#    - increasing/decreasing exponentially
#    - increasing/decreasing logarithmically
#    - sinus as AM
#    - random smoothed
# SPECIFY SOME MAX POSSIBLE CHANGE OF MAGNITUDE

# 3) single sine signal, amplitude and freq changes
#    - types of changes same as above

# 4) above + noise

# 5) two signals - same as with one signal, but two signals :)

# 6) two signals, amplitude changes

# 7) ...

# X) three, four, five signals

# JITTER, SHIMMER, QUANTIZATION ERROR ETC.