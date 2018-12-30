import numpy as np
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from timefreq_pytorch import utils


class PitchDataset(Dataset):
    def __init__(self, n_samples=1600, freq_range=(45, 550), label_freq_range=(40, 600), a4_hz=440, random_state=None,
                 fs=16000, noise_db_path=None):
        self.n_samples = n_samples
        self.log_min_freq = np.log(freq_range[0])
        self.log_max_freq = np.log(freq_range[1])
        self.log_min
        self.difficulty = 0  # 0 to 10
        self.random_state = np.random.RandomState(random_state)
        self.fs = fs
        self.dt = 1 / self.fs

        self.xnew = np.linspace(0, 4, self.n_samples)
        self.ones = np.ones(self.n_samples)
        self.noise_db_path = noise_db_path


    def __len__(self):
        if self.mode == 'test':
            return len(self.wav_list)
        else:
            return len(self.wav_list) + self.n_silence

    def __getitem__(self, idx):
        base_freq = np.exp(utils.random_float(self.log_min_freq, self.log_max_freq, 1, self.random_state))[0]
        label =

        if self.difficulty < 4:
            freq_function_per_sample = np.ones(self.n_samples) * base_freq
        else:
            freq_change = (self.difficulty - 3) * 0.015
            freq_function = utils.random_float(base_freq * (1 - freq_change), base_freq * (1 + freq_change), 5)
            # make sure center frequency of given frame is base_freq
            freq_function -= freq_function[2] - base_freq
            f_cubic = interp1d(np.arange(5), freq_function, kind='cubic')
            freq_function_per_sample = f_cubic(self.xnew)

        if self.difficulty < 3:
            n_harmonics = self.random_state.randint(1, 1000)
        elif self.difficulty < 7:
            n_harmonics = self.random_state.randint(1, 12)
        else:
            n_harmonics = self.random_state.randint(1, 8)
        max_n_harmonics = int((self.fs / 2) * 0.94 / base_freq)
        n_harmonics = min(n_harmonics, max_n_harmonics)

        if n_harmonics < 3:
            chosen_harmonics = np.arange(3) + 1
        elif n_harmonics < 5:
            chosen_harmonics = self.random_state.choice(np.arange(6), n_harmonics) + 1
        else:
            chosen_harmonics = self.random_state.choice(np.arange(max_n_harmonics), n_harmonics) + 1

        # start phases are not based on difficulty
        start_phases = utils.random_float(0, 2 * np.pi, n_harmonics)

        # based on difficulty magnitude signals
        magnitudes = []
        if self.difficulty < 2:
            for i_harmonic in range(n_harmonics):
                magnitude_vector = self.ones
                magnitudes.append(magnitude_vector)

        elif self.difficulty < 6:
            magnitude_function = utils.random_float(0.05, 1.0, 5)
            f_cubic = interp1d(np.arange(5), magnitude_function, kind='cubic')
            magnitude_vector = f_cubic(self.xnew)
            for i_harmonic in range(n_harmonics):
                magnitudes.append(magnitude_vector)
        else:
            for i_harmonic in range(n_harmonics):
                magnitude_function = utils.random_float(0.05, 1.0, 5)
                f_cubic = interp1d(np.arange(5), magnitude_function, kind='cubic')
                magnitude_vector = f_cubic(self.xnew)
                magnitudes.append(magnitude_vector)

        sum_signals = []
        for i_harmonic in range(n_harmonics):
            phase_derivative = freq_function_per_sample * self.dt * chosen_harmonics[i_harmonic]
            phase_signal = np.cumsum(phase_derivative) + start_phases[i_harmonic]
            sine_signal = np.sin(phase_signal)
            sum_signals.append(magnitudes[i_harmonic] * sine_signal)

        sum_signal = np.sum(sum_signals, axis=0)



        return sum_signal

    def increase_difficulty(self):
        self.difficulty += 1

    def decrease_difficulty(self):
        self.difficulty -= 1

if __name__ == '__main__':

    noise_db_path = r''

    pitch_dataset = PitchDataset(noise_db_path)
    pitch_dataset.difficulty = 6
    pitch_dataset.__getitem__(1)