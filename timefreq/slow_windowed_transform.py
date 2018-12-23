# comparison of time-freq analysis algorithms
# - stft  with window function
# - stft  without window function
# - slow window transform (windowed analytic signals)


import matplotlib.pyplot as plt
import numpy as np
from timefreq import tools

TWO_PI = 2 * np.pi

if __name__ == '__main__':
    some_wave_path = r"/media/dankal/Y/bazy/audio/snuv/snuv_database/100k36_28lat/115_28lat.wav"
    some_wave_path = r"/media/dankal/Y/bazy/audio/snuv/snuv_database/10m36_21lat/5_21lat.wav"

    wave_data, fs = tools.read_wave(some_wave_path) #, resample_to_freq=8000)
    wave_data -= np.mean(wave_data)

    FRAME_LENGTH = 0.05
    FRAME_RATE = 0.005
    NFFT = 2048

    frame_length_in_samples = int(FRAME_LENGTH * fs)
    frame_rate_in_samples = int(FRAME_RATE * fs)
    window = np.hamming(frame_length_in_samples)

    # stft without window function
    stft_without_window_spectra = []
    for frame in tools.frame_generator(wave_data, frame_length_in_samples, frame_rate_in_samples):
        magnitude_spectrum = np.abs(np.fft.fft(frame, NFFT))
        magnitude_spectrum = magnitude_spectrum[0:int(len(magnitude_spectrum) / 2)]
        stft_without_window_spectra.append(magnitude_spectrum)

    stft_without_window_spectra = np.vstack(stft_without_window_spectra)

    # stft with window function
    stft_window_spectra = []
    for frame in tools.frame_generator(wave_data, frame_length_in_samples, frame_rate_in_samples):
        windowed_frame = window * frame
        magnitude_spectrum = np.abs(np.fft.fft(windowed_frame, NFFT))
        magnitude_spectrum = magnitude_spectrum[0:int(len(magnitude_spectrum) / 2)]
        stft_window_spectra.append(magnitude_spectrum)

    stft_window_spectra = np.vstack(stft_window_spectra)

    # slow windowed transform
    # first prepare analytic signals
    analysis_sine_signals = []
    analysis_cosine_signals = []
    analysed_freqs = np.linspace(0, fs/2 - fs/NFFT, int(NFFT/2))
    x_domain = np.linspace(0, FRAME_LENGTH - FRAME_LENGTH/frame_length_in_samples, frame_length_in_samples)
    for freq in analysed_freqs:
        sine_signal = np.sin((TWO_PI * freq) * x_domain)
        windowed_sine_signal = sine_signal * window
        analysis_sine_signals.append(windowed_sine_signal)

        cosine_signal = np.cos((TWO_PI * freq) * x_domain)
        windowed_cosine_signal = cosine_signal * window
        analysis_cosine_signals.append(windowed_cosine_signal)
    analysis_sine_signals = np.vstack(analysis_sine_signals)
    analysis_cosine_signals = np.vstack(analysis_cosine_signals)

    swt_spectra = []
    for frame in tools.frame_generator(wave_data, frame_length_in_samples, frame_rate_in_samples):
        windowed_frame = window * frame
        sine_spectrum = np.dot(analysis_sine_signals, frame)
        cosine_spectrum = np.dot(analysis_cosine_signals, frame)
        magnitude_spectrum = np.sqrt(sine_spectrum ** 2 + cosine_spectrum ** 2)

        swt_spectra.append(magnitude_spectrum)

    swt_spectra = np.vstack(stft_without_window_spectra)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
    axes[0].imshow(np.log(stft_without_window_spectra.T), aspect='auto')
    axes[0].set_ylim(0, NFFT / 8)
    axes[1].imshow(np.log(stft_window_spectra.T), aspect='auto')
    axes[1].set_ylim(0, NFFT / 8)
    # axes[2].imshow(np.log(swt_spectra.T), aspect='auto')
    # axes[0].set_ylim(0, NFFT/8)
    # axes[3].imshow(np.log(stft_window_spectra.T) - np.log(stft_without_window_spectra.T), aspect='auto')
    plt.show()

    stop = 1
