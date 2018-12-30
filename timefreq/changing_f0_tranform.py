# comparison of time-freq analysis algorithms
# - stft  with window function
# - stft  without window function
# - slow window transform (windowed analytic signals)


import matplotlib.pyplot as plt
import numpy as np
from timefreq import tools
from collections import defaultdict

TWO_PI = 2 * np.pi


def main():
    some_wave_path = r"/media/dankal/Y/bazy/audio/snuv/snuv_database/100k36_28lat/115_28lat.wav"
    some_wave_path = r"/media/dankal/Y/bazy/audio/snuv/snuv_database/10m36_21lat/5_21lat.wav"

    wave_data, fs = tools.read_wave(some_wave_path) #, resample_to_freq=8000)
    wave_data -= np.mean(wave_data)

    NFFT = 4096
    f0_vector, voiced_measures, spectrogram, perc_change_winners \
        = parametrise(fs, wave_data, nfft=NFFT)

    f0_vector2, voiced_measures2, spectrogram2, perc_change_winners2 \
        = parametrise(fs, wave_data, use_window=True, nfft=NFFT)

    f0_vector3, voiced_measures3, spectrogram3, perc_change_winners3 \
        = parametrise(fs, wave_data, use_window=True, nfft=NFFT, perc_change_array=np.zeros(1))

    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(15, 10))
    axes[0].imshow(np.log(spectrogram.T), aspect='auto')
    axes[0].set_ylim(0, NFFT / 8)
    axes[1].imshow(np.log(spectrogram2.T), aspect='auto')
    axes[1].set_ylim(0, NFFT / 8)
    axes[2].imshow(np.log(spectrogram3.T), aspect='auto')
    axes[2].set_ylim(0, NFFT / 8)
    axes[3].plot(f0_vector)
    axes[3].plot(f0_vector2)
    axes[3].plot(f0_vector3)
    axes[4].plot(voiced_measures)
    axes[4].plot(voiced_measures2)
    axes[4].plot(voiced_measures3)
    axes[5].plot(perc_change_winners)
    axes[5].plot(perc_change_winners2)
    axes[5].plot(perc_change_winners3)
    plt.show()

    stop = 1


def parametrise(fs, wave_data, use_window=False, nfft=1024, perc_change_array=None):
    FRAME_LENGTH = 0.05
    FRAME_RATE = 0.01
    frame_length_in_samples = int(FRAME_LENGTH * fs)
    frame_rate_in_samples = int(FRAME_RATE * fs)
    window = np.hanning(frame_length_in_samples)

    analysis_sine_signals = defaultdict(list)
    analysis_cosine_signals = defaultdict(list)
    x_domain = np.linspace(0, FRAME_LENGTH - FRAME_LENGTH / frame_length_in_samples, frame_length_in_samples)

    if perc_change_array is None:
        perc_change_array = np.arange(-10.0, 10.1, 1.0)

    for f0_perc_change in perc_change_array:
        for freq in np.linspace(0, fs/2, int(nfft / 2)):  # fixme make constants
            freq_array = np.linspace(freq, freq + freq * f0_perc_change / 100.0, frame_length_in_samples)

            temp = TWO_PI * freq_array * x_domain
            sine_signal = np.sin(temp)
            cosine_signal = np.cos(temp)

            analysis_sine_signals[f0_perc_change].append(sine_signal)
            analysis_cosine_signals[f0_perc_change].append(cosine_signal)

        analysis_sine_signals[f0_perc_change] = np.vstack(analysis_sine_signals[f0_perc_change])
        analysis_cosine_signals[f0_perc_change] = np.vstack(analysis_cosine_signals[f0_perc_change])

    # checked f0 freqs
    f0_frequencies = np.exp(np.linspace(np.log(50), np.log(500), 200))
    # prepare __|¯¯|__|¯¯|__|¯¯|__|¯¯|__|¯¯|__|¯¯|  vectors
    min_freq = 200.0  # [Hz]
    max_freq = 2500.0  # [Hz]
    span = 25.0  # [%]
    peaks_indicators = []
    valleys_indicators = []
    for freq in f0_frequencies:
        first_peak = freq * np.ceil(min_freq / freq)
        peaks_harmonics = np.arange(first_peak, max_freq, freq)

        half_range = int(span / 100 * freq * nfft / fs / 2)

        temp_peaks = np.zeros(int(nfft / 2))
        for harmonic in peaks_harmonics:
            harmonic_idx = int(nfft * harmonic / fs)
            temp_peaks[harmonic_idx - half_range: harmonic_idx + half_range + 1] = 1
        peaks_indicators.append(temp_peaks)

        temp_valleys = np.zeros(int(nfft / 2))
        valleys_harmonics = peaks_harmonics - freq / 2  # fixme +/-
        for harmonic in valleys_harmonics:
            harmonic_idx = int(nfft * harmonic / fs)
            temp_valleys[harmonic_idx - half_range: harmonic_idx + half_range + 1] = 1
        valleys_indicators.append(temp_valleys)
    peaks_indicators = np.vstack(peaks_indicators)
    valleys_indicators = np.vstack(valleys_indicators)
    peaks_indicators = (peaks_indicators.T / np.sum(peaks_indicators, axis=-1)).T
    valleys_indicators = (valleys_indicators.T / np.sum(valleys_indicators, axis=-1)).T

    spectrogram = []
    voiced_measures = []
    f0_vector = []
    ranking_values = []
    perc_change_winners = []
    idx_frame = 0
    for frame in tools.frame_generator(wave_data, frame_length_in_samples, frame_rate_in_samples):
        if idx_frame > 55:
            stop = 1
        idx_frame += 1

        if use_window:
            frame *= window

        # first, find best perc change
        spectra = []
        voicing_measures = []
        f0_per_perc_change = []
        for f0_perc_change, sine_signals in analysis_sine_signals.items():
            cosine_signals = analysis_cosine_signals[f0_perc_change]

            sine_dot_product = np.dot(sine_signals, frame)
            cosine_dot_product = np.dot(cosine_signals, frame)

            energy_spectrum = sine_dot_product ** 2 + cosine_dot_product ** 2

            # next, find f0 for given perc change
            log_energy_spectrum = 10.0 * np.log10(energy_spectrum)

            peaks_energy = np.dot(peaks_indicators, log_energy_spectrum)
            valleys_energy = np.dot(valleys_indicators, log_energy_spectrum)
            log_energy_ratios = peaks_energy - valleys_energy

            voicing_measure = np.max(log_energy_ratios)
            f0_idx = np.argmax(log_energy_ratios)
            f0 = f0_frequencies[f0_idx]

            spectra.append(energy_spectrum)
            voicing_measures.append(voicing_measure)
            f0_per_perc_change.append(f0)

        idx_best_perc_change = np.argmax(np.array(voicing_measures))

        voiced_measures.append(voicing_measures[idx_best_perc_change])
        f0_vector.append(f0_per_perc_change[idx_best_perc_change])
        perc_change_winners.append(perc_change_array[idx_best_perc_change])
        spectrogram.append(spectra[idx_best_perc_change])

    return np.array(f0_vector), np.array(voiced_measures), np.vstack(spectrogram), np.array(perc_change_winners)


if __name__ == '__main__':
    main()