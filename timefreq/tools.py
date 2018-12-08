import pickle
import fnmatch
import os
import resampy

import numpy as np
from scipy.io.wavfile import read, write
from line_profiler import LineProfiler


def make_sure_dir_exists(directory_path, is_file_path=False):
    if is_file_path:
        directory_path = os.path.dirname(directory_path)
    try:
        os.makedirs(directory_path)
    except OSError:
        if not os.path.isdir(directory_path):
            raise


def read_wave(wave_path, normalize_to_1=True, resample_to_freq=None):
    fs, wave_data = read(wave_path)

    if resample_to_freq and fs != resample_to_freq:
        wave_data = resampy.resample(wave_data, fs, resample_to_freq)
        fs = resample_to_freq

        # CARE RESAMPY DOES SOME STRANGE NOISE FLOOR

    if normalize_to_1:
        wave_data = wave_data.astype(np.float64)
        wave_data /= np.max(np.abs(wave_data))
    return wave_data, fs


def save_wave(wave_data, fs, wave_path, make_dir_if_not_exists=False):
    if make_dir_if_not_exists:
        make_sure_dir_exists(wave_path, is_file_path=True)
    write(wave_path, fs, wave_data)


def save_pickled_data(data_to_save, file_path, make_sure_dir_exist=False, protocol=None):
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    if make_sure_dir_exist:
        make_sure_dir_exists(file_path, is_file_path=True)
    with open(file_path, 'wb') as file:
        pickle.dump(data_to_save, file, protocol=protocol)


def load_pickled_data(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def do_profile(follow=None):
    """
    Decorate a function with this to get a profiling report printed.

    :param follow: list of names of functions called by the function being profiled that we also want to profile
    """
    if not follow:
        follow = []

    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()

        return profiled_func

    return inner


def concatenate_dict_values(a_dict, values_type='list'):
    """
    Assuming each value in a dict is a list!
    """
    if values_type is 'list':
        concatenated_list = []
        for values in a_dict.itervalues():
            concatenated_list += values
        return concatenated_list
    elif values_type is 'ndarray':
        concatenated_list = []
        for values in a_dict.itervalues():
            concatenated_list.append(values)
        return np.concatenate(concatenated_list)
    else:
        raise NotImplementedError()


def get_files(path, get_absolute=True, extension='wav'):
    path = path.replace('\\', '/')
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.' + extension):
            if get_absolute:
                filepath = os.path.join(root, filename)
            else:
                filepath = os.path.join(root[len(path) + 1:], filename)
            matches.append(filepath.replace('\\', '/'))
    return matches
