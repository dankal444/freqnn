import numpy as np


def random_float(min, max, size, random_state=None):
    if random_state is None:
        return min + (max - min) * np.random.rand(size)
    else:
        return min + (max - min) * random_state.rand(size)

def make_sure_dir_exists(directory_path, is_file_path=False):
    if is_file_path:
        directory_path = os.path.dirname(directory_path)
    try:
        os.makedirs(directory_path)
    except OSError:
        if not os.path.isdir(directory_path):
            raise
