import os
import subprocess
from shutil import copyfile

import pandas as pd

def make_sure_dir_exists(directory_path, is_file_path=False):
    if is_file_path:
        directory_path = os.path.dirname(directory_path)
    try:
        os.makedirs(directory_path)
    except OSError:
        if not os.path.isdir(directory_path):
            raise


if __name__ == '__main__':
    noise_db_csv_path = r'Y:\bazy\audio\acoustic_simulator\acoustic_simulator\noise-samples\noise-db.csv'
    noise_db_path = os.path.dirname(noise_db_csv_path)
    noise_db_output_path = r'C:\bazy\noises'
    ffmpeg_path = r'C:\Programy\ffmpeg-4.1-win64-static\bin\ffmpeg'

    noise_db = pd.read_csv(noise_db_csv_path, sep=';')
    # convert each wave to 16kHz, mono
    # for i_row, wave in noise_db.iterrows():
    #     noise_input_path = os.path.join(noise_db_path, wave['link'])
    #     noise_output_path = os.path.join(noise_db_output_path, wave['link'])
    #     make_sure_dir_exists(noise_output_path, is_file_path=True)
    #     args = [ffmpeg_path,
    #             '-y',
    #             '-i', noise_input_path,
    #             '-map_channel', '0.0.0', '-ar', '16000', noise_output_path]
    #     subprocess.call(args)

    # copy .csv base description file too
    output_csv_path = os.path.join(noise_db_output_path, os.path.basename(noise_db_csv_path))
    copyfile(noise_db_csv_path, output_csv_path)