import argparse
import logging
import os
import wave

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def parse_file(filepath, label):
    label = label if label else filepath
    wav_file = wave.open(filepath, 'rb')
    frames = wav_file.getnframes()
    channels = wav_file.getnchannels()
    logger.info(f'total frames: {frames}')
    logger.info(f'channels: {channels}')

    # Extract Raw Audio from Wav File
    signal = wav_file.readframes(-1)
    signal = np.fromstring(signal, "Int16")
    framerate = wav_file.getframerate()
    logger.info(f'framerate: {framerate}, length (s): {len(signal) / framerate}')

    time_vector = np.linspace(0, len(signal) / framerate, num=len(signal))

    wave_data = pd.DataFrame({'signal': signal, 'time': time_vector})
    wave_data['signal_abs'] = wave_data['signal'].abs()
    logger.info(wave_data.describe())

    save_waveform(wave_data[['signal_abs', 'time']], label)


def parse_args():
    parser = argparse.ArgumentParser(description='process wavfiles.')
    parser.add_argument('--file_path', dest='file_path', action='store', help='path to wavfile', required=True)
    parser.add_argument('--label', dest='label', action='store', help='label for waveform', required=True)
    args = parser.parse_args()
    return args


def save_waveform(wave_data: pd.DataFrame, label: str):
    f = plt.figure(1)
    plt.title(f"Signal Wave: {label}")
    plt.plot(wave_data['time'], wave_data['signal_abs'])
    plt.show()
    logging.info(f'saving plot to {label}.pdf')
    f.savefig(f'{label}.pdf', format='pdf', bbox_inches='tight')


def main():
    logger.info('initialising')
    args = parse_args()
    parse_file(args.file_path, args.label)


if __name__ == '__main__':
    main()
