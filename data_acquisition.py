import glob
import os
from collections import defaultdict

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.signal import resample

from consts import BASE_DIR, ORIGINAL_SAMPLE_RATE, TARGET_SAMPLE_RATE, WIN_LENGTH_SAMPLES, HOP_LENGTH_SAMPLES, N_FILTER_BANKS, NUM_DIGITS


def get_resampled_audio(filepath):
    loaded_audio, loaded_rate = librosa.load(filepath, sr=ORIGINAL_SAMPLE_RATE)

    num_samples = int(len(loaded_audio) * TARGET_SAMPLE_RATE / ORIGINAL_SAMPLE_RATE)
    resampled_audio = resample(loaded_audio, num_samples)

    return resampled_audio


def get_gender_name_digit(filepath):
    gender, name, filename = filepath.split(os.sep)[-3:]
    digit = int(filename.split('.')[0])

    return gender, name, digit


def get_recordings():
    recordings = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    for filepath in glob.glob(os.path.join(BASE_DIR, 'recordings', '**', '*.wav'), recursive=True):
        gender, name, digit = get_gender_name_digit(filepath)
        resampled_audio = get_resampled_audio(filepath)

        recordings[gender][name][digit] = resampled_audio

    return recordings


def get_melspectrogram(audio):
    mstft = librosa.feature.melspectrogram(y=audio, sr=TARGET_SAMPLE_RATE, n_fft=WIN_LENGTH_SAMPLES, hop_length=HOP_LENGTH_SAMPLES, n_mels=N_FILTER_BANKS)
    mstft_db = librosa.power_to_db(mstft, ref=np.max)

    return mstft_db


def question1(recordings):
    females = [(audio, 'f') for audio in recordings['females'].values()]
    males = [(audio, 'm') for audio in recordings['males'].values()]

    class_representative = males[0][0]
    training_set = females[:2] + males[1:3]
    evaluation_set = females[2:] + males[3:]

    return class_representative, training_set, evaluation_set


def question2(recordings):
    names_and_recordings = list(recordings['females'].items()) + list(recordings['males'].items())
    fig, axes = plt.subplots(nrows=len(names_and_recordings), ncols=NUM_DIGITS, figsize=(20, 20))

    for y, (name, audios) in enumerate(names_and_recordings):
        for x in range(NUM_DIGITS):
            ax = axes[y, x]
            ax.set_title(f'{x}: {name}')

            melspectrogram = get_melspectrogram(audios[x])
            librosa.display.specshow(melspectrogram, sr=TARGET_SAMPLE_RATE, hop_length=HOP_LENGTH_SAMPLES, y_axis='hz', x_axis='s', ax=ax)

            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'plots/melspectrograms.jpg'))
    plt.close()
