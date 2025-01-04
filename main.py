import math
import os
from collections import defaultdict

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000
WIN_LENGTH_SAMPLES = int(0.025 * TARGET_SAMPLE_RATE)
HOP_LENGTH_SAMPLES = int(0.010 * TARGET_SAMPLE_RATE)
N_FILTER_BANKS = 80


def resample_audio(audio, sr, new_sr):
    num_samples = int(len(audio) * new_sr / sr)
    resampled_audio = resample(audio, num_samples)
    return resampled_audio


def load_recording(filepath):
    loaded_audio, loaded_rate = librosa.load(filepath, sr=ORIGINAL_SAMPLE_RATE)
    resampled_audio = resample_audio(loaded_audio, ORIGINAL_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    return resampled_audio


def load_recordings():
    recordings = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    for root, dirs, files in os.walk(os.path.join(BASE_DIR, 'recordings')):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                gender, name, filename = filepath.split(os.sep)[-3:]
                digit = filename.split('.')[0]
                recordings[gender][name][digit] = load_recording(filepath)
    return recordings


def get_melspectrogram(audio):
    mstft = librosa.feature.melspectrogram(y=audio, sr=TARGET_SAMPLE_RATE, n_fft=WIN_LENGTH_SAMPLES, hop_length=HOP_LENGTH_SAMPLES, n_mels=N_FILTER_BANKS)
    mstft_db = librosa.power_to_db(mstft, ref=np.max)
    return mstft_db


def plot_melspectrograms(recordings):
    females = sorted(recordings.get('females', {}).items())
    males = sorted(recordings.get('males', {}).items())
    all = females + males
    fig, axes = plt.subplots(nrows=len(all), ncols=10, figsize=(20, 20))
    for y, (name, audios) in enumerate(all):
        for x in range(10):
            melspectrogram = get_melspectrogram(audios[str(x)])
            ax = axes[y, x]
            librosa.display.specshow(melspectrogram, sr=TARGET_SAMPLE_RATE, hop_length=HOP_LENGTH_SAMPLES, y_axis='hz', x_axis='time', ax=ax)
            ax.set_title(f'{x}: {name}')

    plt.tight_layout()
    plt.savefig('melspectrograms.jpg')
    plt.close()


## TODO: Ohad DTW

def CTC_collapse_B(path):
    collapsed_path = []
    prev = None
    for curr in path:
        if curr != '^' and curr != prev:
            collapsed_path.append(curr)
        prev = curr
    return collapsed_path


def sequence_to_padded_sequence_indices(sequence, alphabet_labels, blank_index):
    alphabet_indices = {l: i for i, l in alphabet_labels.items()}
    padded_sequence_indices = [blank_index if i % 2 == 0 else alphabet_indices[sequence[(i - 1) // 2]] for i in range(len(sequence) * 2 + 1)]
    return padded_sequence_indices


def CTC_forward_pass(sequence, pred, alphabet_labels, blank_index, force_align=False):
    padded_sequence_indices = sequence_to_padded_sequence_indices(sequence, alphabet_labels, blank_index)

    T, padded_sequence_length = pred.shape[0], len(padded_sequence_indices)
    alpha_matrix = np.zeros(shape=(T, padded_sequence_length), dtype=np.float32)
    backpointers_matrix = np.zeros(shape=(T, padded_sequence_length), dtype=np.int32)

    def alpha_and_backpointer(t, s):
        p = pred[t, padded_sequence_indices[s]]
        if t == 0:
            return p if s <= 1 else 0, -1
        alpha_tags = [alpha_matrix[t - 1, s]]
        if s >= 1:
            alpha_tags.append(alpha_matrix[t - 1, s - 1])
        if s >= 2 and not (padded_sequence_indices[s] == blank_index or (s >= 2 and padded_sequence_indices[s] == padded_sequence_indices[s - 2])):
            alpha_tags.append(alpha_matrix[t - 1, s - 2])
        alpha_tag = max(alpha_tags) if force_align else sum(alpha_tags)
        alpha = alpha_tag * p
        backpointer = math.argmax(alpha_tags) if force_align else None
        return alpha, backpointer

    for t in range(0, T):
        for s in range(0, padded_sequence_length):
            alpha_matrix[t, s], backpointers_matrix[t, s] = alpha_and_backpointer(t, s)

    return alpha_matrix, backpointers_matrix


def plot_alpha_matrix(alpha_matrix, padded_sequence_labels):
    alpha_matrix = alpha_matrix.T
    plt.imshow(alpha_matrix, cmap='viridis', interpolation='nearest')
    plt.title('Alpha Matrix')
    plt.xlabel('Timestep')
    plt.ylabel('Padded Sequence Label')

    y_labels = [padded_sequence_labels[i] for i in range(alpha_matrix.shape[0])]
    plt.yticks(ticks=range(len(y_labels)), labels=y_labels)

    rows, cols = alpha_matrix.shape
    for i in range(rows):
        for j in range(cols):
            plt.text(j, i, f'{alpha_matrix[i, j]:.2f}',
                     ha='center', va='center',
                     color='white' if alpha_matrix[i, j] < 0.5 else 'black')

    plt.savefig('alpha_matrix.jpg')
    plt.close()


if __name__ == '__main__':
    # Data Acquisition
    recordings = load_recordings()
    #
    plot_melspectrograms(recordings)

    # Forward Algorithm
    pred = np.zeros(shape=(5, 3), dtype=np.float32)
    pred[0][0] = 0.8
    pred[0][1] = 0.2
    pred[1][0] = 0.2
    pred[1][1] = 0.8
    pred[2][0] = 0.3
    pred[2][1] = 0.7
    pred[3][0] = 0.09
    pred[3][1] = 0.8
    pred[3][2] = 0.11
    pred[4][2] = 1.00
    alphabet_labels, blank_index = {0: 'a', 1: 'b', 2: '^'}, 2
    sequence = 'aba'
    padded_sequence_indices = sequence_to_padded_sequence_indices(sequence, alphabet_labels, blank_index)
    padded_sequence_labels = [alphabet_labels[i] for i in padded_sequence_indices]
    #
    alpha_matrix, _ = CTC_forward_pass(sequence, pred, alphabet_labels, blank_index)
    plot_alpha_matrix(alpha_matrix, padded_sequence_labels)
    alpha_matrix_force, backpointers_matrix_force = CTC_forward_pass(sequence, pred, alphabet_labels, blank_index, force_align=True)
    plot_alpha_matrix(alpha_matrix, padded_sequence_labels)
