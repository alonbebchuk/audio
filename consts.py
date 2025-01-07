import os

import numpy as np

# common
BASE_DIR = os.path.dirname(__file__)
# data acquisition
ORIGINAL_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000
NUM_DIGITS = 10
WIN_LENGTH_SAMPLES = int(0.025 * TARGET_SAMPLE_RATE)
HOP_LENGTH_SAMPLES = int(0.010 * TARGET_SAMPLE_RATE)
N_FILTER_BANKS = 80
# forward algorithm
SEQUENCE = 'aba'
ALPHABET_LABELS = {0: 'a', 1: 'b', 2: '^'}
PRED = np.array([
    [0.8, 0.2, 0.0],
    [0.2, 0.8, 0.0],
    [0.3, 0.7, 0.0],
    [0.09, 0.8, 0.11],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
