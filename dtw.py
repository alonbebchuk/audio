from data_acquisition import question1, get_recordings, get_melspectrogram
from datasets import Dataset
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator
from consts import (
    ORIGINAL_SAMPLE_RATE,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH_SAMPLES,
    HOP_LENGTH_SAMPLES,
    N_FILTER_BANKS,
    NUM_DIGITS,
)
import librosa
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
# Get predictions for all samples in the training set
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os


# python3 dtw.py
"""
Question 3. Implement dynamic time warping (DTW):

a. Select the class representative recordings as the reference database (DB)

b. Implement the DTW algorithm that was described in the lecture.

c. Compare each audio recording in the training set with each of the audios in the DB using DTW algorithm. This algorithm aligns two sequences of feature vectors (in this case, Mel Spectrograms) by finding the optimal alignment that minimizes the total distance between them.

d. Distance Matrix: Construct a distance matrix with dimensions $4 \times 10$, where 4 is the number of speakers and 10 is the number of digits. Each cell in the matrix represents the DTW distance between a recording in the training set and a recording in the DB.

e. Show the distance matrix. This matrix has the dimensions of $4 \times 10$, where 4 is the number of speakers, and 10 is the number of digits.

f. Set Threshold and Determine Classification: Set up a threshold on the distances to determine the classification of each audio signal. Classify each recording in the training set based on the closest match in the DB. Calculate the accuracy over the training set.

g. Apply Threshold on Validation Set: Apply the selected threshold on the distance matrix of the validation set. Construct a confusion matrix to evaluate the classification accuracy over the validation set.

h. Try to improve results by:
    i. Normalizing the audio (using AGC)
    ii. Normalizing the distance w.r.t the length of each audio file
    iii. Plot the confusion matrix
"""


import numpy as np


def dtw_forward(x, y):
    n = x.shape[1]
    m = y.shape[1]

    # Initialize DTW matrix with infinity
    DTW = np.full((n, m), np.inf)

    # Set base case
    DTW[0, 0] = 0

    # Create cost matrix using broadcasting
    # Convert sequences to feature vectors using melspectrogram
    # Ensure same number of features by transposing
    x_mel = x.T  # Shape: (time_steps, n_mels)
    y_mel = y.T  # Shape: (time_steps, n_mels)

    # Calculate euclidean distance between feature vectors
    cost_matrix = np.linalg.norm(x_mel[:, np.newaxis] - y_mel, axis=2)

    # Fill in DTW matrix using vectorized operations
    for i in range(n):
        for j in range(m):
            to_consider = []
            if i > 0 and j > 0:
                to_consider.append(DTW[i - 1, j - 1])
            if i > 0:
                to_consider.append(DTW[i - 1, j])
            if j > 0:
                to_consider.append(DTW[i, j - 1])
            if i == 0 and j == 0:
                cost = 0
            else:
                cost = cost_matrix[i, j] + np.minimum.reduce(to_consider)
            DTW[i, j] = cost

    return DTW[n - 1, m - 1]


def create_dataset(raw_dataset):
    data_list = []
    for speaker_id, (speaker_rec, gender) in enumerate(raw_dataset):
        for digit, recording in speaker_rec.items():
            data_list.append(
                dict(
                    digit=digit,
                    recording=np.array(recording),
                    gender=gender,
                    speaker_id=speaker_id,
                )
            )
    ds = Dataset.from_list(data_list)
    ds.set_format("np")
    return ds


def apply_agc(audio, frame_length=2048, attack_time=0.01, release_time=0.1, target_rms=-20):
    """
    Apply Automatic Gain Control with dynamic time constants.
    
    Args:
        audio: Input audio signal
        frame_length: Length of processing frames in samples
        attack_time: Time constant for gain increase in seconds
        release_time: Time constant for gain decrease in seconds
        target_rms: Target RMS level in dB
    """
    # Convert target RMS from dB to linear
    target_rms_linear = 10 ** (target_rms / 20)
    
    # Calculate time constants in frames
    attack_coef = np.exp(-1 / (attack_time * TARGET_SAMPLE_RATE / frame_length))
    release_coef = np.exp(-1 / (release_time * TARGET_SAMPLE_RATE / frame_length))
    
    # Process audio in frames
    num_frames = len(audio) // frame_length
    gain = 1.0
    processed_audio = np.zeros_like(audio)
    
    for i in range(num_frames):
        frame = audio[i * frame_length:(i + 1) * frame_length]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        
        # Avoid division by zero
        if frame_rms < 1e-10:
            frame_rms = 1e-10
            
        # Calculate desired gain
        desired_gain = target_rms_linear / frame_rms
        
        # Apply time constants
        if desired_gain > gain:
            gain = attack_coef * gain + (1 - attack_coef) * desired_gain
        else:
            gain = release_coef * gain + (1 - release_coef) * desired_gain
            
        # Apply gain to frame
        processed_audio[i * frame_length:(i + 1) * frame_length] = frame * gain
    
    # Process remaining samples
    if len(audio) % frame_length != 0:
        remaining = audio[num_frames * frame_length:]
        processed_audio[num_frames * frame_length:] = remaining * gain
    
    return processed_audio


def get_melspectrogram(audio, use_agc=False):
    if use_agc:
        audio = apply_agc(audio)

    mstft = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_fft=WIN_LENGTH_SAMPLES,
        hop_length=HOP_LENGTH_SAMPLES,
        n_mels=N_FILTER_BANKS,
    )
    mstft_db = librosa.power_to_db(mstft, ref=np.max)

    return mstft_db


def get_datasets():
    recordings = get_recordings()
    raw_database, raw_training_set, raw_evaluation_set = question1(recordings)
    train_ds = create_dataset(raw_training_set)
    eval_ds = create_dataset(raw_evaluation_set)
    return raw_database, train_ds, eval_ds


class DTWModel(BaseEstimator, ClassifierMixin):
    def __init__(self, raw_database, threshold=None, normalization=None, use_agc=False, normalize_by_length=False):
        self.raw_database = raw_database
        self.threshold = threshold
        self.normalization = normalization
        self.use_agc = use_agc
        self.normalize_by_length = normalize_by_length
        self.melspec_database = {
            digit: np.array(get_melspectrogram(sample, use_agc=use_agc)) for digit, sample in raw_database.items()
        }

    def __call__(self, recording):

        melspec = get_melspectrogram(recording, use_agc=self.use_agc)
        sample = np.array(melspec)
        distances = np.zeros(len(self.melspec_database))
        for digit, ref_melspec in self.melspec_database.items():
            dtw_dist = dtw_forward(sample, ref_melspec)
            if self.normalize_by_length:
                dtw_dist = dtw_dist / (sample.shape[1] + ref_melspec.shape[1])
            distances[int(digit)] = dtw_dist
        return distances

    def fit(self, X, y=None):
        if self.threshold is None:
            distances = []
            for x in X:
                dist = self.__call__(x["recording"])
                distances.extend(dist)
            # Set threshold as mean + std of distances
            self.threshold = np.mean(distances) + np.std(distances)
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            distances = self.__call__(x["recording"])
            if np.min(distances) > self.threshold:
                predictions.append(-1)  # Reject if all distances above threshold
            else:
                predictions.append(np.argmin(distances))
        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        valid_mask = y_pred != -1
        if not np.any(valid_mask):
            return 0.0
        return accuracy_score(y[valid_mask], y_pred[valid_mask])

    def get_params(self, deep=True):
        return {
            'raw_database': self.raw_database,
            'threshold': self.threshold,
            'use_agc': self.use_agc,
            'normalize_by_length': self.normalize_by_length
        }

    def set_params(self, **params):
        for param, value in params.items():
            if param == 'raw_database':
                assert False
                # self.raw_database = value
                # self.melspec_database = {
                #     digit: np.array(get_melspectrogram(sample))
                #     for digit, sample in value.items()
                # }
            elif hasattr(self, param):
                setattr(self, param, value)
        return self

    def evaluate(self, ds):
        # Calculate accuracy by comparing predictions with ground truth
        assert self.threshold is not None
        predictions = self.predict(ds)
        correct = sum(1 for pred, true in zip(predictions, ds["digit"]) if pred == true)
        total = len(predictions)
        accuracy = correct / total
        return accuracy

    def plot_distance_matrix(self, train_ds, eval_ds):
        self.fit(train_ds)
        predictions = []
        distances = []
        for sample in tqdm(train_ds, desc="Processing samples"):
            result = self(sample["recording"])
            predictions.append(result)
            distances.append(result)

        # Organize distances into a 4x10 matrix
        num_speakers = 4
        num_digits = 10

        distance_matrix = np.zeros((num_speakers, num_digits))
        current_speaker = 0
        current_sample = 0

        for sample_distances in distances:
            for digit, dist in enumerate(sample_distances):
                distance_matrix[current_speaker, digit] = dist

            current_sample += 1
            if current_sample % num_digits == 0:
                current_speaker += 1

        return distance_matrix, self.evaluate(train_ds), self.evaluate(eval_ds), self.predict(eval_ds), eval_ds["digit"]


def main():
    raw_database, train_ds, eval_ds = get_datasets()

    os.makedirs("plots", exist_ok=True)
    
    # Create a 4x2 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(20, 32))
    fig.suptitle("DTW Model Comparison", fontsize=16)
    
    configs = [(False, False), (True, False), (False, True), (True, True)]
    
    for idx, (use_agc, normalize_by_length) in enumerate(configs):
        model = DTWModel(raw_database, use_agc=use_agc, normalize_by_length=normalize_by_length)
        distance_matrix, training_accuracy, eval_accuracy, eval_predictions, eval_true = model.plot_distance_matrix(
            train_ds,
            eval_ds,
        )
        
        # Plot distance matrix
        sns.heatmap(
            distance_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=range(10),
            yticklabels=[f"Speaker {i+1}" for i in range(4)],
            ax=axes[idx, 0]
        )
        title = f"DTW Distance Matrix\nAGC: {use_agc}, Length Norm: {normalize_by_length}\nTrain Acc: {training_accuracy:.2%}, Eval Acc: {eval_accuracy:.2%}"
        axes[idx, 0].set_title(title)
        axes[idx, 0].set_xlabel("Digits")
        axes[idx, 0].set_ylabel("Speakers")

        # Plot confusion matrix
        cm = confusion_matrix(eval_true, eval_predictions)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx, 1])
        axes[idx, 1].set_title(f"Confusion Matrix\nAGC: {use_agc}, Length Norm: {normalize_by_length}\nTrain Acc: {training_accuracy:.2%}, Eval Acc: {eval_accuracy:.2%}")
        axes[idx, 1].set_xlabel("Predicted")
        axes[idx, 1].set_ylabel("True")
        
        print(
            f"use_agc: {use_agc}, normalize_by_length: {normalize_by_length}, training_accuracy: {training_accuracy:.2%}, eval_accuracy: {eval_accuracy:.2%}"
        )
    
    plt.tight_layout()
    plt.savefig("plots/all_models_comparison.pdf")
    plt.close()
    print("Plots saved to plots/all_models_comparison.pdf")


if __name__ == "__main__":
    main()
