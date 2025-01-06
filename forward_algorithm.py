import os

import numpy as np
from matplotlib import pyplot as plt

from consts import BASE_DIR


class CTC:
    def __init__(self, sequence, alphabet_labels, pred, force_align=False):
        self.sequence, self.alphabet_labels, self.pred, self.force_align = sequence, alphabet_labels, pred, force_align

        self.blank_index, self.blank_label = list(alphabet_labels.items())[-1]

        self.alphabet_indices = {label: i for i, label in alphabet_labels.items()}

        self.padded_sequence_labels = [self.blank_label if i % 2 == 0 else self.sequence[(i - 1) // 2] for i in range(len(self.sequence) * 2 + 1)]
        self.padded_sequence_indices = [self.alphabet_indices[label] for label in self.padded_sequence_labels]

        self.log_alpha_matrix, self.backpointer_matrix, self.best_path, self.best_path_coordinates = None, None, None, None

    @staticmethod
    def __get_log_prob__(p):
        return -np.inf if p == 0 else np.log(p)

    @staticmethod
    def __get_prob__(lp):
        return 0 if lp == -np.inf else np.exp(lp)

    def __get_log_alpha__(self, t, s):
        p = self.pred[t, self.padded_sequence_indices[s]]

        if t == 0:
            log_alpha = self.__get_log_prob__(p if s <= 1 else 0)
            return (log_alpha, -1) if self.force_align else log_alpha

        alpha_tags = [self.__get_prob__(self.log_alpha_matrix[t - 1, s])]
        if s >= 1:
            alpha_tags.append(self.__get_prob__(self.log_alpha_matrix[t - 1, s - 1]))
        if s >= 2 and self.padded_sequence_indices[s] != self.padded_sequence_indices[s - 2]:
            alpha_tags.append(self.__get_prob__(self.log_alpha_matrix[t - 1, s - 2]))

        if self.force_align:
            log_alpha = self.__get_log_prob__(max(alpha_tags)) + self.__get_log_prob__(p)
            backpointer = np.argmax(alpha_tags)
            return log_alpha, backpointer
        else:
            log_alpha = self.__get_log_prob__(sum(alpha_tags)) + self.__get_log_prob__(p)
            return log_alpha

    def __forward_pass__(self):
        self.log_alpha_matrix = np.zeros((len(self.pred), len(self.padded_sequence_indices)), dtype=np.float32)
        if self.force_align:
            self.backpointer_matrix = np.zeros((len(self.pred), len(self.padded_sequence_indices)), dtype=np.int32)

        for t in range(len(self.pred)):
            for s in range(len(self.padded_sequence_indices)):
                if self.force_align:
                    self.log_alpha_matrix[t, s], self.backpointer_matrix[t, s] = self.__get_log_alpha__(t, s)
                else:
                    self.log_alpha_matrix[t, s] = self.__get_log_alpha__(t, s)

    def __backtrack__(self):
        if self.force_align:
            t = len(self.pred)
            self.best_path, self.best_path_coordinates = np.empty(t, dtype=str), np.empty(t, dtype=object)

            t -= 1
            if self.log_alpha_matrix[t, len(self.padded_sequence_indices) - 1] >= self.log_alpha_matrix[t, len(self.padded_sequence_indices) - 2]:
                best_s = len(self.padded_sequence_indices) - 1
            else:
                best_s = len(self.padded_sequence_indices) - 2
            self.best_path[t], self.best_path_coordinates[t] = self.alphabet_labels[self.padded_sequence_indices[best_s]], (t, best_s)

            while t > 0:
                best_s -= self.backpointer_matrix[t, best_s]
                t -= 1
                self.best_path[t], self.best_path_coordinates[t] = self.alphabet_labels[self.padded_sequence_indices[best_s]], (t, best_s)

    def plot(self, title, filename):
        self.__forward_pass__()
        self.__backtrack__()

        transpose_log_alpha_matrix = self.log_alpha_matrix.T

        rows, cols = transpose_log_alpha_matrix.shape
        fig_width = max(10, cols // 10)
        fig_height = max(5, rows // 10)
        plt.figure(figsize=(fig_width, fig_height))

        plt.imshow(transpose_log_alpha_matrix, aspect='auto', cmap='viridis')
        plt.suptitle(title)
        plt.xlabel('Timestep')
        plt.ylabel('Padded Sequence Label')
        plt.yticks(ticks=range(len(self.padded_sequence_labels)), labels=self.padded_sequence_labels)

        rows, cols = transpose_log_alpha_matrix.shape
        for i in range(rows):
            for j in range(cols):
                plt.text(j, i, f'{transpose_log_alpha_matrix[i, j]:.2f}', ha='center', va='center', color='white' if transpose_log_alpha_matrix[i, j] < 0.5 else 'black')

        if self.force_align:
            plt.title(''.join(self.best_path))
            for (x1, y1), (x2, y2) in zip(self.best_path_coordinates[:-1], self.best_path_coordinates[1:]):
                plt.arrow(x1, y1, x2 - x1, y2 - y1, color='red', head_width=0.2, head_length=0.2, length_includes_head=True)

        plt.savefig(os.path.join(BASE_DIR, filename))
        plt.close()
