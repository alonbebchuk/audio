import os
import pickle as pkl

from consts import SEQUENCE, ALPHABET_LABELS, PRED, BASE_DIR
from data_acquisition import get_recordings, question1, question2
from dtw import question3
from forward_algorithm import CTC

if __name__ == '__main__':
    os.makedirs("plots", exist_ok=True)
    recordings = get_recordings()
    # Q1
    class_representative, training_set, evaluation_set = question1(recordings)
    # Q2
    question2(recordings)
    # Q3
    question3()
    # Q5
    ctc = CTC(SEQUENCE, ALPHABET_LABELS, PRED, force_align=False)
    ctc.plot('Log Alpha Matrix', 'log_alpha_matrix.jpg')
    # Q6
    ctc = CTC(SEQUENCE, ALPHABET_LABELS, PRED, force_align=True)
    ctc.plot('Log Alpha Matrix Force Align', 'log_alpha_matrix_force_align.jpg')
    # Q7
    data = pkl.load(open(os.path.join(BASE_DIR, 'force_align.pkl'), 'rb'))
    sequence, alphabet_labels, pred = data['text_to_align'], data['label_mapping'], data['acoustic_model_out_probs']
    ctc = CTC(sequence, alphabet_labels, pred, force_align=True)
    ctc.plot('Log Alpha Matrix Force Align Pickle', 'log_alpha_matrix_force_align_pickle.jpg')
