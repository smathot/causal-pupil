"""
# Memoization entry point

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*

This script runs all time-consuming steps of the analyses. The intermediate
results are stored in the `.memoize` folder and can be copied to another
computer for further analysis.
"""

import multiprocessing as mp
from analysis_utils import *
import eeg_eyetracking_parser as eet
import itertools as it


def memoize_subject(subject_nr):
    print(f'memoizing subject {subject_nr}')
    # Comment/ uncomment to run decoding on the cleaned data
    # EPOCHS_KWARGS['reject_by_annotation'] = True
    subject_data(subject_nr)
    decode_subject(subject_nr)
    for f1, f2 in it.product(FACTORS, FACTORS):
        if f1 != f2:
            crossdecode_subject(subject_nr, f1, f2)
    for factor in FACTORS:
        ica_perturbation_decode(subject_nr, factor)
    print(f'finished subject {subject_nr}')


if __name__ == '__main__':
    with mp.Pool(2) as pool:
        pool.map(memoize_subject, SUBJECTS)
