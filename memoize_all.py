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
    raw, events, metadata = read_subject(subject_nr)
    left_tgt_epoch = get_tgt_epoch(raw, events, metadata, LEFT_CHANNELS)
    right_tgt_epoch = get_tgt_epoch(raw, events, metadata, RIGHT_CHANNELS)
    tgt_epoch = get_tgt_epoch(raw, events, metadata, ALL_CHANNELS)
    fix_epoch = get_fix_epoch(raw, events, metadata, ALL_CHANNELS)
    decode_subject(subject_nr)
    blocked_decode_subject(subject_nr, 'inducer')
    for f1, f2 in it.product(FACTORS, FACTORS):
        if f1 != f2:
            crossdecode_subject(subject_nr, f1, f2)
    for factor in FACTORS:
        ica_perturbation_decode(subject_nr, factor)
    print(f'finished subject {subject_nr}')


with mp.Pool() as pool:
    pool.map(memoize_subject, SUBJECTS)
