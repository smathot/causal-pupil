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
import argparse


def memoize_subject(subject_nr):
    print(f'memoizing subject {subject_nr}')
    subject_data(subject_nr)
    try:
        decode_subject(subject_nr)
    except ValueError:
        print(f'cannot decode subject {subject_nr}')
        return
    for f1, f2 in it.product(FACTORS, FACTORS):
        if f1 != f2:
            crossdecode_subject(subject_nr, f1, f2)
    for factor in FACTORS:
        ica_perturbation_decode(subject_nr, factor)
    print(f'finished subject {subject_nr}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reject-by-annotation', action='store_true',
                        default=False)
    parser.add_argument('--n-process', action='store', default=2)
    parser.add_argument('--subjects', action='store', default='all')
    args = parser.parse_args()
    print(f'using {args.n_process} processes')
    if args.subjects == 'all':
        subjects = SUBJECTS
    else:
        subjects = [int(s) for s in args.subjects.split(',')]
    print(f'processing subjects: {subjects}')
    print(f'reject by annotation: {args.reject_by_annotation}')
    if args.reject_by_annotation:
        EPOCHS_KWARGS['reject_by_annotation'] = True
    if len(subjects) == 1:
        memoize_subject(subjects[0])
    else:
        with mp.Pool(args.n_process) as pool:
            pool.map(memoize_subject, subjects)
