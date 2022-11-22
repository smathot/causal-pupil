"""
# Analysis utilities

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*

This module contains various constants and functions that are used in the main
analysis scripts.
"""
import mne; mne.set_log_level(False)
import eeg_eyetracking_parser as eet
from eeg_eyetracking_parser import braindecode_utils as bdu, \
    _eeg_preprocessing as epp
import numpy as np
from datamatrix import DataMatrix, convert as cnv, operations as ops, \
    functional as fnc, SeriesColumn
from mne.time_frequency import tfr_morlet
from matplotlib import pyplot as plt
from scipy.stats import mode
import logging; logging.basicConfig(level=logging.INFO, force=True)

FIXATION_TRIGGER = 1
CUE_TRIGGER = 2
INTERVAL_TRIGGER = 3
TARGET_TRIGGER = 4
RESPONSE_TRIGGER = 5
N_CHANNELS = 26
# Occipital
LEFT_CHANNELS = 'O1', 'P3', 'P7'
RIGHT_CHANNELS = 'O2', 'P4', 'P8'
MIDLINE_CHANNELS = 'Oz', 'POz', 'Pz'
# Parietal
# LEFT_CHANNELS = 'CP1',
# RIGHT_CHANNELS = 'CP2',
# MIDLINE_CHANNELS = 'Pz',
FACTORS = ['inducer', 'intensity', 'valid']
LABELS = [
    'unattended\ndim\nblue',
    'unattended\ndim\nred',
    'unattended\nbright\nblue',
    'unattended\nbright\nred',
    'attended\ndim\nblue',
    'attended\ndim\nred',
    'attended\nbright\nblue',
    'attended\nbright\nred'
]
ALPHA = .05
N_CONDITIONS = 8  # 3 factors with 2 levels each
ALL_CHANNELS = LEFT_CHANNELS + RIGHT_CHANNELS + MIDLINE_CHANNELS
FULL_FREQS = np.arange(4, 30, 1)
DELTA_FREQS = np.arange(.5, 4, .5)
THETA_FREQS = np.arange(4, 8, .5)
ALPHA_FREQS = np.arange(8, 12.5, .5)
BETA_FREQS = np.arange(13, 30, .5)
SUBJECTS = list(range(1, 34))
SUBJECTS.remove(7)   # technical error
SUBJECTS.remove(5)   # negative inducer effect
SUBJECTS.remove(18)  # negative inducer effect
DATA_FOLDER = 'data'
EPOCHS_KWARGS = dict(tmin=-.1, tmax=.75, picks='eeg',
                     preload=True, reject_by_annotation=False,
                     baseline=None)


def read_subject(subject_nr):
    return eet.read_subject(subject_nr=subject_nr,
                            saccade_annotation='BADS_SACCADE',
                            min_sacc_size=128)


def get_tgt_epoch(raw, events, metadata, channels):
    return eet.autoreject_epochs(
        raw, eet.epoch_trigger(events, TARGET_TRIGGER), tmin=-.1, tmax=.5,
        metadata=metadata, picks=channels)
    
    
def get_fix_epoch(raw, events, metadata, channels):
    return eet.autoreject_epochs(
        raw, eet.epoch_trigger(events, FIXATION_TRIGGER), tmin=-.5, tmax=2.5,
        metadata=metadata, picks=channels)


@fnc.memoize(persistent=True)
def get_merged_data():
    bigdm = DataMatrix(length=0)
    for subject_nr in SUBJECTS:
        print(f'Processing subject {subject_nr}')
        raw, events, metadata = read_subject(subject_nr)
        print('- left target epoch')
        left_tgt_epoch = get_tgt_epoch(raw, events, metadata, LEFT_CHANNELS)
        print('- right target epoch')
        right_tgt_epoch = get_tgt_epoch(raw, events, metadata, RIGHT_CHANNELS)
        print('- target epoch')
        tgt_epoch = get_tgt_epoch(raw, events, metadata, ALL_CHANNELS)
        print('- fix epoch')
        fix_epoch = get_fix_epoch(raw, events, metadata, ALL_CHANNELS)
        print('- time frequencies')
        alpha = tfr_morlet(fix_epoch, freqs=ALPHA_FREQS, n_cycles=4, n_jobs=-1,
                           return_itc=False, use_fft=True, average=False,
                           decim=4)
        alpha.crop(0, 2)
        theta = tfr_morlet(fix_epoch, freqs=THETA_FREQS, n_cycles=4, n_jobs=-1,
                           return_itc=False, use_fft=True, average=False,
                           decim=4)
        theta.crop(0, 2)
        full = tfr_morlet(fix_epoch, freqs=FULL_FREQS, n_cycles=4, n_jobs=-1,
                          return_itc=False, use_fft=True, average=False,
                          decim=4)
        full.crop(0, 2)
        print('- pupils')
        pupil_fix = eet.PupilEpochs(
            raw, eet.epoch_trigger(events, FIXATION_TRIGGER), tmin=0, tmax=2,
            metadata=metadata, baseline=None)
        pupil_target = eet.PupilEpochs(
            raw, eet.epoch_trigger(events, TARGET_TRIGGER), tmin=-.05, tmax=2,
            metadata=metadata)
        del raw
        dm = cnv.from_pandas(metadata)
        dm.pupil_fix = eet.epochs_to_series(dm, pupil_fix,
                                            baseline_trim=(-2, 2))
        dm.pupil_target = eet.epochs_to_series(dm, pupil_target,
                                               baseline_trim=(-2, 2))
        dm.left_erp = eet.epochs_to_series(dm, left_tgt_epoch)
        dm.right_erp = eet.epochs_to_series(dm, right_tgt_epoch)
        dm.erp = eet.epochs_to_series(dm, tgt_epoch)
        dm.alpha = eet.epochs_to_series(dm, alpha)
        dm.alpha = ops.z(dm.alpha)
        dm.theta = eet.epochs_to_series(dm, theta)
        dm.theta = ops.z(dm.theta)
        dm.tfr = eet.tfr_to_surface(dm, full)
        dm.tfr = ops.z(dm.tfr)
        bigdm <<= dm
    return bigdm


def decode_subject(subject_nr):
    read_subject_kwargs = dict(subject_nr=subject_nr,
                               saccade_annotation='BADS_SACCADE',
                               min_sacc_size=128)
    return bdu.decode_subject(read_subject_kwargs=read_subject_kwargs,
         factors=FACTORS, epochs_kwargs=EPOCHS_KWARGS,
         trigger=TARGET_TRIGGER, window_stride=1, window_size=200,
         n_fold=4, epochs=4)


def crossdecode_subject(subject_nr, from_factor, to_factor):
    read_subject_kwargs = dict(subject_nr=subject_nr,
                               saccade_annotation='BADS_SACCADE',
                               min_sacc_size=128)
    return bdu.decode_subject(read_subject_kwargs=read_subject_kwargs,
         factors=from_factor, crossdecode_factors=to_factor,
         epochs_kwargs=EPOCHS_KWARGS, trigger=TARGET_TRIGGER, window_stride=1,
         window_size=200, n_fold=4, epochs=4)


@fnc.memoize(persistent=True)
def blocked_decode_subject(subject_nr, factor, query1, query2):
    read_subject_kwargs = dict(subject_nr=subject_nr,
                               saccade_annotation='BADS_SACCADE',
                               min_sacc_size=128)
    train_data, train_labels, train_metadata = bdu.read_decode_dataset(
        read_subject_kwargs, factor, EPOCHS_KWARGS, TARGET_TRIGGER, query1)
    test_data, test_labels, test_metadata = bdu.read_decode_dataset(
        read_subject_kwargs, factor, EPOCHS_KWARGS, TARGET_TRIGGER, query2)
    clf = bdu.train(train_data, test_data)
    y_pred = clf.predict(test_data)
    resized_pred = y_pred.copy()
    resized_pred.resize(
            (len(test_data.datasets), len(test_data.datasets[0])))
    resized_pred = mode(resized_pred, axis=1)[0].flatten()
    y_true = [d.y[0] for d in test_data.datasets]
    return np.mean([p == t for p, t in zip(resized_pred, y_true)])


def test_confusion(dm, weights):
    scores = []
    for subject_nr, sdm in ops.split(dm.subject_nr):
        cm_prob = bdu.build_confusion_matrix(sdm.braindecode_label,
                                             sdm.braindecode_probabilities)
        score = 0
        for weight, f1, f2 in weights:
            score += weight * (cm_prob[LABELS.index(f1), LABELS.index(f2)] +
                              cm_prob[LABELS.index(f2), LABELS.index(f1)])
        scores.append(score)
    return scores


def statsplot(rm):
    rm = rm[:]
    rm.sign = SeriesColumn(depth=rm.p.depth)
    colors = ['red', 'green', 'blue', 'orange']
    for y, row in enumerate(rm[1:]):
        for linewidth, alpha in [(1, .05), (2, .01), (4, .005), (8, .001),]:
            row.sign[row.p >= alpha] = np.nan
            row.sign[row.p < alpha] = y
            plt.plot(row.sign, '-', label=f'{row.effect}, p < {alpha}',
                     linewidth=linewidth, color=colors[y])
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")


def select_ica(raw, events, metadata, exclude_component=0):
    
    global weights_dict
    print(f'running ica to exclude component {exclude_component}')
    @fnc.memoize(persistent=True)
    def run_ica(raw):
        return epp.run_ica(raw)
    # run_ica.clear()
    raw.info['bads'] = []
    ica = run_ica(raw)
    print('applying ica')
    ica.apply(raw, exclude=[exclude_component])
    weights = np.dot(ica.mixing_matrix_[:, exclude_component].T,
                     ica.pca_components_[:ica.n_components_])
    weights_dict = {ch_name: weight
                    for ch_name, weight in zip(ica.ch_names, weights)}
    print(f'weights: {weights_dict} (len={len(weights_dict)})')
    return raw, events, metadata


@fnc.memoize(persistent=True)
def ica_perturbation_decode(subject_nr, factor):
    read_subject_kwargs = dict(subject_nr=subject_nr,
                               saccade_annotation='BADS_SACCADE',
                               min_sacc_size=128)
    fdm = bdu.decode_subject(read_subject_kwargs=read_subject_kwargs,
        factors=factor, epochs_kwargs=EPOCHS_KWARGS,
        trigger=TARGET_TRIGGER, window_stride=1, window_size=200,
        n_fold=4, epochs=4)
    print(f'full-data accuracy: {fdm.braindecode_correct.mean}')
    perturbation_results = {}
    for exclude_component in range(N_CHANNELS):
        bdu.decode_subject.clear()
        dm = bdu.decode_subject(read_subject_kwargs=read_subject_kwargs,
            factors=factor, epochs_kwargs=EPOCHS_KWARGS,
            trigger=TARGET_TRIGGER, window_stride=1, window_size=200,
            n_fold=4, epochs=4,
            patch_data_func=lambda raw, events, metadata: select_ica(
                    raw, events, metadata, exclude_component))
        perturbation_results[exclude_component] = dm, weights_dict
        print(f'perturbation accuracy({exclude_component}): {dm.braindecode_correct.mean}')
    return fdm, perturbation_results