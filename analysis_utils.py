"""
# Analysis utilities

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*

This module contains various constants and functions that are used in the main
analysis scripts.
"""
import random
import multiprocessing as mp
import mne; mne.set_log_level(False)
import eeg_eyetracking_parser as eet
from eeg_eyetracking_parser import braindecode_utils as bdu, \
    _eeg_preprocessing as epp
import numpy as np
from datamatrix import DataMatrix, convert as cnv, operations as ops, \
    functional as fnc, SeriesColumn, io, MultiDimensionalColumn
from mne.time_frequency import tfr_morlet
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import mode
import logging; logging.basicConfig(level=logging.INFO, force=True)

FIXATION_TRIGGER = 1
CUE_TRIGGER = 2
INTERVAL_TRIGGER = 3
TARGET_TRIGGER = 4
RESPONSE_TRIGGER = 5
N_CHANNELS = 26
CHANNEL_GROUP = 'parietal'
CHANNEL_GROUPS = 'parietal', 'occipital', 'frontal', 'central'
# Occipital
LEFT_OCCIPITAL = 'O1',
RIGHT_OCCIPITAL = 'O2',
MIDLINE_OCCIPITAL = 'Oz',
# Parietal
LEFT_PARIETAL = 'P3', 'P7', 'CP1'
RIGHT_PARIETAL = 'P4', 'P8', 'CP2'
MIDLINE_PARIETAL = 'Pz', 'POz'
# Central
LEFT_CENTRAL = 'T7', 'C3'
RIGHT_CENTRAL = 'T8', 'C4'
MIDLINE_CENTRAL = 'Cz',
# Frontal
LEFT_FRONTAL = 'FC1', 'F3', 'F7', 'FP1'
RIGHT_FRONTAL = 'FC2', 'F4', 'F8', 'FP2'
MIDLINE_FRONTAL = 'Fz', 'FPz'
FACTORS = ['inducer', 'bin_pupil', 'intensity', 'valid']
LABELS = [
    'unattended\ndim\nsmall\nblue',
    'unattended\ndim\nlarge\nblue',
    'unattended\ndim\nsmall\nred',
    'unattended\ndim\nlarge\nred',
    'unattended\nbright\nsmall\nblue',
    'unattended\nbright\nlarge\nblue',
    'unattended\nbright\nsmall\nred',
    'unattended\nbright\nlarge\nred',
    'attended\ndim\nsmall\nblue',
    'attended\ndim\nlarge\nblue',
    'attended\ndim\nsmall\nred',
    'attended\ndim\nlarge\nred',
    'attended\nbright\nsmall\nblue',
    'attended\nbright\nlarge\nblue',
    'attended\nbright\nsmall\nred',
    'attended\nbright\nlarge\nred'
]
ALPHA = .05
N_CONDITIONS = 16  # 4 factors with 2 levels each
FULL_FREQS = np.arange(4, 30, 1)
NOTCH_FREQS = np.exp(np.linspace(np.log(4), np.log(30), 15))
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
# Plotting colors
RED = '#F44336'
BLUE = '#2196F3'
# Plotting style
mpl.rcParams['font.family'] = 'Roboto Condensed'
plt.style.use('default')


def read_subject(subject_nr):
    return eet.read_subject(subject_nr=subject_nr,
                            saccade_annotation='BADS_SACCADE',
                            min_sacc_size=128)


def get_tgt_epoch(raw, events, metadata, channels):
    return eet.autoreject_epochs(
        raw, eet.epoch_trigger(events, TARGET_TRIGGER), tmin=-.1, tmax=.5,
        metadata=metadata, picks=channels, ar_kwargs=dict(n_jobs=8))
    
    
def get_fix_epoch(raw, events, metadata, channels):
    return eet.autoreject_epochs(
        raw, eet.epoch_trigger(events, FIXATION_TRIGGER), tmin=-.5, tmax=2.5,
        metadata=metadata, picks=channels, ar_kwargs=dict(n_jobs=8))
    

def get_morlet(epochs, freqs):
    morlet = tfr_morlet(epochs, freqs=freqs, n_cycles=4, n_jobs=-1,
                        return_itc=False, use_fft=True, average=False, decim=4,
                        picks=np.arange(len(epochs.info['ch_names'])))
    morlet.crop(0, 2)
    return morlet


def subject_data(subject_nr):
    print(f'Processing subject {subject_nr}')
    raw, events, metadata = read_subject(subject_nr)
    raw['PupilSize'] = area_to_mm(raw['PupilSize'][0])
    dm = cnv.from_pandas(metadata)
    print('- eeg')
    tgt_epoch = get_tgt_epoch(raw, events, metadata, None)
    dm.tgt_erp = cnv.from_mne_epochs(tgt_epoch)
    fix_epoch = get_fix_epoch(raw, events, metadata, None)
    fix_tfr = get_morlet(fix_epoch, FULL_FREQS)
    dm.fix_erp = cnv.from_mne_epochs(tgt_epoch)
    dm.fix_tfr = cnv.from_mne_tfr(fix_tfr, ch_avg=True)
    dm.fix_tfr = ops.z(dm.fix_tfr)
    for label, lch, rch, mch in (
        ('occipital', LEFT_OCCIPITAL, RIGHT_OCCIPITAL, MIDLINE_OCCIPITAL),
        ('parietal', LEFT_PARIETAL, RIGHT_PARIETAL, MIDLINE_PARIETAL),
        ('central', LEFT_CENTRAL, RIGHT_CENTRAL, MIDLINE_CENTRAL),
        ('frontal', LEFT_FRONTAL, RIGHT_FRONTAL, MIDLINE_FRONTAL)
    ):
        print(f'- {label}')
        fix_epoch = get_fix_epoch(raw, events, metadata, lch + rch + mch)
        tfr = get_morlet(fix_epoch, FULL_FREQS)
        alpha = get_morlet(fix_epoch, ALPHA_FREQS)
        theta = get_morlet(fix_epoch, THETA_FREQS)
        dm[f'alpha_{label}'] = cnv.from_mne_tfr(alpha, ch_avg=True, 
                                                freq_avg=True)
        dm[f'theta_{label}'] = cnv.from_mne_tfr(theta, ch_avg=True,
                                                freq_avg=True)
        dm[f'tfr_{label}'] = cnv.from_mne_tfr(tfr, ch_avg=True)
        dm[f'alpha_{label}'] = ops.z(dm[f'alpha_{label}'])
        dm[f'theta_{label}'] = ops.z(dm[f'theta_{label}'])
        dm[f'tfr_{label}'] = ops.z(dm[f'tfr_{label}'])
    print('- pupils')
    pupil_fix = eet.PupilEpochs(
        raw, eet.epoch_trigger(events, FIXATION_TRIGGER), tmin=0, tmax=2,
        metadata=metadata, baseline=None)
    pupil_target = eet.PupilEpochs(
        raw, eet.epoch_trigger(events, TARGET_TRIGGER), tmin=-.05, tmax=2,
        metadata=metadata)
    del raw
    dm.pupil_fix = cnv.from_mne_epochs(pupil_fix, ch_avg=True)
    dm.pupil_target = cnv.from_mne_epochs(pupil_target, ch_avg=True)
    return dm


@fnc.memoize(persistent=True, key='merged-data')
def get_merged_data():
    return fnc.stack_multiprocess(subject_data, SUBJECTS, processes=8)


def add_bin_pupil(raw, events, metadata):
    # This adds the bin_pupil pseudo-factor to the data. This requires that
    # this has been generated already by `analyze.py`.
    dm = io.readtxt('output/bin-pupil.csv')
    dm = dm.subject_nr == metadata.subject_nr[0]
    metadata.loc[16:, 'bin_pupil'] = dm.bin_pupil
    dummy_factor = 192 * [0] + 192 * [1]
    random.shuffle(dummy_factor)
    metadata.loc[16:, 'dummy_factor'] = dummy_factor
    return raw, events, metadata


def decode_subject(subject_nr):
    read_subject_kwargs = dict(subject_nr=subject_nr,
                               saccade_annotation='BADS_SACCADE',
                               min_sacc_size=128)
    return bdu.decode_subject(read_subject_kwargs=read_subject_kwargs,
         factors=FACTORS, epochs_kwargs=EPOCHS_KWARGS,
         trigger=TARGET_TRIGGER, window_stride=1, window_size=200,
         n_fold=4, epochs=4, patch_data_func=add_bin_pupil)


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
    raw, events, metadata = add_bin_pupil(raw, events, metadata)
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
        n_fold=4, epochs=4, patch_data_func=add_bin_pupil)
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


def notch_filter(raw, events, metadata, freq):
    
    global weights_dict
    raw, events, metadata = add_bin_pupil(raw, events, metadata)
    print(f'notch-filtering frequency band: {freq}')
    raw.notch_filter(freq, notch_widths=2, trans_bandwidth=2)
    return raw, events, metadata


@fnc.memoize(persistent=True)
def freq_perturbation_decode(subject_nr, factor):
    read_subject_kwargs = dict(subject_nr=subject_nr,
                               saccade_annotation='BADS_SACCADE',
                               min_sacc_size=128)
    fdm = bdu.decode_subject(read_subject_kwargs=read_subject_kwargs,
        factors=factor, epochs_kwargs=EPOCHS_KWARGS,
        trigger=TARGET_TRIGGER, window_stride=1, window_size=200,
        n_fold=4, epochs=4)
    print(f'full-data accuracy: {fdm.braindecode_correct.mean}')
    perturbation_results = {}
    for freq in NOTCH_FREQS:
        bdu.decode_subject.clear()
        dm = bdu.decode_subject(read_subject_kwargs=read_subject_kwargs,
            factors=factor, epochs_kwargs=EPOCHS_KWARGS,
            trigger=TARGET_TRIGGER, window_stride=1, window_size=200,
            n_fold=4, epochs=4,
            patch_data_func=lambda raw, events, metadata: notch_filter(
                    raw, events, metadata, freq))
        perturbation_results[freq] = dm
        print(f'perturbation accuracy({freq}): {dm.braindecode_correct.mean}')
    return fdm, perturbation_results


def area_to_mm(au):
    return -0.9904 + 0.1275 * au ** .5
