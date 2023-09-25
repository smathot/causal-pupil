"""
# Analysis utilities

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*

This module contains various constants and functions that are used in the main
analysis scripts.
"""
import random
import sys
import multiprocessing as mp
import mne; mne.set_log_level(False)
import eeg_eyetracking_parser as eet
from eeg_eyetracking_parser import braindecode_utils as bdu, \
    _eeg_preprocessing as epp
import numpy as np
import time_series_test as tst
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
# Only CP1 and CP2, which were the best channels
LEFT_CP = 'CP1',
RIGHT_CP = 'CP2',
MIDLINE_CP = tuple()
# Pz, POz, Oz
LEFT_OPM = 'Pz',
RIGHT_OPM = 'POz',
MIDLINE_OPM = 'Oz',
# Select a channel group for further processing. The main analyses focus on the
# the parietal group.
CHANNEL_GROUPS = 'parietal', 'occipital', 'frontal', 'central', 'CP', \
    'occipital-parietal-midline'
# Allow the channel group to be specified on the command line
for arg in sys.argv:
    if arg in CHANNEL_GROUPS:
        CHANNEL_GROUP = arg
        break
else:
    CHANNEL_GROUP = 'parietal'
if CHANNEL_GROUP == 'parietal':
    LEFT_CHANNELS = LEFT_PARIETAL
    RIGHT_CHANNELS = RIGHT_PARIETAL
    MIDLINE_CHANNELS = MIDLINE_PARIETAL
elif CHANNEL_GROUP == 'occipital':
    LEFT_CHANNELS = LEFT_OCCIPITAL
    RIGHT_CHANNELS = RIGHT_OCCIPITAL
    MIDLINE_CHANNELS = MIDLINE_OCCIPITAL
elif CHANNEL_GROUP == 'frontal':
    LEFT_CHANNELS = LEFT_FRONTAL
    RIGHT_CHANNELS = RIGHT_FRONTAL
    MIDLINE_CHANNELS = MIDLINE_FRONTAL
elif CHANNEL_GROUP == 'central':
    LEFT_CHANNELS = LEFT_CENTRAL
    RIGHT_CHANNELS = RIGHT_CENTRAL
    MIDLINE_CHANNELS = MIDLINE_CENTRAL
elif CHANNEL_GROUP == 'CP':
    LEFT_CHANNELS = LEFT_CP
    RIGHT_CHANNELS = RIGHT_CP
    MIDLINE_CHANNELS = MIDLINE_CP
elif CHANNEL_GROUP == 'occipital-parietal-midline':
    LEFT_CHANNELS = LEFT_OPM
    RIGHT_CHANNELS = RIGHT_OPM
    MIDLINE_CHANNELS = MIDLINE_OPM
else:
    raise ValueError(f'Invalid channel group: {CHANNEL_GROUP}')
ALL_CHANNELS = LEFT_CHANNELS + RIGHT_CHANNELS + MIDLINE_CHANNELS
FACTORS = ['inducer', 'bin_pupil', 'intensity', 'valid']
LABELS = ['00:blue:0:100:no',
          '01:blue:0:100:yes',
          '02:blue:0:255:no',
          '03:blue:0:255:yes',
          '04:blue:1:100:no',
          '05:blue:1:100:yes',
          '06:blue:1:255:no',
          '07:blue:1:255:yes',
          '08:red:0:100:no',
          '09:red:0:100:yes',
          '10:red:0:255:no',
          '11:red:0:255:yes',
          '12:red:1:100:no',
          '13:red:1:100:yes',
          '14:red:1:255:no',
          '15:red:1:255:yes']
ALPHA = .05
N_CONDITIONS = 16  # 4 factors with 2 levels each
FULL_FREQS = np.arange(4, 30, 1)
NOTCH_FREQS = np.exp(np.linspace(np.log(4), np.log(30), 15))
DELTA_FREQS = np.arange(.5, 4, .5)
THETA_FREQS = np.arange(4, 8, .5)
ALPHA_FREQS = np.arange(8, 12.5, .5)
BETA_FREQS = np.arange(13, 30, .5)
PERTURB_TIMES = [(-.1, .47),
                 (.18, .74)]
SUBJECTS = list(range(1, 34))
SUBJECTS.remove(7)   # technical error
SUBJECTS.remove(5)   # negative inducer effect
SUBJECTS.remove(18)  # negative inducer effect
DATA_FOLDER = 'data'
EPOCHS_KWARGS = dict(tmin=-.1, tmax=.75, picks='eeg',
                     preload=True, reject_by_annotation=False,
                     baseline=None)
# Plotting colors
RED = 'red'
BLUE = 'blue'
FACTOR_COLORS = {
    'inducer': '#B71C1C',
    'bin_pupil': '#4A148C',
    'intensity': '#263238',
    'valid': '#1B5E20'
}
# TFR plotting parameters
Y_FREQS = np.array([0, 4, 9, 25])
VMIN = -.2
VMAX = .2
CMAP = 'coolwarm'
# Plotting style
plt.style.use('default')
mpl.rcParams['font.family'] = 'Roboto Condensed'
# DATA_CHECKPOINT = 'checkpoints/18012023.dm'
DATA_CHECKPOINT = f'checkpoints/19072023-{CHANNEL_GROUP}.dm'


def read_subject(subject_nr):
    """A simple wrapper function that calls eet.read_subject() with the correct
    parameters.
    
    Parameters
    ----------
    subject_nr: int
    
    Returns
    -------
    tuple
        A (raw, events, metadata) tuple
    """
    return eet.read_subject(subject_nr=subject_nr,
                            saccade_annotation='BADS_SACCADE',
                            min_sacc_size=128)


def get_tgt_epoch(raw, events, metadata, channels=None, tmin=-.1, tmax=.5,
                  baseline=(None, 0)):
    """A simple wrapper function that uses eet.autoreject_epochs() to get
    an Epochs object around the target onset.
    
    Parameters
    ----------
    raw: Raw
    events: tuple
    metadata: DataFrame
    channels: list or None, optional
        A list of channel indices or None to select all channels
    tmin: float, optional
    tmax: float, optional
    baseline: tuple, optional
    
    Returns
    -------
    Epochs
    """
    return eet.autoreject_epochs(
        raw, eet.epoch_trigger(events, TARGET_TRIGGER), tmin=tmin, tmax=tmax,
        metadata=metadata, picks=channels, baseline=baseline,
        ar_kwargs=dict(n_jobs=8))
    
    
def get_fix_epoch(raw, events, metadata, channels=None):
    """A simple wrapper function that uses eet.autoreject_epochs() to get
    an Epochs object around the fixation onset.
    
    Parameters
    ----------
    raw: Raw
    events: tuple
    metadata: DataFrame
    channels: list or None, optional
        A list of channel indices or None to select all channels
    
    Returns
    -------
    Epochs
    """
    return eet.autoreject_epochs(
        raw, eet.epoch_trigger(events, FIXATION_TRIGGER), tmin=-.5, tmax=2.5,
        metadata=metadata, picks=channels, ar_kwargs=dict(n_jobs=8))
    

def get_morlet(epochs, freqs, crop=(0, 2), decim=8, n_cycles=2):
    """A simple wrapper function that uses tfr_morlet() to extract
    time-frequency data.
    
    Parameters
    ----------
    epochs: Epochs
    freqs: array
        An array of frequencies
    crop: tuple, optional
        A time window to crop after extracting the time-frequency data to
        reduce edge artifacts.
    decim: int, optional
        Downsampling factor to reduce memory consumption
    n_cycles: int, optional
        The number of cycles of the morlet wavelet
        
    Returns
    -------
    EpochsTFR
    """
    morlet = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, n_jobs=-1,
                        return_itc=False, use_fft=True, average=False,
                        decim=decim,
                        picks=np.arange(len(epochs.info['ch_names'])))
    morlet.crop(*crop)
    return morlet


def z_by_freq(col):
    """Performs z-scoring across trials, channels, and time points but 
    separately for each frequency.
    
    Parameters
    ----------
    col: MultiDimensionalColumn
    
    Returns
    -------
    MultiDimensionalColumn
    """
    zcol = col[:]
    for i in range(zcol.shape[2]):
        zcol._seq[:, :, i] = (
            (zcol._seq[:, :, i] - np.nanmean(zcol._seq[:, :, i]))
            / np.nanstd(zcol._seq[:, :, i])
        )
    return zcol


def subject_data(subject_nr):
    """Performs preprocessing for a single participant. This involves basic
    EEG preprocessing and subsequent epoching and extraction of TFR data. The
    result is a single DataMatrix that contains all information for final
    analysis.
    
    Parameters
    ----------
    subject_nr: int
    
    Returns
    -------
    DataMatrix
    """
    print(f'Processing subject {subject_nr}')
    raw, events, metadata = read_subject(subject_nr)
    raw['PupilSize'] = area_to_mm(raw['PupilSize'][0])
    dm = cnv.from_pandas(metadata)
    print('- eeg')
    tgt_epoch = get_tgt_epoch(raw, events, metadata)
    dm.tgt_erp = cnv.from_mne_epochs(tgt_epoch)
    tgt_tfr = get_morlet(
        get_tgt_epoch(raw, events, metadata, baseline=None, tmax=1),
        FULL_FREQS, crop=(0, .5), decim=4)
    dm.tgt_tfr = cnv.from_mne_tfr(tgt_tfr)
    dm.tgt_tfr = z_by_freq(dm.tgt_tfr)
    fix_epoch = get_fix_epoch(raw, events, metadata)
    fix_tfr = get_morlet(fix_epoch, FULL_FREQS)
    dm.fix_erp = cnv.from_mne_epochs(fix_epoch)
    dm.fix_tfr = cnv.from_mne_tfr(fix_tfr)
    dm.fix_tfr = z_by_freq(dm.fix_tfr)
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
    """Merges data for all participants into a single DataMatrix. Uses
    multiprocessing for performance.
    
    Returns
    -------
    DataMatrix
    """
    return fnc.stack_multiprocess(subject_data, SUBJECTS, processes=10)


def add_bin_pupil(raw, events, metadata):
    """Adds bin pupil to the metadata. This is a patch to allow decoding to
    take bin pupil as a decoding factor into account.
    
    Parameters
    ----------
    raw: Raw
    events: tuple
    metadata: DataFrame
    
    Returns
    -------
    tuple
        A (raw, events, metadata) tuple where bin_pupil has been added as a
        column to metadata.
    """
    # This adds the bin_pupil pseudo-factor to the data. This requires that
    # this has been generated already by `analyze.py`.
    dm = io.readtxt('output/bin-pupil.csv')
    dm = dm.subject_nr == metadata.subject_nr[0]
    metadata.loc[16:, 'bin_pupil'] = dm.bin_pupil
    dummy_factor = 192 * [0] + 192 * [1]
    random.shuffle(dummy_factor)
    metadata.loc[16:, 'dummy_factor'] = dummy_factor
    return raw, events, metadata


def decode_subject(subject_nr, factor=FACTORS):
    """A wrapper function around bdu.decode_subject() that performs overall
    decoding for one subject.
    
    Parameters
    ----------
    subject_nr: int
    factor: str or list, optional
    
    Returns
    -------
    DataMatrix
        See bdu.decode_subject()
    """
    read_subject_kwargs = dict(subject_nr=subject_nr,
                               saccade_annotation='BADS_SACCADE',
                               min_sacc_size=128)
    return bdu.decode_subject(
        read_subject_kwargs=read_subject_kwargs, factors=factor,
        epochs_kwargs=EPOCHS_KWARGS, trigger=TARGET_TRIGGER, window_stride=1,
        window_size=200, n_fold=4, epochs=4, patch_data_func=add_bin_pupil)


def crossdecode_subject(subject_nr, from_factor, to_factor):
    """A wrapper function around bdu.decode_subject() that performs
    cross-decoding for one subject.
    
    Parameters
    ----------
    subject_nr: int
    from_factor: str
        The factor to train on
    to_factor: str
        The factor to test on
    
    Returns
    -------
    DataMatrix
        See bdu.decode_subject()
    """
    read_subject_kwargs = dict(subject_nr=subject_nr,
                               saccade_annotation='BADS_SACCADE',
                               min_sacc_size=128)
    if 'bin_pupil' in (from_factor, to_factor):
        return bdu.decode_subject(
            read_subject_kwargs=read_subject_kwargs, factors=from_factor,
            crossdecode_factors=to_factor, epochs_kwargs=EPOCHS_KWARGS,
            trigger=TARGET_TRIGGER, window_stride=1, window_size=200, n_fold=4,
            epochs=4, patch_data_func=add_bin_pupil)
    return bdu.decode_subject(
        read_subject_kwargs=read_subject_kwargs, factors=from_factor,
        crossdecode_factors=to_factor, epochs_kwargs=EPOCHS_KWARGS,
        trigger=TARGET_TRIGGER, window_stride=1, window_size=200, n_fold=4,
        epochs=4)


@fnc.memoize(persistent=True)
def blocked_decode_subject(subject_nr, factor, query1, query2):
    """Decodes a factor for a single subject, using two different queries to
    separate the training and testing data.
    
    Parameters
    ----------
    subject_nr: int
    factor: str
    query1: str
        A pandas-style query to select the training set
    query2: str
        A pandas-style query to select the testing set
    
    Returns
    -------
    float
        Decoding accuracy
    """
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


def statsplot(rm):
    """A simple wrapper function that plots statistical values as a function of
    time and factors.
    
    Parameters
    ----------
    rm: DataMatrix
        A DataMatrix with statistical results as returned by time_series_test
        functions.
    """
    rm = rm[:]
    rm.sign = SeriesColumn(depth=rm.p.depth)
    colors = ['red', 'green', 'blue', 'orange']
    for y, row in enumerate(rm[1:]):
        for linewidth, alpha in [(1, .05), (2, .01), (4, .005), (8, .001)]:
            row.sign[row.p >= alpha] = np.nan
            row.sign[row.p < alpha] = y
            plt.plot(row.sign, '-', label=f'{row.effect}, p < {alpha}',
                     linewidth=linewidth, color=colors[y])
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")


def select_ica(raw, events, metadata, exclude_component=0):
    """A helper function that excludes (rather than selects, as the name 
    suggests) an independent component from the signal.
    
    Parameters
    ----------
    raw: Raw
    events: tuple
    metadata: DataFrame
    exclude_component: int, optional
        The index of the excluded independent component
        
    Returns
    -------
    tuple
        A (raw, events, metadata) tuple where the independent component has 
        been excluded from the `raw` object.
    """
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
    """Performs the ICA perturbation analysis.
    
    Parameters
    ----------
    subject_nr: int
    factor: str
    
    Returns
    -------
    tuple
        The first element of the tuple is a DataMatrix with the regular
        decoding results. The second element is a dict with independent
        component indices as keys and (dm, weights_dict) tuples as values.
        Here, dm is the DataMatrix with the decoding results after excluding
        the independent component, and weights_dict is a mapping with channel
        names as keys and weights (loading of the channel of the independent
        component) as values.
    """
    read_subject_kwargs = dict(subject_nr=subject_nr,
                               saccade_annotation='BADS_SACCADE',
                               min_sacc_size=128)
    fdm = bdu.decode_subject(
        read_subject_kwargs=read_subject_kwargs, factors=factor,
        epochs_kwargs=EPOCHS_KWARGS, trigger=TARGET_TRIGGER, window_stride=1,
        window_size=200, n_fold=4, epochs=4, patch_data_func=add_bin_pupil)
    print(f'full-data accuracy: {fdm.braindecode_correct.mean}')
    perturbation_results = {}
    for exclude_component in range(N_CHANNELS):
        bdu.decode_subject.clear()
        dm = bdu.decode_subject(
            read_subject_kwargs=read_subject_kwargs, factors=factor,
            epochs_kwargs=EPOCHS_KWARGS, trigger=TARGET_TRIGGER,
            window_stride=1, window_size=200, n_fold=4, epochs=4,
            patch_data_func=lambda raw, events, metadata: select_ica(
                    raw, events, metadata, exclude_component))
        perturbation_results[exclude_component] = dm, weights_dict
        print(f'perturbation accuracy({exclude_component}): '
              f'{dm.braindecode_correct.mean}')
    return fdm, perturbation_results


def notch_filter(raw, events, metadata, freq):
    """A helper function that excludes a frequency from the signal using a
    notch filter.
    
    Parameters
    ----------
    raw: Raw
    events: tuple
    metadata: DataFrame
    freq: float
        The frequency to remove.
        
    Returns
    -------
    tuple
        A (raw, events, metadata) tuple where the frequency has been removed
        from the `raw` object using a notch filter.
    """
    global weights_dict
    raw, events, metadata = add_bin_pupil(raw, events, metadata)
    width = np.exp(np.log(freq / 4))
    print(f'notch-filtering frequency band: {freq:.2f} / {width:.2f}')
    raw.notch_filter(freq, notch_widths=width, trans_bandwidth=width)
    return raw, events, metadata


@fnc.memoize(persistent=True)
def freq_perturbation_decode(subject_nr, factor):
    """Performs the frequency perturbation analysis.
    
    Parameters
    ----------
    subject_nr: int
    factor: str
    
    Returns
    -------
    tuple
        The first element of the tuple is a DataMatrix with the regular
        decoding results. The second element is a dict with frequencies
        as keys and the DataMatrix objects with the decoding results after 
        excluding the frequencies as values.
    """
    read_subject_kwargs = dict(subject_nr=subject_nr,
                               saccade_annotation='BADS_SACCADE',
                               min_sacc_size=128)
    fdm = bdu.decode_subject(
        read_subject_kwargs=read_subject_kwargs, factors=factor,
        epochs_kwargs=EPOCHS_KWARGS, trigger=TARGET_TRIGGER, window_stride=1,
        window_size=200, n_fold=4, epochs=4, patch_data_func=add_bin_pupil)
    print(f'full-data accuracy: {fdm.braindecode_correct.mean}')
    perturbation_results = {}
    for freq in NOTCH_FREQS:
        bdu.decode_subject.clear()
        dm = bdu.decode_subject(
            read_subject_kwargs=read_subject_kwargs, factors=factor,
            epochs_kwargs=EPOCHS_KWARGS, trigger=TARGET_TRIGGER,
            window_stride=1, window_size=200, n_fold=4, epochs=4,
            patch_data_func=lambda raw, events, metadata: notch_filter(
                    raw, events, metadata, freq))
        perturbation_results[freq] = dm
        print(f'perturbation accuracy({freq}): {dm.braindecode_correct.mean}')
    return fdm, perturbation_results


def area_to_mm(au):
    """Converts in arbitrary units to millimeters of diameter. This is specific
    to the recording set-up.
    
    Parameters
    ----------
    au: float
    
    Returns
    -------
    float
    """
    return -0.9904 + 0.1275 * au ** .5


def pupil_plot(dm, dv='pupil_target', **kwargs):
    """A simple wrapper function that plots pupil size over time.
    
    Parameters
    ----------
    dm: DataMatrix
    dv: str, optional
    **kwargs: dict, optional
    """
    tst.plot(dm, dv=dv, legend_kwargs={'loc': 'lower left'},
             **kwargs)
    x = np.linspace(12, 262, 6)
    t = [f'{int(s)}' for s in np.linspace(0, 1000, 6)]
    plt.xticks(x, t)
    plt.xlabel('Time (ms)')
    if dv == 'pupil_target':
        plt.axhline(0, linestyle=':', color='black')
        plt.ylim(-.6, .2)
    else:
        plt.ylim(2, 8)
    plt.xlim(0, 250)
    plt.ylabel('Baseline-corrected pupil size (mm)')


def erp_plot(dm, dv='lat_erp', ylim=None, **kwargs):
    """A simple wrapper function that plots ERPs.
    
    Parameters
    ----------
    dm: DataMatrix
    dv: str, optional
    ylim: float or None, optional
    **kwargs: dict, optional
    """
    tst.plot(dm, dv=dv, **kwargs)
    plt.xticks(np.arange(25, 150, 25), np.arange(0, 500, 100))
    plt.axvline(25, color='black', linestyle=':')
    plt.axhline(0, color='black', linestyle=':')
    plt.xlabel('Time (ms)')
    if ylim:
        plt.ylim(*ylim)


def tfr_plot(dm, dv):
    """A simple wrapper function that creates a multipanel TFR plot.
    
    Parameters
    ----------
    dm: DataMatrix
    dv: str, optional
    """
    plt.figure(figsize=(12, 4))
    plt.subplots_adjust(wspace=0)
    plt.subplot(141)
    tfr_red = (dm.inducer == 'red')[dv][...]
    tfr_blue = (dm.inducer == 'blue')[dv][...]
    plt.title('a) Induced Pupil Size (Large - Small)')
    plt.imshow(tfr_red - tfr_blue, aspect='auto', vmin=VMIN, vmax=VMAX,
               cmap=CMAP, interpolation='bicubic')
    plt.yticks(Y_FREQS, FULL_FREQS[Y_FREQS])
    plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
     
    plt.subplot(142)
    tfr_large = (dm.bin_pupil == 1)[dv][...]
    tfr_small = (dm.bin_pupil == 0)[dv][...]
    plt.title('b) Spontaneous Pupil Size (Large - Small)')
    plt.imshow(tfr_large - tfr_small, aspect='auto', vmin=VMIN, vmax=VMAX,
               cmap=CMAP, interpolation='bicubic')
    plt.gca().get_yaxis().set_visible(False)
    plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
    plt.xlabel('Time (ms)')
    
    plt.subplot(143)
    tfr_bright = (dm.intensity == 255)[dv].mean
    tfr_dim = (dm.intensity == 100)[dv].mean
    plt.title('c) Stimulus Intensity (Bright - Dim)')
    plt.imshow(tfr_bright - tfr_dim, aspect='auto', vmin=VMIN, vmax=VMAX,
               cmap=CMAP, interpolation='bicubic')
    plt.gca().get_yaxis().set_visible(False)
    plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
    plt.xlabel('Time (ms)')
    
    plt.subplot(144)
    tfr_attended = (dm.valid == 'yes')[dv].mean
    tfr_unattended = (dm.valid == 'no')[dv].mean
    plt.title('d) Covert Visual Attention (Attended - Unattended)')
    plt.imshow(tfr_attended - tfr_unattended, aspect='auto', vmin=VMIN,
               vmax=VMAX, cmap=CMAP, interpolation='bicubic')
    plt.gca().get_yaxis().set_visible(False)
    plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
    plt.xlabel('Time (ms)')
