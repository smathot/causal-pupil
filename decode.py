"""
# Decoding analysis

This script is the main decoding analysis for the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*


## Imports
"""
%load_ext autoreload
import mne; mne.set_log_level(False)
import numpy as np
from scipy.stats import ttest_1samp
from matplotlib import pyplot as plt; plt.style.use('default')
from datamatrix import DataMatrix, functional as fnc, series as srs
import eeg_eyetracking_parser as eet
from eeg_eyetracking_parser import braindecode_utils as bdu
from braindecode.visualization import plot_confusion_matrix
import itertools as it
from analysis_utils import *
from scipy.stats import ttest_1samp, ttest_rel
import pingouin as pg
import logging; logging.basicConfig(level=logging.INFO, force=True)


"""
## Control analysis

The code below should be uncommented and executed to perform the decoding after
rejecting epochs based on BAD annotations. This also requires excluding some
subjects who did not have any observations in one of the cells after trial
exclusion.

For the main analyses, decoding is performed on the uncleaned data, which is
why the cells below are commented out by default.
"""
# EPOCHS_KWARGS['reject_by_annotation'] = True
# SUBJECTS.remove(15)
# SUBJECTS.remove(13)


"""
## Overall decoding of full design matrix

Use four-fold crossvalidation to decode each of the eight conditions. This is
done separately per participant. The result is stored in a large DataMatrix
where each row corresponds to a trial with the decoding results added as 
`braindecode_*` columns.
"""
%autoreload
dm = DataMatrix()
for i, subject_nr in enumerate(SUBJECTS):
    print(f'decoding {subject_nr}')
    dm <<= decode_subject(subject_nr)


"""
Plot decoding confusion matrix based on correct predictions.
"""
cm_pred = bdu.build_confusion_matrix(dm.braindecode_label,
                                     dm.braindecode_prediction)
plot_confusion_matrix(cm_pred, LABELS, rotate_col_labels=45,
                      rotate_row_labels=45, figsize=(10, 10))
for factor, acc in zip(['overall'] + FACTORS,
                       bdu.summarize_confusion_matrix(FACTORS, cm_pred)):
    print(f'{factor}: {acc}%')
    
    
"""
Plot confusion matrix based on predicted 'probabilities'. Here, probabilities
aren't actual probabilities but rarther continuous values from each of the
eight output nodes, where the prediction is the highest value.
"""
cm_prob = bdu.build_confusion_matrix(dm.braindecode_label,
                                     dm.braindecode_probabilities)
plt.figure(figsize=(12, 12))
plt.imshow(cm_prob)
plt.xticks(range(N_CONDITIONS), LABELS)
plt.yticks(range(N_CONDITIONS), LABELS)


"""
Test cross-decoding based on probability matrix. Here the logic is that some
conditions are likely to be confused with each other, and therefore
probabilities in the corresponding cells of the confusion matrix as compared
to control combinations of conditions.
"""
%autoreload
weights_transfer_intensity_inducer = [
    (1, 'unattended\nbright\nblue', 'unattended\ndim\nred'),
    (1, 'attended\nbright\nblue', 'attended\ndim\nred'),
    (-1, 'attended\ndim\nblue', 'attended\nbright\nred'),
    (-1, 'unattended\ndim\nblue', 'unattended\nbright\nred')
]
weights_transfer_attention_inducer = [
    (1, 'attended\nbright\nblue', 'unattended\nbright\nred'),
    (1, 'attended\ndim\nblue', 'unattended\ndim\nred'),
    (-1, 'unattended\nbright\nblue', 'attended\nbright\nred'),
    (-1, 'unattended\ndim\nblue', 'attended\ndim\nred'),
]
weights_transfer_attention_intensity = [
    (1, 'attended\ndim\nblue', 'unattended\nbright\nblue'),
    (1, 'attended\ndim\nred', 'unattended\nbright\nred'),
    (-1, 'unattended\ndim\nblue', 'attended\nbright\nblue'),
    (-1, 'unattended\ndim\nred', 'attended\nbright\nred'),
]
print('Intensity <> inducer')
print(ttest_1samp(test_confusion(dm, weights_transfer_intensity_inducer), 0))
print('Attention <> inducer')
print(ttest_1samp(test_confusion(dm, weights_transfer_attention_inducer), 0))
print('Attention <> intensity')
print(ttest_1samp(test_confusion(dm, weights_transfer_attention_intensity), 0))


"""
## Control analysis: Account for temporal proximity in decoding of inducer

We decode the inducer factor by training the classfier on the first two blocks
and then testing it on the second two blocks. Because the inducer factor was
varied in a counterbalanced ABAB fashion, this eleminates any confounds due to
trials from the same inducer being closer to each other in time than trials
from different inducers.
"""
%autoreload
query1 = 'practice == "no" and block < 3'
query2 = 'practice == "no" and block >= 3'
# Train:  AABB----
# Test:   ----AABB
acc1 = [blocked_decode_subject(subject_nr, 'inducer', query1, query2)
        for subject_nr in SUBJECTS]
t, p = ttest_1samp(acc1, popmean=.5)
print('Control decoding of inducer (first half -> second half))')
print(f'M = {np.mean(acc1):.2f}, t = {t:.4f}, p = {p:.4f}')
# Train:  ----AABB
# Test:   AABB----
acc2 = [blocked_decode_subject(subject_nr, 'inducer', query2, query1)
        for subject_nr in SUBJECTS]
t, p = ttest_1samp(acc2, popmean=.5)
print('Control decoding of inducer (second half -> first half))')
print(f'M = {np.mean(acc2):.2f}, t = {t:.4f}, p = {p:.4f}')
# Combined across both splits
acc3 = .5 * (np.array(acc1) + np.array(acc2))
t, p = ttest_1samp(acc3, popmean=.5)
print('Control decoding of inducer (averaged))')
print(f'M = {np.mean(acc3):.2f}, t = {t:.4f}, p = {p:.4f}')


"""
The above analysis is overly conservative because temporal proximity now works
against decoding the inducer factor. So we do another control analysis, this
time splitting each block in half. Because the first and last half-block are
still unbalanced, these are removed from the test sets.
"""
%autoreload
# Train:  A-B-A-B-
# Test:   -A-B-A--
query_train1 = 'practice == "no" and ((trialid >= 16 and trialid < 64) or '\
    '(trialid >= 112 and trialid < 160) or (trialid >= 208 and trialid < 256)'\
    'or (trialid >= 304 and trialid < 352))'
query_test1 = 'practice == "no" and ((trialid >= 64 and trialid < 112) or '\
    '(trialid >= 160 and trialid < 208) or (trialid >= 256 and trialid < 304))'
# Train:  -A-B-A-B
# Test:   --B-A-B-
query_train2 = 'practice == "no" and ((trialid >= 64 and trialid < 112) or '\
    '(trialid >= 160 and trialid < 208) or (trialid >= 256 and trialid < 304)'\
    'or trialid >= 352)'
query_test2 = 'practice == "no" and ('\
    '(trialid >= 112 and trialid < 160) or (trialid >= 208 and trialid < 256)'\
    'or (trialid >= 304 and trialid < 352))'
acc1 = [blocked_decode_subject(subject_nr, 'inducer', query_train1,
                               query_test1) for subject_nr in SUBJECTS]
t, p = ttest_1samp(acc1, popmean=.5)
print('Control decoding of inducer (split blocks, skip first))')
print(f'M = {np.mean(acc1):.2f}, t = {t:.4f}, p = {p:.4f}')
acc2 = [blocked_decode_subject(subject_nr, 'inducer', query_train2,
                               query_test2) for subject_nr in SUBJECTS]
t, p = ttest_1samp(acc2, popmean=.5)
print('Control decoding of inducer (split blocks, skip last))')
print(f'M = {np.mean(acc2):.2f}, t = {t:.4f}, p = {p:.4f}')
# Combined across both splits
acc3 = .5 * (np.array(acc1) + np.array(acc2))
t, p = ttest_1samp(acc3, popmean=.5)
print('Control decoding of inducer (averaged)')
print(f'M = {np.mean(acc3):.2f}, t = {t:.4f}, p = {p:.4f}')


"""
## Cross-decoding

We perform cross decoding by training the model on one set of labels, and then
testing it on another set of labels. We do this separately for each of the
three condition combinations and each participant.
"""
%autoreload
cross_results = {}
for f1, f2 in it.product(FACTORS, FACTORS):
    if f1 == f2:
        continue
    print(f'{f1} → {f2}')
    dm = DataMatrix()
    for subject_nr in SUBJECTS:
        dm <<= crossdecode_subject(subject_nr, f1, f2)
    cross_results[f'{f1} {f2}'] = dm


"""
Statistically analyze the cross-decoding results.
"""
for transfer, dm in cross_results.items():
    print(transfer)
    acc = [sdm.braindecode_correct.mean
           for subject_nr, sdm in ops.split(dm.subject_nr)]
    # plt.plot(sorted(acc), 'o')
    # plt.axhline(.5)
    # plt.show()
    print(f'mean decoding accuracy: {np.mean(acc)}')
    print(ttest_1samp(acc, 0.5))


"""
## ICA perturbation

First load the data into an 3D array where subjects are the first dimension,
factors are the second dimension, and channels are the third dimension. Values
are weights that indicate how much each channel contributes to decoding for
a specific subject and factor.
"""
N_SUB = len(SUBJECTS)
grand_data = np.empty((N_SUB, 3, 26))
FACTORS = ['inducer', 'intensity', 'valid']
for i, subject_nr in enumerate(SUBJECTS[:N_SUB]):
    print(f'ICA perturbation analysis for {subject_nr}')
    for j, factor in enumerate(FACTORS):
        fdm, perturbation_results = ica_perturbation_decode(subject_nr, factor)
        blame_dict = {}
        for component, (sdm, weights_dict) in perturbation_results.items():
            punishment = fdm.braindecode_correct.mean - \
                sdm.braindecode_correct.mean
            for ch, w in weights_dict.items():
                if ch not in blame_dict:
                    blame_dict[ch] = 0
                blame_dict[ch] += punishment * abs(w)
        data = np.array(list(blame_dict.values()))
        grand_data[i, j] = data
# Read any subject to get the raw data. This is necessary for plotting the
# topomaps later on.
raw, events, metadata = read_subject(SUBJECTS[0])


"""
Visualize the topomaps. Weights are z-scored for each particpant independently
for visualization purposes (but not for statistical analysis).
"""
MAX = .75
COLORMAP = 'YlGnBu'

# Z-score the contributions per participant so that each participant
# contributes equally to the analysis
zdata = grand_data.copy()
for i in range(zdata.shape[0]):
    print(f'subject {SUBJECTS[i]}: M={zdata[i].mean()}, SD={zdata[i].std()}')
    zdata[i] -= zdata[i].mean()
    zdata[i] /= zdata[i].std()
print('Inducer')
mne.viz.plot_topomap(zdata[:, 0].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'], vmin=-MAX, vmax=MAX,
                     cmap=COLORMAP)
print('Intensity')
mne.viz.plot_topomap(zdata[:, 1].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'], vmin=-MAX, vmax=MAX,
                     cmap=COLORMAP)
print('Valid')
mne.viz.plot_topomap(zdata[:, 2].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'], vmin=-MAX, vmax=MAX,
                     cmap=COLORMAP)


"""
Conduct a repeated measures ANOVA to test the effect of channel and factor
on decoding weights. This is done on the non-z-scored data.
"""
CONDITIONS = ['inducer', 'intensity']
sdm = DataMatrix(length=len(SUBJECTS) * 3 * 26)
for row, (subject_nr, ch, f) in zip(
        sdm, it.product(range(len(SUBJECTS)), range(26), range(3))):
    row.subject_nr = SUBJECTS[subject_nr]
    row.channel = raw.info['ch_names'][ch]
    row.factor = FACTORS[f]
    row.weight = grand_data[subject_nr, f, ch]
aov = pg.rm_anova(dv='weight', within=['factor', 'channel'],
                  subject='subject_nr', data=sdm)
print(aov)


"""
Conduct paired-sample t-tests on individual channels between factors.
"""
for f1, f2, channel in it.product(FACTORS, FACTORS, raw.info['ch_names']):
    if f1 >= f2 or channel not in sdm.channel:
        continue
    fdm1 = (sdm.factor == f1) & (sdm.channel == channel)
    fdm2 = (sdm.factor == f2) & (sdm.channel == channel)
    t, p = ttest_rel(fdm1.weight, fdm2.weight)
    if p < .05:
        print('*')
    print(f'{f1} <> {f2}, {channel}, t={t:.4f}, p={p:.4f}')
