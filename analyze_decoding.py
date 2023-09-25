"""
# Decoding analysis

This script is the main decoding analysis for the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*


## Imports
"""
import mne; mne.set_log_level(False)
import numpy as np
from scipy.stats import ttest_1samp
from matplotlib import pyplot as plt; plt.style.use('default')
from datamatrix import DataMatrix, functional as fnc, series as srs
import eeg_eyetracking_parser as eet
from eeg_eyetracking_parser import braindecode_utils as bdu
from braindecode.visualization import plot_confusion_matrix
import itertools as it
from sklearn.metrics import roc_auc_score
from analysis_utils import *
from scipy.stats import ttest_1samp, ttest_rel
import pingouin as pg
import logging; logging.basicConfig(level=logging.INFO, force=True)

EPOCHS_KWARGS['reject_by_annotation'] = True
#SUBJECTS.remove(15)
#SUBJECTS.remove(13)

"""
## Overall decoding of individual factors
"""
factor_results = {}
for factor in FACTORS:
    def decode_factor(subject_nr):
        return decode_subject(subject_nr, factor)
    factor_results[factor] = fnc.stack_multiprocess(
        decode_factor, SUBJECTS, processes=12)
    
"""
Summarize overall decoding
"""
auc_mean = []
auc_std = []
for factor, fdm in factor_results.items():
    auc = []
    for subject_nr, sdm in ops.split(fdm.subject_nr):
        auc.append(roc_auc_score(
            sdm.braindecode_label, sdm.braindecode_probabilities[:,1]))
    print(f'{factor}: {fdm.braindecode_correct.mean}, {np.mean(auc)}')
    auc_mean.append(np.mean(auc))
    auc_std.append(np.std(auc) * 1.96 / np.sqrt(len(auc)))
    print(ttest_1samp(auc, .5))
    

"""
Visualize overall decoding accuracy
"""
x = [0, 1, 2, 3]
plt.figure(figsize=(4, 4))
plt.axhline(.5, color='black', linestyle=':')
plt.bar(x, auc_mean, color='gray')
plt.errorbar(x, auc_mean, yerr=auc_std, linestyle='', color='black')
plt.ylim(0, 1)
plt.ylabel('Decoding accuracy (AUC)')
plt.xticks(x, ['Induced Pupil Size', 'Spontaneous Pupil Size',
               'Stimulus Intensity', 'Covert Visual Attention'],
           rotation=90)
plt.savefig('svg/decoding-barplot.svg')
plt.show()


"""
## Overall decoding of full design matrix

Use four-fold crossvalidation to decode each of the eight conditions. This is
done separately per participant. The result is stored in a large DataMatrix
where each row corresponds to a trial with the decoding results added as 
`braindecode_*` columns.
"""
dm = fnc.stack_multiprocess(decode_subject,
                            [s for s in SUBJECTS if s not in (13, 15)],
                            processes=12)

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
plt.savefig('svg/decoding-matrix.svg')
plt.show()


"""
## Control analysis: cross-decoding sanity check
"""
query1 = 'practice == "no" and intensity == 100'
query2 = 'practice == "no" and intensity == 255'
acc = [blocked_decode_subject(subject_nr, 'valid', query1, query2)
       for subject_nr in SUBJECTS]
t, p = ttest_1samp(acc, popmean=.5)
print('Control cross-decoding of validity')
print(f'M = {np.mean(acc):.2f}, t = {t:.4f}, p = {p:.4f}')

"""
## Control analysis: Account for temporal proximity in decoding of inducer

We decode the inducer factor by training the classfier on the first two blocks
and then testing it on the second two blocks. Because the inducer factor was
varied in a counterbalanced ABAB fashion, this eleminates any confounds due to
trials from the same inducer being closer to each other in time than trials
from different inducers.
"""
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
cross_results = {}
for f1, f2 in it.product(FACTORS, FACTORS):
    if f1 == f2:
        continue
    print(f'{f1} → {f2}')
    key = f'{f1} ←→ {f2}'
    if key not in cross_results:
        cross_results[key] = DataMatrix()
    for subject_nr in SUBJECTS:
        cross_results[key] <<= crossdecode_subject(subject_nr, f1, f2)

"""
Statistically analyze the cross-decoding results.
"""
for transfer, dm in cross_results.items():
    print(transfer)
    acc = [sdm.braindecode_correct.mean
           for subject_nr, sdm in ops.split(dm.subject_nr)]
    auc = []
    for subject_nr, sdm in ops.split(dm.subject_nr):
        auc.append(roc_auc_score(
            sdm.braindecode_label, sdm.braindecode_probabilities[:,1]))    
    plt.plot(sorted(auc), 'o')
    plt.axhline(.5)
    plt.show()
    print(acc)
    print(f'mean decoding accuracy: {np.mean(auc)}')
    print(ttest_1samp(auc, 0.5))

"""
## ICA perturbation

First load the data into an 3D array where subjects are the first dimension,
factors are the second dimension, and channels are the third dimension. Values
are weights that indicate how much each channel contributes to decoding for
a specific subject and factor.
"""
N_SUB = len(SUBJECTS)
grand_data = np.empty((N_SUB, len(FACTORS) + 1, 26))
for i, subject_nr in enumerate(SUBJECTS[:N_SUB]):
    print(f'ICA perturbation analysis for {subject_nr}')
    for j, factor in enumerate(FACTORS + ['dummy_factor']):
        fdm, perturbation_results = ica_perturbation_decode(subject_nr, factor)
        blame_dict = {}
        for component, (sdm, weights_dict) in perturbation_results.items():
            punishment = (fdm.braindecode_correct.mean -
                sdm.braindecode_correct.mean) / fdm.braindecode_correct.mean
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
# contributes equally to the analysis. We don't take the dummy factor into
# account for z-scoring.
zdata = grand_data.copy()
for i in range(zdata.shape[0]):
    print(f'subject {SUBJECTS[i]}: M={zdata[i].mean()}, SD={zdata[i].std()}')
    zdata[i] -= zdata[i, :4].mean()
    zdata[i] /= zdata[i, :4].std()
print('Inducer')
mne.viz.plot_topomap(zdata[:, 0].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'], vlim=(-MAX, MAX),
                     cmap=COLORMAP, show=False)
plt.savefig('svg/topomap-inducer.svg')
plt.show()
print('Bin pupil')
mne.viz.plot_topomap(zdata[:, 1].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'], vlim=(-MAX, MAX),
                     cmap=COLORMAP, show=False)
plt.savefig('svg/topomap-bin_pupil.svg')
plt.show()
print('Intensity')
mne.viz.plot_topomap(zdata[:, 2].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'], vlim=(-MAX, MAX),
                     cmap=COLORMAP, show=False)
plt.savefig('svg/topomap-intensity.svg')
plt.show()
print('Valid')
mne.viz.plot_topomap(zdata[:, 3].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'], vlim=(-MAX, MAX),
                     cmap=COLORMAP, show=False)
plt.savefig('svg/topomap-valid.svg')
plt.show()

"""
Visualize the topomap for the dummy factor. This should give a more-or-less
random distribution.
"""
print('Dummy factor')
mne.viz.plot_topomap(zdata[:, 4].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'],vlim=(-MAX, MAX),
                     cmap=COLORMAP)

"""
Conduct a repeated measures ANOVA to test the effect of channel and factor
on decoding weights. This is done on the non-z-scored data.
"""
sdm = DataMatrix(length=N_SUB * len(FACTORS) * 26)
for row, (subject_nr, ch, f) in zip(
        sdm, it.product(range(N_SUB), range(26), range(len(FACTORS)))):
    row.subject_nr = SUBJECTS[subject_nr]
    row.channel = raw.info['ch_names'][ch]
    row.factor = FACTORS[f]
    row.weight = grand_data[subject_nr, f, ch]
aov = pg.rm_anova(dv='weight', within=['factor', 'channel'],
                  subject='subject_nr', data=sdm)
print(aov)

for factor, fdm in ops.split(sdm.factor):
    print(f'Simple effect for {factor}')
    aov = pg.rm_anova(dv='weight', within=['channel'], subject='subject_nr',
                      data=fdm)
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
    if p >= .05:
        # print('*')
        continue
    print(f'{f1} <> {f2}, {channel}, t={t:.4f}, p={p:.4f}')

"""
## Frequency-band perturbation
"""
N_SUB = len(SUBJECTS)
grand_data = np.empty((N_SUB, len(FACTORS) + 1, len(NOTCH_FREQS)))
for i, subject_nr in enumerate(SUBJECTS[:N_SUB]):
    print(f'Frequency perturbation analysis for {subject_nr}')
    for j, factor in enumerate(FACTORS + ['dummy_factor']):
        fdm, perturbation_results = freq_perturbation_decode(subject_nr, factor)
        gm = fdm.braindecode_correct.mean
        data = np.array([r.braindecode_correct.mean
                        for r in perturbation_results.values()])
        data = (data - gm) / gm
        grand_data[i, j] = data

"""
Visualize the frequency-band perturbation
"""
from scipy.interpolate import make_interp_spline

plt.figure(figsize=(8, 4))
plt.axhline(0, linestyle=':', color='black')
x = np.linspace(NOTCH_FREQS.min(), NOTCH_FREQS.max(), 100)
zdata = -100 * grand_data.copy()
grand_mean = zdata.mean()
for i in range(len(SUBJECTS)):
    zdata[i] -= zdata[i].mean()
zdata += grand_mean
for j, factor in enumerate(FACTORS):
    mean = zdata[:, j, :].mean(axis=0)
    err = zdata[:, j, :].std(axis=0) / np.sqrt(len(SUBJECTS))
    spline_y = make_interp_spline(NOTCH_FREQS, mean)(x)
    spline_err = make_interp_spline(NOTCH_FREQS, err)(x)
    plt.fill_between(x, spline_y - spline_err, spline_y + spline_err, alpha=.2,
                     label=factor, color=FACTOR_COLORS[factor])
    plt.plot(NOTCH_FREQS, mean, 'o', color=FACTOR_COLORS[factor])
    plt.plot(x, spline_y, '-', color=FACTOR_COLORS[factor])
plt.xlim(NOTCH_FREQS.min(), NOTCH_FREQS.max())
plt.xlabel('Frequency (Hz)')
plt.ylabel('Decoding contribution (%)')
plt.xticks([4, 8, 13, 29])
plt.legend()
plt.savefig('svg/decoding-frequencies.svg')
plt.show()

"""
Visualize the frequency-band perturbations for individual subjects
"""
plt.title('Induced pupil size')
plt.axhline(0, linestyle='-', color='black')
plt.plot(NOTCH_FREQS, zdata[:, 0].T, 'o:', color='black', alpha=.2)
plt.show()
plt.title('Spontaneous pupil size')
plt.axhline(0, linestyle='-', color='black')
plt.plot(NOTCH_FREQS, zdata[:,1].T, 'o-', color='black', alpha=.2)
plt.show()
plt.title('Stimulus intensity')
plt.axhline(0, linestyle='-', color='black')
plt.plot(NOTCH_FREQS, zdata[:,2].T, 'o-', color='black', alpha=.2)
plt.show()
plt.title('Covert visual attention')
plt.axhline(0, linestyle='-', color='black')
plt.plot(NOTCH_FREQS, zdata[:,3].T, 'o-', color='black', alpha=.2)
plt.show()

"""
Conduct a repeated measures ANOVA to test the effect of frequency band and
factor on decoding weights. This is done on the non-z-scored data.
"""
sdm = DataMatrix(length=N_SUB * len(FACTORS) * len(NOTCH_FREQS))
for row, (subject_nr, freq, f) in zip(
        sdm, it.product(range(N_SUB), range(len(NOTCH_FREQS)),
                        range(len(FACTORS)))):
    row.subject_nr = SUBJECTS[subject_nr]
    row.freqs = NOTCH_FREQS[freq]
    row.factor = FACTORS[f]
    row.weight = grand_data[subject_nr, f, freq]
aov = pg.rm_anova(dv='weight', within=['factor', 'freqs'],
                  subject='subject_nr', data=sdm)
print(aov)
for factor, fdm in ops.split(sdm.factor):
    print(f'Simple effect for {factor}')
    aov = pg.rm_anova(dv='weight', within=['freqs'], subject='subject_nr',
                      data=fdm)
    print(aov)
