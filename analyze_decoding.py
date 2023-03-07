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
from analysis_utils import *
from scipy.stats import ttest_1samp, ttest_rel
import pingouin as pg
import logging; logging.basicConfig(level=logging.INFO, force=True)

EPOCHS_KWARGS['reject_by_annotation'] = True
SUBJECTS.remove(15)
SUBJECTS.remove(13)



"""
## Overall decoding of full design matrix

Use four-fold crossvalidation to decode each of the eight conditions. This is
done separately per participant. The result is stored in a large DataMatrix
where each row corresponds to a trial with the decoding results added as 
`braindecode_*` columns.
"""
dm = DataMatrix()
for i, subject_nr in enumerate(SUBJECTS):
    print(f'decoding {subject_nr}')
    dm <<= decode_subject(subject_nr)



# % output
# decoding 1
# decoding 2
# decoding 3
# decoding 4
# decoding 6
# decoding 8
# decoding 9
# decoding 10
# decoding 11
# decoding 12
# decoding 14
# decoding 16
# decoding 17
# decoding 19
# decoding 20
# decoding 21
# decoding 22
# decoding 23
# decoding 24
# decoding 25
# decoding 26
# decoding 27
# decoding 28
# decoding 29
# decoding 30
# decoding 31
# decoding 32
# decoding 33
# 
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



# % output
# overall: 42.28391934339852%
# inducer: 77.50531968791164%
# bin_pupil: 69.89563278954301%
# intensity: 69.01408450704226%
# valid: 86.1384132130915%
# ![](/home/sebastiaan/.opensesame/image_annotations/e0820e7eec4845aeb5e5418c119a8d9b.png)
# 
"""
Statistical analysis of decoding accuracy
"""
acc_results = {}
for subject_nr, sdm in ops.split(dm.subject_nr):
    cm_pred = bdu.build_confusion_matrix(sdm.braindecode_label,
                                         sdm.braindecode_prediction)
    for factor, acc in zip(['overall'] + FACTORS,
                           bdu.summarize_confusion_matrix(FACTORS, cm_pred)):
        if factor not in acc_results:
            acc_results[factor] = []
        acc_results[factor].append(acc)
        
x = np.arange(0, 5)
mean = np.empty(5)
err = np.empty(5)
for i, (factor, results) in enumerate(acc_results.items()):
    if factor == 'overall':
        chance = 6.25
    else:
        chance = 50
    t, p = ttest_1samp(results, 50)
    mean[i] = np.mean(results)
    err[i] = 1.96 * np.std(results) / np.sqrt(len(results))
    print(f'{factor}: {np.mean(results):.4f}% (chance: {chance}%), t = {t:.4f}, p = {p:.4f}')
    
    

# % output
# overall: 42.9020% (chance: 6.25%), t = -5.6811, p = 0.0000
# inducer: 77.5222% (chance: 50%), t = 16.2226, p = 0.0000
# bin_pupil: 70.5371% (chance: 50%), t = 18.0899, p = 0.0000
# intensity: 69.1389% (chance: 50%), t = 21.0678, p = 0.0000
# valid: 86.2847% (chance: 50%), t = 56.2981, p = 0.0000
# 
"""
Visualize decoding accuracy
"""
plt.figure(figsize=(4, 4))
plt.axhline(6.25, xmin=0, xmax=.22, linestyle=':', color='black')
plt.axhline(50, xmin=.22, xmax=1, linestyle=':', color='black')
plt.bar(x, mean, color='gray')
plt.errorbar(x, mean, yerr=err, linestyle='', color='black')
plt.ylim(0, 100)
plt.ylabel('Decoding accuracy (%)')
plt.xticks(x, ['Overall', 'Induced Pupil Size', 'Spontaneous Pupil Size',
               'Stimulus Intensity', 'Covert Visual Attention'],
           rotation=90)
plt.savefig('svg/decoding-barplot.svg')
plt.show()



# % output
# ![](/home/sebastiaan/.opensesame/image_annotations/2669af6aa7994888bb4e9b1d0ad63e56.png)
# 
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



# % output
# ([<matplotlib.axis.YTick at 0x7f6b44f51960>,
#   <matplotlib.axis.YTick at 0x7f6b44f520e0>,
#   <matplotlib.axis.YTick at 0x7f6b44f52ef0>,
#   <matplotlib.axis.YTick at 0x7f6b2a462e30>,
#   <matplotlib.axis.YTick at 0x7f6b2a4626b0>,
#   <matplotlib.axis.YTick at 0x7f6b2a461d20>,
#   <matplotlib.axis.YTick at 0x7f6b2a4617b0>,
#   <matplotlib.axis.YTick at 0x7f6b2a4f78b0>,
#   <matplotlib.axis.YTick at 0x7f6b2a461c60>,
#   <matplotlib.axis.YTick at 0x7f6b2a462f80>,
#   <matplotlib.axis.YTick at 0x7f6b2a460af0>,
#   <matplotlib.axis.YTick at 0x7f6b2a460190>,
#   <matplotlib.axis.YTick at 0x7f6b2a360040>,
#   <matplotlib.axis.YTick at 0x7f6b2a360940>,
#   <matplotlib.axis.YTick at 0x7f6b2a4607c0>,
#   <matplotlib.axis.YTick at 0x7f6b2a4f6020>],
#  [Text(0, 0, '00:blue:0:100:no'),
#   Text(0, 1, '01:blue:0:100:yes'),
#   Text(0, 2, '02:blue:0:255:no'),
#   Text(0, 3, '03:blue:0:255:yes'),
#   Text(0, 4, '04:blue:1:100:no'),
#   Text(0, 5, '05:blue:1:100:yes'),
#   Text(0, 6, '06:blue:1:255:no'),
#   Text(0, 7, '07:blue:1:255:yes'),
#   Text(0, 8, '08:red:0:100:no'),
#   Text(0, 9, '09:red:0:100:yes'),
#   Text(0, 10, '10:red:0:255:no'),
#   Text(0, 11, '11:red:0:255:yes'),
#   Text(0, 12, '12:red:1:100:no'),
#   Text(0, 13, '13:red:1:100:yes'),
#   Text(0, 14, '14:red:1:255:no'),
#   Text(0, 15, '15:red:1:255:yes')])
# ![](/home/sebastiaan/.opensesame/image_annotations/c3f629f5c7824d69bd27bb1616ad39f3.png)
#


"""
## Control analysis: cross-decoding sanity check


"""
query1 = 'practice == "no" and intensity == 100'
query2 = 'practice == "no" and intensity == 255'
acc = [blocked_decode_subject(subject_nr, 'valid', query1, query2)
       for subject_nr in SUBJECTS]
t, p = ttest_1samp(acc, popmean=.5)
print('Control cross-decoding of validity')
print(f'M = {np.mean(acc1):.2f}, t = {t:.4f}, p = {p:.4f}')


# % output
# Control cross-decoding of validity
# M = 0.58, t = 4.4980, p = 0.0001
#

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



# % output
# INFO:eeg_eyetracking_parser:37 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 17 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:37 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9778          0.5591              0.4839          0.8187   0.0006  6.7275
#       2             1.0000          0.2260              0.5376         0.9008  0.0005  4.1893
#       3            1.0000         0.1235             0.5323        0.9223  0.0002  4.2390
#       4            1.0000         0.1064             0.5269        0.9233  0.0000  4.1187
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:146 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 21 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:146 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 32 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.5000          0.2477              0.5000          4.0468   0.0006  3.6937
#       2             1.0000          0.0316              0.6753          1.0029   0.0005  3.4316
#       3            1.0000         0.0180             0.6558         0.9899   0.0002  3.4981
#       4            1.0000         0.0145             0.6558         0.9885   0.0000  3.5302
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9140          0.5265              0.5368          0.9524   0.0006  4.5796
#       2             0.9946          0.2212             0.5316        1.0111  0.0005  4.6107
#       3             1.0000          0.1187             0.5263        1.0431  0.0002  4.6783
#       4            1.0000         0.0931             0.5316        1.0432  0.0000  4.3534
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:150 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:150 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.5000          0.4465              0.5062          2.2808   0.0006  3.2012
#       2             1.0000          0.1433              0.5375          1.0537   0.0005  2.9834
#       3            1.0000         0.0799              0.5563          1.0339   0.0002  2.8295
#       4            1.0000         0.0703             0.5500        1.0374  0.0000  3.0751
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 6 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2689              0.4323          1.6762   0.0006  4.4198
#       2            1.0000         0.0498             0.4271        2.2058  0.0005  4.3004
#       3            1.0000         0.0218             0.4323        2.4693  0.0002  4.3571
#       4            1.0000         0.0174              0.4375         2.5016  0.0000  4.3466
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:223 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 4 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:223 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 37 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.5000          0.3874              0.5000          5.1032   0.0006  2.4662
#       2             0.9878          0.1182              0.5328          1.4216   0.0005  2.2748
#       3             1.0000          0.0609             0.4754         1.0603   0.0002  2.2756
#       4            1.0000         0.0541             0.4836         1.0527   0.0000  2.3679
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 4 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 10 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9946          0.3503              0.5104          1.0928   0.0006  4.6382
#       2             1.0000          0.0570              0.5729          1.0856   0.0005  4.2627
#       3            1.0000         0.0257             0.5625        1.1716  0.0002  4.4416
#       4            1.0000         0.0216             0.5677        1.2111  0.0000  4.3191
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:115 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:115 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 14 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9688          0.2350              0.6190          0.8859   0.0006  3.4965
#       2             1.0000          0.0161             0.5000        2.4926  0.0005  3.3870
#       3            1.0000         0.0053             0.5000        2.9248  0.0002  3.2765
#       4            1.0000         0.0034             0.5000        2.7587  0.0000  3.3277
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:11 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:11 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.1849              0.5000          3.7820   0.0006  4.4824
#       2            1.0000         0.0268             0.5000        4.5809  0.0005  4.4229
#       3            1.0000         0.0117             0.5000        4.8934  0.0002  4.3951
#       4            1.0000        0.0122            0.5000        4.6160  0.0000  4.4280
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 4 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9787          0.4436              0.5215          1.2572   0.0006  4.6642
#       2             1.0000          0.0878             0.4892        1.5043  0.0005  4.2203
#       3            1.0000         0.0338             0.4731        1.7571  0.0002  4.2436
#       4            1.0000         0.0268             0.4785        1.7455  0.0000  4.3493
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:38 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 8 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:38 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9483          0.2431              0.6522          1.0514   0.0006  4.2068
#       2             1.0000          0.0339              0.7989          0.5719   0.0005  4.1672
#       3            1.0000         0.0189             0.7826         0.5707   0.0002  4.0425
#       4            1.0000         0.0146             0.7989        0.5751  0.0000  4.0295
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9062          0.4737              0.5729          0.9861   0.0006  4.5987
#       2             1.0000          0.1786             0.5729         0.9380   0.0005  4.5054
#       3            1.0000         0.0839              0.5938         1.0208  0.0002  4.5124
#       4            1.0000         0.0671             0.5938        1.0209  0.0000  4.4612
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9149          0.3841              0.4734          1.5668   0.0006  4.3573
#       2             1.0000          0.0973             0.4149         1.3650   0.0005  4.2005
#       3            1.0000         0.0425             0.4043        1.5457  0.0002  4.1817
#       4            1.0000         0.0348             0.4043        1.5407  0.0000  4.5494
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9896          0.3173              0.4219          1.0930   0.0006  4.7399
#       2             1.0000          0.0667             0.4115        1.4678  0.0005  4.3723
#       3            1.0000         0.0325             0.3958        1.5495  0.0002  4.5910
#       4            1.0000         0.0283             0.4115        1.5154  0.0000  4.4429
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:5 epochs were dropped
# INFO:eeg_eyetracking_parser:5 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9896          0.4257              0.5000          1.2408   0.0006  4.4544
#       2             1.0000          0.1216             0.4896        1.4772  0.0005  4.5559
#       3            1.0000         0.0541             0.4896        1.5155  0.0002  4.5779
#       4            1.0000         0.0463             0.4896        1.5244  0.0000  4.4926
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:4 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:4 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2019              0.6198          1.0094   0.0006  4.4985
#       2            1.0000         0.0154              0.6927         1.0097  0.0005  4.5004
#       3            1.0000         0.0077             0.6302        1.1893  0.0002  4.5788
#       4            1.0000        0.0079            0.6562        1.0818  0.0000  4.3952
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:3 epochs were dropped
# INFO:eeg_eyetracking_parser:3 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.3294              0.7684          0.5514   0.0006  4.5664
#       2            1.0000         0.0570             0.7579        0.6378  0.0005  4.3203
#       3            1.0000         0.0248              0.7737         0.6427  0.0002  4.4554
#       4            1.0000         0.0222             0.7737        0.6431  0.0000  4.5174
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:6 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:6 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.1676              0.7500          0.5056   0.0006  4.5089
#       2            1.0000         0.0102              0.7865          0.4392   0.0005  4.4569
#       3            1.0000         0.0052              0.8177          0.3748   0.0002  4.5153
#       4            1.0000         0.0046              0.8229          0.3628   0.0000  4.3583
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:24 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 13 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:24 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2987              0.4792          1.2072   0.0006  4.5691
#       2            1.0000         0.0631              0.5156         1.2939  0.0005  4.3247
#       3            1.0000         0.0276              0.5521         1.3344  0.0002  4.2029
#       4            1.0000         0.0227             0.5469        1.3197  0.0000  4.3419
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.1988              0.5895          1.2417   0.0006  4.6754
#       2            1.0000         0.0156              0.6789          1.0177   0.0005  4.3348
#       3            1.0000         0.0092             0.6684        1.0437  0.0002  4.3481
#       4            1.0000         0.0066             0.6684        1.0450  0.0000  4.5018
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9948          0.2342              0.8646          0.3229   0.0006  4.5915
#       2             1.0000          0.0340              0.9010          0.2429   0.0005  4.4599
#       3            1.0000         0.0169              0.9062          0.2319   0.0002  4.4496
#       4            1.0000         0.0139             0.9062         0.2235   0.0000  4.4570
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2317              0.4785          1.5381   0.0006  4.2909
#       2            1.0000         0.0283              0.4946         1.7679  0.0005  4.2260
#       3            1.0000         0.0144             0.4839        1.8687  0.0002  4.2883
#       4            1.0000         0.0115             0.4785        1.8504  0.0000  4.4545
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.3082              0.4462          1.6775   0.0006  4.4695
#       2            1.0000         0.0507             0.4409        2.3570  0.0005  4.2865
#       3            1.0000         0.0215             0.4462        2.4051  0.0002  4.3317
#       4            1.0000         0.0187             0.4462        2.3804  0.0000  4.2845
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9844          0.4166              0.6771          0.5853   0.0006  4.5586
#       2             1.0000          0.0808              0.7865          0.5009   0.0005  4.3814
#       3            1.0000         0.0343             0.7812        0.5511  0.0002  4.6584
#       4            1.0000         0.0273             0.7812        0.5451  0.0000  4.4125
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:42 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 12 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:42 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 12 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9892          0.2429              0.2692          3.5264   0.0006  4.3873
#       2             1.0000          0.0544              0.2912         3.6202  0.0005  4.2759
#       3            1.0000         0.0302             0.2692        4.0046  0.0002  4.3016
#       4            1.0000         0.0256             0.2637        4.0535  0.0000  4.2088
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9844          0.3723              0.6198          0.6928   0.0006  4.6484
#       2             1.0000          0.0985              0.7188          0.6502   0.0005  4.4347
#       3            1.0000         0.0423             0.7135        0.6576  0.0002  4.4634
#       4            1.0000         0.0339             0.7083        0.6576  0.0000  4.3988
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:7 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:7 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9531          0.3354              0.5052          1.4050   0.0006  4.4230
#       2             1.0000          0.0572              0.5365          1.1609   0.0005  4.3096
#       3            1.0000         0.0258              0.5417         1.1690  0.0002  4.3696
#       4            1.0000         0.0207              0.5469         1.1748  0.0000  4.4277
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9427          0.4618              0.5316          0.9573   0.0006  4.5813
#       2             1.0000          0.1494             0.5316        1.0510  0.0005  4.4322
#       3            1.0000         0.0629              0.5579         1.0434  0.0002  4.4669
#       4            1.0000         0.0509             0.5579        1.0464  0.0000  4.3550
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# Control decoding of inducer (first half -> second half))
# M = 0.58, t = 2.7183, p = 0.0113
# INFO:eeg_eyetracking_parser:37 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:37 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 17 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.4924              0.5111          0.8635   0.0006  4.3680
#       2            1.0000         0.1723              0.5556         0.9572  0.0005  4.2116
#       3            1.0000         0.0880             0.5333        1.0374  0.0002  4.1492
#       4            1.0000         0.0747             0.5444        1.0337  0.0000  4.4256
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:146 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 32 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:146 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 21 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.5000          0.2756              0.5000          3.7707   0.0006  3.6061
#       2             1.0000          0.0452              0.6959          0.6759   0.0005  3.4213
#       3            1.0000         0.0230             0.6824        0.8006  0.0002  3.4572
#       4            1.0000         0.0178             0.6892        0.7686  0.0000  3.4293
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.8421          0.5573              0.4839          1.0060   0.0006  4.4192
#       2             0.9842          0.2591              0.4946         1.0623  0.0005  4.2908
#       3             0.9947          0.1376              0.5430         1.2572  0.0002  4.7728
#       4             1.0000          0.1148             0.5323        1.2216  0.0000  4.5542
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:150 epochs were dropped
# INFO:eeg_eyetracking_parser:150 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9625          0.5101              0.5000          0.9031   0.0006  3.0211
#       2             1.0000          0.1956             0.4884        0.9519  0.0005  2.7604
#       3            1.0000         0.0941              0.5233         1.0878  0.0002  2.8668
#       4            1.0000         0.0800             0.5000        1.0702  0.0000  3.0168
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 6 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9896          0.4890              0.3542          1.4600   0.0006  4.7205
#       2             0.9948          0.1583             0.2812        1.7740  0.0005  4.6416
#       3             1.0000          0.0738             0.3125        2.0467  0.0002  4.4202
#       4            1.0000         0.0585             0.3021        2.0189  0.0000  4.4834
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:223 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 37 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:223 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 4 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.5000          0.2324              0.5000          4.5015   0.0006  2.6058
#       2             1.0000          0.0334             0.4512         1.7165   0.0005  2.5557
#       3            1.0000         0.0166             0.4390        1.9762  0.0002  2.4315
#       4            1.0000         0.0147             0.4756        2.0693  0.0000  2.4347
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 10 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 4 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9948          0.4211              0.5489          0.8112   0.0006  4.7478
#       2             1.0000          0.1151             0.5489        0.8446  0.0005  4.4151
#       3            1.0000         0.0504              0.5598         0.8434  0.0002  4.2476
#       4            1.0000         0.0444              0.5652         0.8406  0.0000  4.2666
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:115 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 14 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:115 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9921          0.4418              0.5375          0.9150   0.0006  3.3717
#       2             1.0000          0.1295              0.5563         0.9333  0.0005  3.2284
#       3            1.0000         0.0682              0.5625         0.9244  0.0002  3.2552
#       4            1.0000         0.0562             0.5500        0.9470  0.0000  3.4395
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:11 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:11 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.7917          0.4574              0.3548          1.4799   0.0006  4.6110
#       2             1.0000          0.1207             0.3065        2.0155  0.0005  4.5662
#       3            1.0000         0.0535             0.2849        2.1891  0.0002  4.5882
#       4            1.0000         0.0450             0.2849        2.2163  0.0000  4.5541
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 4 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9785          0.4489              0.5000          1.0950   0.0006  4.7866
#       2             0.9946          0.1754              0.5053         1.1925  0.0005  4.5199
#       3             1.0000          0.0916             0.4681        1.2359  0.0002  4.5095
#       4            1.0000         0.0775             0.4628        1.2330  0.0000  4.4294
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:38 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:38 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 8 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2658              0.8046          0.5175   0.0006  4.3240
#       2            1.0000         0.0320              0.8161         0.6307  0.0005  4.2330
#       3            1.0000         0.0173             0.8161        0.6615  0.0002  4.1259
#       4            1.0000         0.0149             0.8161        0.6594  0.0000  4.0109
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9323          0.4391              0.5208          0.9866   0.0006  4.7765
#       2             0.9948          0.1613             0.5052        1.0110  0.0005  4.5315
#       3             1.0000          0.0795             0.4896        1.1317  0.0002  4.4946
#       4            1.0000         0.0651             0.4948        1.1121  0.0000  4.4379
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2201              0.4628          2.3494   0.0006  4.3978
#       2            1.0000         0.0145             0.4521        3.0551  0.0005  4.2736
#       3            1.0000         0.0077             0.4628        3.2995  0.0002  4.3665
#       4            1.0000         0.0064             0.4521        3.2680  0.0000  4.3385
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9844          0.4428              0.4219          1.1152   0.0006  4.5437
#       2             1.0000          0.1064             0.4219        1.5241  0.0005  4.5228
#       3            1.0000         0.0424             0.4010        1.6300  0.0002  4.3721
#       4            1.0000         0.0369             0.4115        1.6422  0.0000  4.4877
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:5 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:5 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9792          0.3744              0.4531          1.5351   0.0006  4.4627
#       2             1.0000          0.0861             0.4167        1.9155  0.0005  4.3410
#       3            1.0000         0.0356             0.4062        2.0300  0.0002  4.2998
#       4            1.0000         0.0297             0.4115        2.0247  0.0000  4.4175
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:4 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:4 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9948          0.2553              0.6458          0.8858   0.0006  4.5309
#       2             1.0000          0.0338              0.7708          0.5878   0.0005  4.4194
#       3            1.0000         0.0179              0.7760         0.5976  0.0002  4.4676
#       4            1.0000         0.0141              0.7917          0.5190   0.0000  4.3854
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:3 epochs were dropped
# INFO:eeg_eyetracking_parser:3 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9895          0.3505              0.7500          0.4985   0.0006  4.5658
#       2             1.0000          0.0382             0.6094        0.8478  0.0005  4.3401
#       3            1.0000         0.0173             0.6198        0.8514  0.0002  4.2976
#       4            1.0000         0.0140             0.6302        0.8123  0.0000  4.2811
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:6 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:6 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.1769              0.7474          0.4507   0.0006  4.3799
#       2            1.0000         0.0094             0.6737        0.6610  0.0005  4.3106
#       3            1.0000         0.0067             0.7158        0.6036  0.0002  4.3365
#       4            1.0000         0.0056             0.7105        0.6217  0.0000  4.3613
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:24 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:24 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 13 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9844          0.4236              0.5604          0.7989   0.0006  4.3585
#       2             1.0000          0.0978              0.6484          0.7283   0.0005  4.3394
#       3            1.0000         0.0319              0.6978          0.6346   0.0002  4.3212
#       4            1.0000         0.0277              0.7033         0.6412  0.0000  4.2548
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.3861              0.7394          0.5615   0.0006  4.4914
#       2            1.0000         0.0772              0.7447         0.5949  0.0005  4.2706
#       3            1.0000         0.0358             0.7394        0.6092  0.0002  4.2990
#       4            1.0000         0.0315             0.7394        0.6086  0.0000  4.2490
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.1583              0.8281          0.3569   0.0006  4.5696
#       2            1.0000         0.0101              0.8490         0.3759  0.0005  4.3452
#       3            1.0000         0.0062             0.7760        0.5264  0.0002  4.3910
#       4            1.0000         0.0051             0.8385        0.4202  0.0000  4.4094
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2621              0.4895          1.4305   0.0006  4.4119
#       2            1.0000         0.0243              0.4947         2.3714  0.0005  4.3117
#       3            1.0000         0.0129             0.4947        2.6929  0.0002  4.3102
#       4            1.0000         0.0107             0.4895        2.4989  0.0000  4.3058
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.8710          0.4894              0.5161          1.2252   0.0006  4.3134
#       2             1.0000          0.1580             0.4892         1.1689   0.0005  4.2425
#       3            1.0000         0.0782             0.4570        1.2111  0.0002  4.2544
#       4            1.0000         0.0653             0.4892        1.2447  0.0000  4.3237
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9583          0.3629              0.7292          0.5508   0.0006  4.5984
#       2             1.0000          0.0726              0.7396         0.5635  0.0005  4.7412
#       3            1.0000         0.0306              0.7552         0.5871  0.0002  4.6358
#       4            1.0000         0.0248              0.7604         0.5731  0.0000  4.6290
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:42 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 12 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:42 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 12 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2945              0.3011          1.4671   0.0006  4.2742
#       2            1.0000         0.0793             0.3011        1.8004  0.0005  4.2401
#       3            1.0000         0.0412             0.2903        1.8655  0.0002  4.4490
#       4            1.0000         0.0355              0.3065         1.8737  0.0000  4.2932
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9740          0.4180              0.6094          0.9333   0.0006  4.5468
#       2             1.0000          0.0997              0.6458          0.8157   0.0005  4.4589
#       3            1.0000         0.0434              0.6667         0.8880  0.0002  4.3410
#       4            1.0000         0.0363             0.6615        0.8742  0.0000  4.3706
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:7 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:7 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9896          0.4322              0.5000          0.8669   0.0006  4.5015
#       2             1.0000          0.1020             0.5000        0.9554  0.0005  4.3784
#       3            1.0000         0.0450             0.4948        1.0105  0.0002  4.5799
#       4            1.0000         0.0366             0.5000        1.0068  0.0000  4.3984
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9684          0.4932              0.5052          0.8875   0.0006  4.5972
#       2             1.0000          0.1500             0.4896        1.1210  0.0005  4.4032
#       3            1.0000         0.0652             0.4740        1.3017  0.0002  4.6990
#       4            1.0000         0.0558             0.4740        1.2923  0.0000  4.3711
# Control decoding of inducer (second half -> first half))
# M = 0.56, t = 1.9475, p = 0.0619
# Control decoding of inducer (averaged))
# M = 0.57, t = 2.4140, p = 0.0228
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# 
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



# % output
# INFO:eeg_eyetracking_parser:37 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 10 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:37 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 54 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9728          0.4343              0.5495          0.8752   0.0006  4.3429
#       2             1.0000          0.1535              0.5824         1.2785  0.0005  4.1435
#       3            1.0000         0.0781             0.5714        1.3326  0.0002  4.3801
#       4            1.0000         0.0693             0.5659        1.3980  0.0000  4.2320
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:146 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 40 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:146 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 46 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.5000          0.2175              0.5000          1.9934   0.0006  3.5470
#       2             1.0000          0.0248              0.7319          0.7408   0.0005  3.7587
#       3            1.0000         0.0140              0.7464         0.7483  0.0002  3.5178
#       4            1.0000         0.0120              0.7681          0.7038   0.0000  3.4840
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 49 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9624          0.5074              0.4628          0.9707   0.0006  4.3343
#       2             0.9946          0.2193             0.4521        1.0377  0.0005  4.2360
#       3             1.0000          0.1123              0.4894         1.0969  0.0002  4.2580
#       4            1.0000         0.0925              0.4947         1.0965  0.0000  4.5115
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:150 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 7 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:150 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 37 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.5000          0.4442              0.5000          3.1073   0.0006  3.1234
#       2             1.0000          0.1342              0.5259          0.9016   0.0005  2.7589
#       3            1.0000         0.0657             0.5259        0.9121  0.0002  2.8274
#       4            1.0000         0.0548             0.5000        0.9217  0.0000  2.8992
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 52 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9948          0.3816              0.4947          0.9189   0.0006  4.7045
#       2             1.0000          0.1017              0.5213         1.1676  0.0005  4.3788
#       3            1.0000         0.0477              0.5532         1.2097  0.0002  4.6689
#       4            1.0000         0.0425             0.5532        1.2178  0.0000  4.5877
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:223 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 12 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:223 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 6 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9412          0.3799              0.5227          1.2334   0.0006  1.8578
#       2             1.0000          0.1049              0.6136          1.1175   0.0005  1.6720
#       3            1.0000         0.0526             0.5909        1.1387  0.0002  1.7693
#       4            1.0000         0.0440              0.6364         1.1429  0.0000  1.7966
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 45 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.3287              0.3722          1.4517   0.0006  4.4945
#       2            1.0000         0.0599             0.3500        1.9577  0.0005  4.4132
#       3            1.0000         0.0269             0.3389        2.0375  0.0002  4.3521
#       4            1.0000         0.0242             0.3444        2.0188  0.0000  4.5104
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:115 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:115 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 38 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.3100              0.6765          0.8082   0.0006  3.5256
#       2            1.0000         0.0868              0.7132         0.9542  0.0005  3.2130
#       3            1.0000         0.0458             0.6985        1.0076  0.0002  3.4021
#       4            1.0000         0.0380             0.6838        1.0387  0.0000  3.5415
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:11 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:11 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 45 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9947          0.4202              0.6087          0.7944   0.0006  4.3262
#       2             1.0000          0.1209             0.5924        0.9912  0.0005  4.3123
#       3            1.0000         0.0609             0.5870        1.0638  0.0002  4.4098
#       4            1.0000         0.0482             0.5652        1.0738  0.0000  4.6148
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 45 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9681          0.4602              0.4341          1.1354   0.0006  4.3425
#       2             1.0000          0.1510             0.4341        1.2380  0.0005  4.2798
#       3            1.0000         0.0745             0.4341        1.2479  0.0002  4.2456
#       4            1.0000         0.0589             0.4341        1.2477  0.0000  4.3442
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:38 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:38 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 37 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2244              0.7692          0.6498   0.0006  4.2724
#       2            1.0000         0.0212              0.8333          0.5246   0.0005  4.0802
#       3            1.0000         0.0122             0.8269        0.6081  0.0002  4.1495
#       4            1.0000         0.0113             0.8269        0.6262  0.0000  4.0801
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9792          0.4560              0.5625          0.8504   0.0006  4.6600
#       2             0.9844          0.1680             0.4792        0.9793  0.0005  4.3066
#       3             1.0000          0.0862             0.5625        0.9672  0.0002  4.5331
#       4            1.0000         0.0686             0.5469        0.9685  0.0000  4.5066
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 47 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.8316          0.3924              0.5699          1.2090   0.0006  4.6343
#       2             1.0000          0.0667              0.5968          0.9358   0.0005  4.5216
#       3            1.0000         0.0300              0.6129         0.9578  0.0002  4.7518
#       4            1.0000         0.0243             0.6075        0.9610  0.0000  4.8871
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.4282              0.5885          0.7952   0.0006  4.5190
#       2            1.0000         0.1234             0.5885        0.8361  0.0005  4.5081
#       3            1.0000         0.0552              0.6094         0.8335  0.0002  4.4424
#       4            1.0000         0.0450              0.6302         0.8383  0.0000  4.4181
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:5 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:5 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9948          0.4339              0.4375          1.0600   0.0006  4.5284
#       2             1.0000          0.1313              0.5000         1.2577  0.0005  4.3223
#       3            1.0000         0.0576              0.5104         1.2850  0.0002  4.5006
#       4            1.0000         0.0466             0.5000        1.2863  0.0000  4.3935
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:4 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:4 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9896          0.2610              0.6615          0.9228   0.0006  4.4949
#       2             1.0000          0.0320              0.7552          0.5943   0.0005  4.4384
#       3            1.0000         0.0163             0.7448        0.6172  0.0002  4.4189
#       4            1.0000         0.0134             0.7552        0.6028  0.0000  4.4406
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:3 epochs were dropped
# INFO:eeg_eyetracking_parser:3 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 47 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9948          0.2993              0.7158          0.8753   0.0006  4.5841
#       2             1.0000          0.0372              0.7842          0.7110   0.0005  4.4248
#       3            1.0000         0.0184             0.7842        0.7992  0.0002  4.4868
#       4            1.0000         0.0152             0.7842        0.7671  0.0000  4.4649
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:6 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:6 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 49 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2081              0.9263          0.1885   0.0006  4.6155
#       2            1.0000         0.0130              0.9579          0.0849   0.0005  4.4892
#       3            1.0000         0.0064              0.9789          0.0598   0.0002  4.5211
#       4            1.0000         0.0052             0.9789        0.0639  0.0000  4.4093
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:24 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 4 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:24 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 57 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9948          0.3781              0.4780          0.7887   0.0006  4.3090
#       2             1.0000          0.0768              0.6703          0.6743   0.0005  4.4887
#       3            1.0000         0.0289              0.6923          0.6346   0.0002  4.4033
#       4            1.0000         0.0238             0.6923         0.6183   0.0000  4.4990
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 46 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9947          0.3948              0.8085          0.4212   0.0006  4.4645
#       2             1.0000          0.0864              0.8457          0.4169   0.0005  4.2413
#       3            1.0000         0.0485             0.8351        0.4191  0.0002  4.3429
#       4            1.0000         0.0389             0.8457         0.4061   0.0000  4.3673
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9896          0.2433              0.8542          0.2934   0.0006  4.5639
#       2             1.0000          0.0350              0.9427          0.1554   0.0005  4.3960
#       3            1.0000         0.0195             0.9375        0.1572  0.0002  4.3990
#       4            1.0000         0.0187             0.9323        0.1695  0.0000  4.3852
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 44 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2854              0.7446          0.7376   0.0006  4.4368
#       2            1.0000         0.0294              0.7772         0.8088  0.0005  4.2767
#       3            1.0000         0.0148              0.7826         0.8458  0.0002  4.2365
#       4            1.0000         0.0120             0.7772        0.8476  0.0000  4.2222
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 49 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9947          0.4839              0.4570          1.0910   0.0006  4.3862
#       2             1.0000          0.1658             0.4301        1.2932  0.0005  4.2629
#       3            1.0000         0.0850             0.4247        1.3265  0.0002  4.2675
#       4            1.0000         0.0703             0.3925        1.2727  0.0000  4.2166
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 49 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9167          0.3569              0.6979          0.7281   0.0006  4.4585
#       2             1.0000          0.0586             0.6927         0.6522   0.0005  4.4138
#       3            1.0000         0.0250             0.6979        0.6817  0.0002  4.3642
#       4            1.0000         0.0204             0.6979        0.7020  0.0000  4.3779
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:42 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:42 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 45 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.3699              0.6037          0.8142   0.0006  4.1059
#       2            1.0000         0.1070             0.5854        0.8371  0.0005  4.0064
#       3            1.0000         0.0599              0.6098         0.8309  0.0002  4.0314
#       4            1.0000         0.0480             0.6037        0.8445  0.0000  3.9916
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 47 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.7292          0.4318              0.5211          1.3488   0.0006  4.4105
#       2             1.0000          0.1172              0.6526          0.7823   0.0005  4.3447
#       3            1.0000         0.0522              0.6579          0.7544   0.0002  4.3292
#       4            1.0000         0.0433             0.6474         0.7543   0.0000  4.3228
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:7 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:7 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.3687              0.5521          0.8683   0.0006  4.5001
#       2            1.0000         0.0808              0.5885         0.9222  0.0005  4.4218
#       3            1.0000         0.0374              0.6094         0.9254  0.0002  4.4692
#       4            1.0000         0.0308             0.6094        0.9264  0.0000  4.4604
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 47 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9740          0.4245              0.5632          0.9081   0.0006  4.3923
#       2             1.0000          0.1169              0.6000         0.9171  0.0005  4.3170
#       3            1.0000         0.0538              0.6105          0.8943   0.0002  4.3661
#       4            1.0000         0.0448             0.6000        0.9097  0.0000  4.2829
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# Control decoding of inducer (split blocks, skip first))
# M = 0.64, t = 4.7627, p = 0.0001
# INFO:eeg_eyetracking_parser:37 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 8 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:37 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 35 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9670          0.4384              0.4756          1.2045   0.0006  4.0442
#       2             1.0000          0.1544              0.4817         1.2789  0.0005  3.9575
#       3            1.0000         0.0761             0.4573        1.2899  0.0002  3.9989
#       4            1.0000         0.0652             0.4634        1.2591  0.0000  4.0174
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:146 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 13 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:146 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.5000          0.3056              0.5000          3.5720   0.0006  2.7103
#       2             1.0000          0.0570              0.6136          0.7118   0.0005  2.5977
#       3            1.0000         0.0285              0.6591         0.7701  0.0002  2.6264
#       4            1.0000         0.0242             0.6477        0.7571  0.0000  2.5990
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 47 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9681          0.5641              0.5000          0.8659   0.0006  4.3955
#       2             1.0000          0.2603             0.4731        0.9881  0.0005  4.2273
#       3            1.0000         0.1452             0.4946        1.0423  0.0002  4.2130
#       4            1.0000         0.1197              0.5161         1.0298  0.0000  4.2173
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:150 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 4 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:150 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 14 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.5484          0.4581              0.4741          2.1140   0.0006  2.8983
#       2             0.9919          0.1703             0.4655         1.1925   0.0005  2.7972
#       3             1.0000          0.0919              0.4828         1.2571  0.0002  2.7105
#       4            1.0000         0.0751             0.4828        1.2367  0.0000  2.7095
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 4 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 49 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9628          0.3914              0.3750          1.2055   0.0006  4.4793
#       2             1.0000          0.1099              0.4635         1.3340  0.0005  4.3620
#       3            1.0000         0.0532              0.4740         1.3844  0.0002  4.3173
#       4            1.0000         0.0438             0.4583        1.3702  0.0000  4.3376
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:223 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 29 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:223 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 30 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.6765          0.2863              0.5294          2.2500   0.0006  2.4647
#       2             1.0000          0.0618              0.5392          1.1856   0.0005  2.7382
#       3            1.0000         0.0319              0.5588          0.9027   0.0002  2.6152
#       4            1.0000         0.0279             0.5588        0.9284  0.0000  2.4204
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 43 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9944          0.3622              0.4560          1.3567   0.0006  4.2430
#       2             1.0000          0.0759              0.4615          1.3078   0.0005  4.2292
#       3            1.0000         0.0351             0.4505        1.3468  0.0002  4.1896
#       4            1.0000         0.0294              0.4670         1.4103  0.0000  4.1970
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:115 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 15 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:115 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 40 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.5441          0.3426              0.4863          3.1584   0.0006  3.4003
#       2             1.0000          0.0828              0.5342          1.4985   0.0005  3.3689
#       3            1.0000         0.0448              0.5548          1.4490   0.0002  3.3482
#       4            1.0000         0.0336              0.5685          1.4068   0.0000  3.3201
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:11 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 3 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:11 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 46 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.3527              0.4947          1.0293   0.0006  4.3664
#       2            1.0000         0.0760             0.4840        1.2159  0.0005  4.3223
#       3            1.0000         0.0362             0.4628        1.2549  0.0002  4.3051
#       4            1.0000         0.0278             0.4628        1.2576  0.0000  4.3934
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:22 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 47 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9615          0.4911              0.4787          0.9662   0.0006  4.2798
#       2             1.0000          0.1881             0.4521        1.0916  0.0005  4.2730
#       3            1.0000         0.0962              0.4947         1.1819  0.0002  4.3327
#       4            1.0000         0.0789             0.4947        1.1739  0.0000  4.2578
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:38 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 7 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:38 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 43 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9941          0.3417              0.7967          0.4889   0.0006  4.1127
#       2             1.0000          0.0646              0.8626          0.3509   0.0005  4.0226
#       3            1.0000         0.0318             0.8352        0.3597  0.0002  4.0361
#       4            1.0000         0.0277             0.8516        0.3544  0.0000  4.0352
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 47 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9271          0.4504              0.5474          1.1220   0.0006  4.4919
#       2             1.0000          0.1470             0.4737        1.2497  0.0005  4.4790
#       3            1.0000         0.0744             0.4526        1.3630  0.0002  4.5311
#       4            1.0000         0.0599             0.4421        1.3506  0.0000  4.4652
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 49 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9946          0.4151              0.8421          0.4059   0.0006  4.5191
#       2             1.0000          0.1039              0.8474          0.3807   0.0005  4.3671
#       3            1.0000         0.0451             0.8474         0.3793   0.0002  4.3505
#       4            1.0000         0.0362             0.8474        0.3832  0.0000  4.3803
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9740          0.4347              0.5469          0.8389   0.0006  4.4885
#       2             1.0000          0.1590              0.6458          0.7999   0.0005  4.4492
#       3            1.0000         0.0787             0.6146        0.8255  0.0002  4.4894
#       4            1.0000         0.0622             0.6198        0.8278  0.0000  4.4963
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:5 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:5 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 47 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9844          0.4508              0.4316          0.9255   0.0006  4.4684
#       2             1.0000          0.1501             0.3684        1.0994  0.0005  4.3855
#       3            1.0000         0.0698              0.4737         1.0835  0.0002  4.4076
#       4            1.0000         0.0551              0.4789         1.0735  0.0000  4.4203
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:4 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:4 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2904              0.6823          0.5777   0.0006  4.5279
#       2            1.0000         0.0401              0.7188         0.6317  0.0005  4.5116
#       3            1.0000         0.0211             0.7031        0.7426  0.0002  4.5193
#       4            1.0000         0.0179             0.7031        0.7536  0.0000  4.4261
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:3 epochs were dropped
# INFO:eeg_eyetracking_parser:3 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.3095              0.7812          0.5565   0.0006  4.4588
#       2            1.0000         0.0395             0.7448        0.6851  0.0005  4.3898
#       3            1.0000         0.0185             0.7448        0.6678  0.0002  4.4314
#       4            1.0000         0.0149             0.7448        0.6516  0.0000  4.4248
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:6 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:6 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 47 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9947          0.2237              0.8211          0.4021   0.0006  4.4463
#       2             1.0000          0.0181              0.9632          0.1076   0.0005  4.3747
#       3            1.0000         0.0086             0.9632         0.1071   0.0002  4.3864
#       4            1.0000         0.0074             0.9632        0.1236  0.0000  4.4231
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:24 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 10 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:24 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 44 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.3458              0.4511          1.2925   0.0006  4.3222
#       2            1.0000         0.0628              0.4783          1.2242   0.0005  4.2725
#       3            1.0000         0.0275              0.5272          1.1203   0.0002  4.2553
#       4            1.0000         0.0217             0.5217        1.1281  0.0000  4.2522
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 47 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.2926              0.7234          0.5963   0.0006  4.4709
#       2            1.0000         0.0402              0.7287         0.6696  0.0005  4.4087
#       3            1.0000         0.0193              0.7340         0.6694  0.0002  4.4248
#       4            1.0000         0.0172              0.7500         0.6563  0.0000  4.5025
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             1.0000          0.1633              0.8958          0.2092   0.0006  4.5408
#       2            1.0000         0.0164              0.9271          0.2023   0.0005  4.5657
#       3            1.0000         0.0094              0.9323          0.1982   0.0002  4.5229
#       4            1.0000         0.0069             0.9271        0.2013  0.0000  4.4525
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:9 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9946          0.3459              0.6895          0.8036   0.0006  4.3921
#       2             1.0000          0.0621              0.7263         0.8038  0.0005  4.3327
#       3            1.0000         0.0282              0.7316         0.8577  0.0002  4.3905
#       4            1.0000         0.0256             0.6947        0.8601  0.0000  4.3803
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 5 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:16 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 49 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9409          0.3382              0.4947          1.5862   0.0006  4.4180
#       2             1.0000          0.0790             0.4211        1.7462  0.0005  4.3210
#       3            1.0000         0.0367             0.4474        1.7019  0.0002  4.2806
#       4            1.0000         0.0316             0.4526        1.7242  0.0000  4.3248
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9271          0.4248              0.6042          0.8333   0.0006  4.4523
#       2             1.0000          0.0890              0.7969          0.5197   0.0005  4.4516
#       3            1.0000         0.0409             0.7969        0.5440  0.0002  4.4530
#       4            1.0000         0.0330             0.7865        0.5236  0.0000  4.4172
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:42 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:42 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 49 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9024          0.4474              0.4286          1.1691   0.0006  4.0201
#       2             1.0000          0.1370              0.4560         1.2959  0.0005  3.9314
#       3            1.0000         0.0749              0.4615         1.3246  0.0002  3.9351
#       4            1.0000         0.0596              0.4725         1.3394  0.0000  4.0259
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:1 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9635          0.3769              0.5833          0.9558   0.0006  4.5847
#       2             1.0000          0.0853              0.6510          0.9534   0.0005  4.4833
#       3            1.0000         0.0403             0.6302        1.0117  0.0002  4.4666
#       4            1.0000         0.0332             0.6302        1.0333  0.0000  4.4589
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:7 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 2 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:7 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 48 observations to code 0 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9948          0.4206              0.4948          0.8775   0.0006  4.5431
#       2             1.0000          0.1031             0.4792        0.9953  0.0005  4.5373
#       3            1.0000         0.0495             0.4583        1.0650  0.0002  4.5035
#       4            1.0000         0.0423             0.4583        1.0649  0.0000  4.4695
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 1 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:2 epochs were dropped
# INFO:eeg_eyetracking_parser:adding 47 observations to code 1 to balance data
# INFO:eeg_eyetracking_parser:enabling cuda for gpu acceleration
#   epoch    train_accuracy    train_loss    valid_accuracy    valid_loss      lr     dur
# -------  ----------------  ------------  ----------------  ------------  ------  ------
#       1             0.9740          0.4490              0.5316          0.8957   0.0006  4.4862
#       2             1.0000          0.1472             0.5316        0.9617  0.0005  4.4480
#       3            1.0000         0.0649              0.5684         0.9505  0.0002  4.5187
#       4            1.0000         0.0547             0.5684        0.9497  0.0000  4.5215
# Control decoding of inducer (split blocks, skip last))
# M = 0.60, t = 3.4476, p = 0.0019
# Control decoding of inducer (averaged)
# M = 0.62, t = 4.2711, p = 0.0002
# /home/sebastiaan/Documents/Research/Projects/P0076 [Red blue inducers]/EEG/analysis_utils.py:218: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
#   resized_pred = mode(resized_pred, axis=1)[0].flatten()
# 
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
    key = f'{f1} ←→ {f2}' if f2 > f1 else f'{f2} ←→ {f1}'
    if key not in cross_results:
        cross_results[key] = DataMatrix()
    for subject_nr in SUBJECTS:
        cross_results[key] <<= crossdecode_subject(subject_nr, f1, f2)



# % output
# inducer → bin_pupil
# inducer → intensity
# inducer → valid
# bin_pupil → inducer
# bin_pupil → intensity
# bin_pupil → valid
# intensity → inducer
# intensity → bin_pupil
# intensity → valid
# valid → inducer
# valid → bin_pupil
# valid → intensity
# 
"""
Statistically analyze the cross-decoding results.
"""
for transfer, dm in cross_results.items():
    print(transfer)
    acc = [sdm.braindecode_correct.mean
           for subject_nr, sdm in ops.split(dm.subject_nr)]
    plt.plot(sorted(acc), 'o')
    plt.axhline(.5)
    plt.show()
    print(f'mean decoding accuracy: {np.mean(acc)}')
    print(ttest_1samp(acc, 0.5))



# % output
# bin_pupil ←→ inducer
# ![](/home/sebastiaan/.opensesame/image_annotations/2d238707c21c4e71a8fc191cc46f1c99.png)
# mean decoding accuracy: 0.49857138656166633
# Ttest_1sampResult(statistic=-0.36061622452519676, pvalue=0.7211924264300917)
# inducer ←→ intensity
# ![](/home/sebastiaan/.opensesame/image_annotations/78ac4958831c4e50afde3bd41bdc94f0.png)
# mean decoding accuracy: 0.4990609495705514
# Ttest_1sampResult(statistic=-0.31617506844800203, pvalue=0.7543010055006425)
# inducer ←→ valid
# ![](/home/sebastiaan/.opensesame/image_annotations/5569fd639d1b4f5ab5c7489ce0a74122.png)
# mean decoding accuracy: 0.49492687579299766
# Ttest_1sampResult(statistic=-1.13600213501976, pvalue=0.26593887733472377)
# bin_pupil ←→ intensity
# ![](/home/sebastiaan/.opensesame/image_annotations/8cc6a9d2c27b4833b4344867aa801836.png)
# mean decoding accuracy: 0.4924343045670372
# Ttest_1sampResult(statistic=-1.3271367367296856, pvalue=0.19557760229892251)
# bin_pupil ←→ valid
# ![](/home/sebastiaan/.opensesame/image_annotations/651bb138ec6c46de8dcc8269dbbf01aa.png)
# mean decoding accuracy: 0.4888497592135028
# Ttest_1sampResult(statistic=-1.5398361089318906, pvalue=0.13523938773242444)
# intensity ←→ valid
# ![](/home/sebastiaan/.opensesame/image_annotations/9f1a63176beb4ca4ac0364ca33720b8f.png)
# mean decoding accuracy: 0.5009087662044155
# Ttest_1sampResult(statistic=0.48053877319897975, pvalue=0.6347151535169571)
# 
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



# % output
# ICA perturbation analysis for 1
# ICA perturbation analysis for 2
# ICA perturbation analysis for 3
# ICA perturbation analysis for 4
# ICA perturbation analysis for 6
# ICA perturbation analysis for 8
# ICA perturbation analysis for 9
# ICA perturbation analysis for 10
# ICA perturbation analysis for 11
# ICA perturbation analysis for 12
# ICA perturbation analysis for 14
# ICA perturbation analysis for 16
# ICA perturbation analysis for 17
# ICA perturbation analysis for 19
# ICA perturbation analysis for 20
# ICA perturbation analysis for 21
# ICA perturbation analysis for 22
# ICA perturbation analysis for 23
# ICA perturbation analysis for 24
# ICA perturbation analysis for 25
# ICA perturbation analysis for 26
# ICA perturbation analysis for 27
# ICA perturbation analysis for 28
# ICA perturbation analysis for 29
# ICA perturbation analysis for 30
# ICA perturbation analysis for 31
# ICA perturbation analysis for 32
# ICA perturbation analysis for 33
# 
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
                     names=raw.info['ch_names'], vmin=-MAX, vmax=MAX,
                     cmap=COLORMAP, show=False)
plt.savefig('svg/topomap-inducer.svg')
plt.show()
print('Bin pupil')
mne.viz.plot_topomap(zdata[:, 1].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'], vmin=-MAX, vmax=MAX,
                     cmap=COLORMAP, show=False)
plt.savefig('svg/topomap-bin_pupil.svg')
plt.show()
print('Intensity')
mne.viz.plot_topomap(zdata[:, 2].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'], vmin=-MAX, vmax=MAX,
                     cmap=COLORMAP, show=False)
plt.savefig('svg/topomap-intensity.svg')
plt.show()
print('Valid')
mne.viz.plot_topomap(zdata[:, 3].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'], vmin=-MAX, vmax=MAX,
                     cmap=COLORMAP, show=False)
plt.savefig('svg/topomap-valid.svg')
plt.show()



# % output
# subject 1: M=0.008488775504431065, SD=0.017714035737727996
# subject 2: M=7.378393023278037e-17, SD=7.391865860734718e-17
# subject 3: M=2.6050236541847242e-17, SD=5.372889065455254e-17
# subject 4: M=1.5981162463068407e-17, SD=3.727100396555696e-17
# subject 6: M=0.005776775219309999, SD=0.013604746775171268
# subject 8: M=-0.004595124255419281, SD=0.0141892230427253
# subject 9: M=0.013878214850293776, SD=0.016036110447857736
# subject 10: M=0.0011946300445868148, SD=0.0029942378505064854
# subject 11: M=0.012630317219291151, SD=0.02612858858049734
# subject 12: M=0.009932732190126919, SD=0.012422238693254083
# subject 14: M=-0.0015125045972246552, SD=0.006630367034309227
# subject 16: M=0.003440011692427109, SD=0.003294363123100682
# subject 17: M=5.157229865011403e-17, SD=7.080079642839918e-17
# subject 19: M=0.0010039598922237096, SD=0.003423980617404151
# subject 20: M=-0.0026190769467237945, SD=0.006569399702855672
# subject 21: M=0.01919773251462534, SD=0.02144667874251794
# subject 22: M=0.001160495883015923, SD=0.0013962084546025027
# subject 23: M=0.023931100324720324, SD=0.026197597882089307
# subject 24: M=0.0003762792826997485, SD=0.009054635581301108
# subject 25: M=0.016206005404100342, SD=0.021786642511470618
# subject 26: M=5.410011097553924e-17, SD=6.260713150763536e-17
# subject 27: M=0.0038793736348288624, SD=0.016296320568083392
# subject 28: M=1.061027637428436e-17, SD=3.305725603253164e-17
# subject 29: M=0.0023567024217978796, SD=0.0030482583762687027
# subject 30: M=0.0014893827563300388, SD=0.0027261760158186596
# subject 31: M=0.016935589988431615, SD=0.022174514106683668
# subject 32: M=0.013360265474919683, SD=0.009406819580195014
# subject 33: M=-0.005747673821862308, SD=0.02157722536047241
# Inducer
# /tmp/ipykernel_355863/2580720971.py:13: FutureWarning: The "vmin" and "vmax" parameters are deprecated and will be removed in version 1.3. Use the "vlim" parameter instead.
#   mne.viz.plot_topomap(zdata[:, 0].mean(axis=0), raw.info, size=4,
# ![](/home/sebastiaan/.opensesame/image_annotations/49c51e551f68400a93386cbafca1793d.png)
# Bin pupil
# /tmp/ipykernel_355863/2580720971.py:19: FutureWarning: The "vmin" and "vmax" parameters are deprecated and will be removed in version 1.3. Use the "vlim" parameter instead.
#   mne.viz.plot_topomap(zdata[:, 1].mean(axis=0), raw.info, size=4,
# ![](/home/sebastiaan/.opensesame/image_annotations/754ee187150345158e9f01e47a099db5.png)
# Intensity
# /tmp/ipykernel_355863/2580720971.py:25: FutureWarning: The "vmin" and "vmax" parameters are deprecated and will be removed in version 1.3. Use the "vlim" parameter instead.
#   mne.viz.plot_topomap(zdata[:, 2].mean(axis=0), raw.info, size=4,
# ![](/home/sebastiaan/.opensesame/image_annotations/638a6494b3df4fe691b370a352bf5683.png)
# Valid
# /tmp/ipykernel_355863/2580720971.py:31: FutureWarning: The "vmin" and "vmax" parameters are deprecated and will be removed in version 1.3. Use the "vlim" parameter instead.
#   mne.viz.plot_topomap(zdata[:, 3].mean(axis=0), raw.info, size=4,
# ![](/home/sebastiaan/.opensesame/image_annotations/3c8b76081c654f74b6010ca8cbb8c23b.png)
# 
"""
Visualize the topomap for the dummy factor. This should give a more-or-less
random distribution.
"""
print('Dummy factor')
mne.viz.plot_topomap(zdata[:, 4].mean(axis=0), raw.info, size=4,
                     names=raw.info['ch_names'], vmin=-MAX, vmax=MAX,
                     cmap=COLORMAP)



# % output
# Dummy factor
# /tmp/ipykernel_355863/1081207082.py:2: FutureWarning: The "vmin" and "vmax" parameters are deprecated and will be removed in version 1.3. Use the "vlim" parameter instead.
#   mne.viz.plot_topomap(zdata[:, 4].mean(axis=0), raw.info, size=4,
# 
# ![](/home/sebastiaan/.opensesame/image_annotations/ef3c3a582c4f4d0ca16c5c21c7738cde.png)
# 
# (<matplotlib.image.AxesImage at 0x7f6b2d9049a0>,
#  <matplotlib.contour.QuadContourSet at 0x7f6b2d904ca0>)
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



# % output
#              Source        SS  ddof1  ddof2        MS         F         p-unc  \
# 0            factor  0.015057      3     81  0.005019  1.765006  1.604166e-01   
# 1           channel  0.022410     25    675  0.000896  3.871133  1.367276e-09   
# 2  factor * channel  0.002302     75   2025  0.000031  1.243309  7.996297e-02   
# 
#    p-GG-corr       ng2       eps  
# 0   0.185616  0.022772  0.585786  
# 1   0.014505  0.033521  0.111072  
# 2   0.299524  0.003549  0.039631  
# /home/sebastiaan/anaconda3/envs/pydata/lib/python3.10/site-packages/pingouin/distribution.py:486: UserWarning: Epsilon values might be innaccurate in two-way repeated measures design where each  factor has more than 2 levels. Please  double-check your results.
#   warnings.warn(
# 
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



# % output
# inducer <> intensity, CP2, t=2.0588, p=0.0493
# inducer <> valid, F4, t=2.1325, p=0.0422
# bin_pupil <> inducer, FC1, t=-2.2759, p=0.0310
# bin_pupil <> inducer, FC2, t=-2.1961, p=0.0369
# bin_pupil <> inducer, CP1, t=-2.5720, p=0.0159
# bin_pupil <> inducer, CP2, t=-2.6407, p=0.0136
# bin_pupil <> inducer, Oz, t=-2.1095, p=0.0443
# 
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



# % output
# Frequency perturbation analysis for 1
# Frequency perturbation analysis for 2
# Frequency perturbation analysis for 3
# Frequency perturbation analysis for 4
# Frequency perturbation analysis for 6
# Frequency perturbation analysis for 8
# Frequency perturbation analysis for 9
# Frequency perturbation analysis for 10
# Frequency perturbation analysis for 11
# Frequency perturbation analysis for 12
# Frequency perturbation analysis for 14
# Frequency perturbation analysis for 16
# Frequency perturbation analysis for 17
# Frequency perturbation analysis for 19
# Frequency perturbation analysis for 20
# Frequency perturbation analysis for 21
# Frequency perturbation analysis for 22
# Frequency perturbation analysis for 23
# Frequency perturbation analysis for 24
# Frequency perturbation analysis for 25
# Frequency perturbation analysis for 26
# Frequency perturbation analysis for 27
# Frequency perturbation analysis for 28
# Frequency perturbation analysis for 29
# Frequency perturbation analysis for 30
# Frequency perturbation analysis for 31
# Frequency perturbation analysis for 32
# Frequency perturbation analysis for 33
# 
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



# % output
# ![](/home/sebastiaan/.opensesame/image_annotations/972e9bbe5ab640bb9a2950caa9e764fa.png)
# 
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



# % output
# ![](/home/sebastiaan/.opensesame/image_annotations/4fb5e3095b854896acd436459878354e.png)
# ![](/home/sebastiaan/.opensesame/image_annotations/a7f6884ac0434667a7e7033ea3197105.png)
# ![](/home/sebastiaan/.opensesame/image_annotations/e72b3cb769f0442bad6960924c18ae4c.png)
# ![](/home/sebastiaan/.opensesame/image_annotations/9837e1bb45d94164a99e44ed87f9b6d8.png)
# 
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

# % output
#            Source        SS  ddof1  ddof2        MS         F         p-unc  \
# 0          factor  0.098879      3     81  0.032960  2.714784  5.015745e-02   
# 1           freqs  0.077229     14    378  0.005516  4.230917  6.094736e-07   
# 2  factor * freqs  0.513163     42   1134  0.012218  9.246072  3.217634e-48   
# 
#       p-GG-corr       ng2       eps  
# 0  6.290808e-02  0.028811  0.813729  
# 1  2.526072e-04  0.022646  0.492974  
# 2  2.049248e-10  0.133418  0.179271  
# Simple effect for bin_pupil
#   Source  ddof1  ddof2        F     p-unc       ng2       eps
# 0  freqs     14    378  3.45683  0.000025  0.065248  0.554206
# Simple effect for inducer
#   Source  ddof1  ddof2          F         p-unc       ng2       eps
# 0  freqs     14    378  14.449552  8.826462e-28  0.222152  0.435238
# Simple effect for intensity
#   Source  ddof1  ddof2         F         p-unc       ng2       eps
# 0  freqs     14    378  8.002313  5.562212e-15  0.173532  0.428445
# Simple effect for valid
#   Source  ddof1  ddof2        F     p-unc       ng2       eps
# 0  freqs     14    378  3.83561  0.000004  0.069558  0.519475
# /home/sebastiaan/anaconda3/envs/pydata/lib/python3.10/site-packages/pingouin/distribution.py:486: UserWarning: Epsilon values might be innaccurate in two-way repeated measures design where each  factor has more than 2 levels. Please  double-check your results.
#   warnings.warn(
# /home/sebastiaan/anaconda3/envs/pydata/lib/python3.10/site-packages/pingouin/parametric.py:551: FutureWarning: Not prepending group keys to the result index of transform-like apply. In the future, the group keys will be included in the index, regardless of whether the applied function returns a like-indexed object.
# To preserve the previous behavior, use
# 
# 	>>> .groupby(..., group_keys=False)
# 
# To adopt the future behavior and silence this warning, use 
# 
# 	>>> .groupby(..., group_keys=True)
#   ss_resall = grp_with.apply(lambda x: (x - x.mean()) ** 2).sum()
# /home/sebastiaan/anaconda3/envs/pydata/lib/python3.10/site-packages/pingouin/parametric.py:551: FutureWarning: Not prepending group keys to the result index of transform-like apply. In the future, the group keys will be included in the index, regardless of whether the applied function returns a like-indexed object.
# To preserve the previous behavior, use
# 
# 	>>> .groupby(..., group_keys=False)
# 
# To adopt the future behavior and silence this warning, use 
# 
# 	>>> .groupby(..., group_keys=True)
#   ss_resall = grp_with.apply(lambda x: (x - x.mean()) ** 2).sum()
# /home/sebastiaan/anaconda3/envs/pydata/lib/python3.10/site-packages/pingouin/parametric.py:551: FutureWarning: Not prepending group keys to the result index of transform-like apply. In the future, the group keys will be included in the index, regardless of whether the applied function returns a like-indexed object.
# To preserve the previous behavior, use
# 
# 	>>> .groupby(..., group_keys=False)
# 
# To adopt the future behavior and silence this warning, use 
# 
# 	>>> .groupby(..., group_keys=True)
#   ss_resall = grp_with.apply(lambda x: (x - x.mean()) ** 2).sum()
# /home/sebastiaan/anaconda3/envs/pydata/lib/python3.10/site-packages/pingouin/parametric.py:551: FutureWarning: Not prepending group keys to the result index of transform-like apply. In the future, the group keys will be included in the index, regardless of whether the applied function returns a like-indexed object.
# To preserve the previous behavior, use
# 
# 	>>> .groupby(..., group_keys=False)
# 
# To adopt the future behavior and silence this warning, use 
# 
# 	>>> .groupby(..., group_keys=True)
#   ss_resall = grp_with.apply(lambda x: (x - x.mean()) ** 2).sum()
# 
