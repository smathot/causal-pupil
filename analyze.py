"""
# Main analysis

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*

This script contains all analyses except for those related to decoding.


## Imports and constants
"""
import mne
from matplotlib import pyplot as plt
import eeg_eyetracking_parser as eet
import time_series_test as tst
import numpy as np
from datamatrix import DataMatrix, convert as cnv, operations as ops, \
    series as srs, functional as fnc, SeriesColumn, io
import seaborn as sns
from scipy.stats import linregress
from mne.time_frequency import tfr_morlet
from analysis_utils import *
from pathlib import Path


"""
## Read the data
"""
# get_merged_data.clear()  # Uncomment to re-merge data
dm = get_merged_data()
dm = dm.practice == 'no'
dm = ops.auto_type(dm)


"""
Print which columns are offloaded to disk. (For debugging purposes.)
"""
for name, col in dm.columns:
    if not col.loaded:
        print(name, col.loaded)


"""
## Select channels

We can select occipital, parietal, central, and frontal channels groups. See
analysis_utils for the exact channels that go into each group.
"""
# Select channels and average them
dm.erp = dm.tgt_erp[:, LEFT_PARIETAL + RIGHT_PARIETAL + MIDLINE_PARIETAL]
dm.erp = dm.erp[:, ...]
dm.left_erp = dm.tgt_erp[:, LEFT_PARIETAL]
dm.left_erp = dm.left_erp[:, ...]
dm.right_erp = dm.tgt_erp[:, RIGHT_PARIETAL]
dm.right_erp = dm.right_erp[:, ...]
# The time-frequency analyses have already been done separately for the
# different channels groups to reduce memory size
dm.tfr = dm[f'tfr_{CHANNEL_GROUP}']
dm.alpha = dm[f'alpha_{CHANNEL_GROUP}']
dm.theta = dm[f'theta_{CHANNEL_GROUP}']


"""
## Recode data

For the target epoch, we determine the difference between contralateral and
ipsilateral electrodes. Also plot the left and right channels as a function
of target_position to visually verify that these lateralized ERPs indeed flip
over as you'd expect them to.

We can select occipital, parietal, and frontal channels. See analysis_utils
for the exact channel names.
"""
dm.lat_erp = SeriesColumn(depth=dm.left_erp.depth)
dm.lat_erp = dm.right_erp - dm.left_erp
for row in dm:
    if row.target_position == 'target_right':
        row.lat_erp *= -1
plt.style.use('default')
plt.subplot(121)
tst.plot(dm, dv='left_erp', hue_factor='target_position')
plt.subplot(122)
tst.plot(dm, dv='right_erp', hue_factor='target_position')


"""
Add a new column to the datamatrix that indicates whether pupil size was large
or small. This is done separately for red and blue inducers to make sure that
bins are not confounded with inducers, but only reflect endogenous flucations
in pupil size.
"""
dm.mean_pupil = srs.reduce(dm.pupil_fix)
dm.bin_pupil = ''
for subject_nr, sdm in ops.split(dm.subject_nr):
    sdm_red, sdm_blue = ops.split(sdm.inducer, 'red', 'blue')
    for binnr, bdm in enumerate(ops.bin_split(sdm_red.mean_pupil, 2)):
        dm.bin_pupil[bdm] = binnr
    for binnr, bdm in enumerate(ops.bin_split(sdm_blue.mean_pupil, 2)):
        dm.bin_pupil[bdm] = binnr
Path('output').mkdir(exist_ok=True)
io.writetxt(dm[dm.subject_nr, dm.bin_pupil], 'output/bin-pupil.csv')


"""
## Data quality

For each participant, check the number of nan signals, and count how many of
the various annotations occur.
"""
# for subject_nr, sdm in ops.split(dm.subject_nr):
#     print(f'subject:{subject_nr}, N(trial)={len(sdm)}')
#     for signal in ('left_erp', 'right_erp', 'erp', 'alpha', 'pupil_fix',
#                    'pupil_target'):
#         n_trials = len(sdm)
#         n_nan = len(srs.nancount(sdm[signal]) == sdm[signal].depth)
#         print(f'- missing({signal})={n_nan}')
#     raw, events, metadata = read_subject(subject_nr)
#     annotations = [a['description']
#                    for a in raw.annotations
#                    if a['description'].startswith('BAD')]
#     for annotation in set(annotations):
#         n = len([a for a in annotations if a == annotation])
#         print(f'- N({annotation})={n}')


"""
## Check inducer effect

For each participant separately, the effect of the inducer on pupil size and
on the ERP is determine. We inspect whether and how these correlate. Finally,
all participants with a zero-or-negative inducer effect
(pupil[blue] >= pupil[red]) are excluded.
"""
erp = []
pupil = []
dm.inducer_effect = 0
for subject_nr, sdm in ops.split(dm.subject_nr):
    sdm_red, sdm_blue = ops.split(sdm.inducer, 'red', 'blue')
    d_erp = srs.reduce(sdm_red.erp).mean \
             - srs.reduce(sdm_blue.erp).mean
    d_pupil = srs.reduce(sdm_red.pupil_fix).mean \
               - srs.reduce(sdm_blue.pupil_fix).mean
    erp.append(d_erp)
    pupil.append(d_pupil)
    dm.inducer_effect[sdm] = d_pupil
plt.axvline(0, color='black')
plt.axhline(0, color='black')
sns.regplot(pupil, erp)
print(linregress(pupil, erp))
dm = dm.inducer_effect > 0


"""
Individual differences in inducer effect.
"""
plt.plot(sorted(pupil), 'o')
plt.axhline(0, color='black', linestyle=':')
plt.xlabel('Participant #')
plt.ylabel('Inducer effect (red - blue)')
plt.savefig('svg/inducer-individual-differences.png', dpi=300)
plt.savefig('svg/inducer-individual-differences.svg')
plt.show()


"""
A three-panel plot to show the inducer effect and how it correlates with the
intensity of the red inducer.
"""
# Main inducer effect
plt.figure(figsize=(12, 4))
plt.subplots_adjust(wspace=.3)
plt.subplot(121)
plt.title('a) Inducer effect during pre-target interval')
plt.axvspan(250, 250 + 124, color='gray', alpha=.2)
tst.plot(dm, dv='pupil_fix', hue_factor='inducer', hues=[BLUE, RED])
x = np.arange(0, 501, 100)
plt.xticks(x, x / 250)
plt.xlabel('Time (s)')
plt.ylabel('Pupil size (mm)')
plt.ylim(4.5, 6.5)
# Inducer effect over time
plt.subplot(122)
plt.title('b) Stability of inducer effect within blocks')
dm.trial_in_block = (dm.trial - 16) - 96 * (dm.block - 1)
sns.lineplot(x='trial_in_block', y='mean_pupil', hue='inducer',
             data=cnv.to_pandas(dm), palette=[RED, BLUE])
plt.legend(title='Inducer')
plt.xlim(1, 97)
plt.ylim(4.5, 6.5)
plt.xticks(range(1, 97, 10))
plt.xlabel('Trial in block (#)')
plt.ylabel('Pupil size (mm)')
plt.savefig('svg/inducer.svg')
plt.savefig('svg/inducer.png', dpi=300)
plt.show()


"""
## Main ERP plots

Plot target-locked ERPs as a function of various factors.
"""
def erp_plot(dm, dv='lat_erp', **kwargs):
    tst.plot(dm, dv=dv, **kwargs)
    # plt.gca().invert_yaxis()
    plt.xticks(np.arange(25, 150, 25), np.arange(0, 500, 100))
    plt.axvline(25, color='black', linestyle=':')
    plt.axhline(0, color='black', linestyle=':')
    plt.xlabel('Time (ms)')
    # plt.ylim(-4.5e-6, 1.5e-6)


plt.figure(figsize=(12, 16))
plt.suptitle(f'{CHANNEL_GROUP} channels')
plt.subplot(421)
erp_plot(dm, hue_factor='inducer', hues=['blue', 'red'])
plt.subplot(422)
erp_plot(dm, hue_factor='inducer', hues=['blue', 'red'], dv='erp')
plt.subplot(423)
erp_plot(dm, hue_factor='bin_pupil', hues=['purple', 'green'])
plt.subplot(424)
erp_plot(dm, hue_factor='bin_pupil', hues=['purple', 'green'], dv='erp')
plt.subplot(425)
erp_plot(dm, hue_factor='intensity', hues=['gray', 'black'])
plt.subplot(426)
erp_plot(dm, hue_factor='intensity', hues=['gray', 'black'], dv='erp')
plt.subplot(427)
erp_plot(dm, hue_factor='valid', hues=['red', 'green'])
plt.subplot(428)
erp_plot(dm, hue_factor='valid', hues=['red', 'green'], dv='erp')
plt.savefig(f'svg/erp-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/erp-{CHANNEL_GROUP}.png', dpi=300)
plt.show()

"""
Only lateralized ERPs in a horizontal arrangement of plots
"""
plt.figure(figsize=(12, 6))
plt.subplots_adjust(wspace=0, hspace=0)
plt.suptitle(f'{CHANNEL_GROUP} channels')
plt.subplot(241)
erp_plot(dm.valid == 'no', hue_factor='inducer', hues=['blue', 'red'])
plt.subplot(242)
erp_plot(dm.valid == 'no', hue_factor='bin_pupil', hues=['purple', 'green'])
plt.subplot(243)
erp_plot(dm.valid == 'no', hue_factor='intensity', hues=['gray', 'black'])
plt.subplot(244)
erp_plot(dm, hue_factor='valid', hues=['red', 'green'])
plt.subplot(245)
erp_plot(dm.valid == 'yes', hue_factor='inducer', hues=['blue', 'red'])
plt.subplot(246)
erp_plot(dm.valid == 'yes', hue_factor='bin_pupil', hues=['purple', 'green'])
plt.subplot(247)
erp_plot(dm.valid == 'yes', hue_factor='intensity', hues=['gray', 'black'])
plt.savefig(f'svg/laterp-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/laterp-{CHANNEL_GROUP}.png', dpi=300)
plt.show()


"""
Statistics
"""
# rm_erp = tst.lmer_series(
#     dm, formula='erp ~ inducer + bin_pupil + intensity + valid',
#     groups='subject_nr', winlen=2)
rm_laterp = tst.lmer_series(
    dm, formula='lat_erp ~ inducer + bin_pupil + intensity + valid',
    groups='subject_nr', winlen=2)
rm_laterp = tst.lmer_series(
    dm, formula='lat_erp ~ inducer',
    groups='subject_nr', winlen=2)
# statsplot(rm_erp)
# plt.show()
statsplot(rm_laterp)
plt.show()

hits = tst.lmer_permutation_test(
    dm, formula='lat_erp ~ inducer + bin_pupil + intensity + valid',
    groups='subject_nr', winlen=2)
print(hits)

"""
## Time-frequency analysis

Create time-frequency heatmaps for the full spectrum. This is done separately
for red and blue inducers.
"""
Y_FREQS = np.array([0, 4, 9, 25])
plt.figure(figsize=(16, 4))
plt.suptitle(f'{CHANNEL_GROUP} channels')
plt.subplot(131)
tfr_red = (dm.inducer == 'red').tfr.mean
tfr_blue = (dm.inducer == 'blue').tfr.mean
plt.title('Red - Blue')
plt.imshow(tfr_red - tfr_blue, aspect='auto')
plt.yticks(Y_FREQS, FULL_FREQS[Y_FREQS])
plt.xticks(np.linspace(0, 125, 5), np.linspace(0, 2, 5))
plt.subplot(132)
tfr_large = (dm.bin_pupil == 1).tfr.mean
tfr_small = (dm.bin_pupil == 0).tfr.mean
plt.title('Large - Small')
plt.imshow(tfr_large - tfr_small, aspect='auto')
plt.yticks(Y_FREQS, FULL_FREQS[Y_FREQS])
plt.xticks(np.linspace(0, 125, 5), np.linspace(0, 2, 5))
plt.subplot(133)
tfr_correct = (dm.accuracy == 1).tfr.mean
tfr_incorrect = (dm.accuracy == 0).tfr.mean
plt.title('Correct - Incorrect')
plt.imshow(tfr_correct- tfr_incorrect, aspect='auto')
plt.yticks(Y_FREQS, FULL_FREQS[Y_FREQS])
plt.xticks(np.linspace(0, 125, 5), np.linspace(0, 2, 5))
plt.savefig(f'svg/tfr-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/tfr-{CHANNEL_GROUP}.png', dpi=300)
plt.show()



"""
Next isolate the alpha band
"""
plt.figure(figsize=(16, 4))
plt.subplot(131)
plt.title('Inducer')
tst.plot(dm, dv='alpha', hue_factor='inducer', hues=['blue', 'red'])
plt.xticks(np.linspace(0, 125, 5), np.linspace(0, 2, 5))
plt.subplot(132)
plt.title('Bin pupil')
tst.plot(dm, dv='alpha', hue_factor='bin_pupil', hues=['blue', 'red'])
plt.xticks(np.linspace(0, 125, 5), np.linspace(0, 2, 5))
plt.subplot(133)
plt.title('Accuracy')
tst.plot(dm, dv='alpha', hue_factor='accuracy', hues=['blue', 'red'])
plt.xticks(np.linspace(0, 125, 5), np.linspace(0, 2, 5))
plt.savefig(f'svg/alpha-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/alpha-{CHANNEL_GROUP}.png', dpi=300)
plt.show()



"""
Statistically test the frequency bands
"""
rm_alpha = tst.lmer_series(dm, formula='alpha ~ inducer + bin_pupil + accuracy',
                           groups='subject_nr')
statsplot(rm_alpha)


"""
## Pupil analysis

Plot the pupil response to the target as a function of the various conditions.
"""
def pupil_plot(dm, dv='pupil_target', **kwargs):
    tst.plot(dm, dv=dv, legend_kwargs={'loc': 'lower left'},
             **kwargs)
    x = np.linspace(12, 262, 6)
    t = [f'{s:.2}' for s in np.linspace(0, 1, 6)]
    plt.xticks(x, t)
    plt.xlabel('Time (s)')
    if dv == 'pupil_target':
        plt.axhline(0, linestyle=':', color='black')
        plt.ylim(-.6, .2)
    else:
        plt.ylim(2, 8)
    plt.xlim(0, 250)
    plt.ylabel('Baseline-corrected pupil size (mm)')

mpl.rcParams['font.family'] = 'Roboto Condensed'
plt.figure(figsize=(8, 4))
plt.subplots_adjust(wspace=0)
plt.subplot(141)
plt.title('a) Pupil size\n(induced)')
pupil_plot(dm, hue_factor='inducer', hues=['blue', 'red'])
plt.subplot(142)
plt.title('b) Pupil size\n(spontanous)')
pupil_plot(dm, hue_factor='bin_pupil', hues=['purple', 'green'])
plt.gca().get_yaxis().set_visible(False)
plt.subplot(143)
plt.title('c) Target intensity\n')
pupil_plot(dm, hue_factor='intensity', hues=['gray', 'black'])
plt.gca().get_yaxis().set_visible(False)
plt.subplot(144)
plt.title('d) Cue validity\n')
pupil_plot(dm, hue_factor='valid', hues=['red', 'green'])
plt.gca().get_yaxis().set_visible(False)
plt.savefig('svg/pupil-target-evoked.svg')
plt.savefig('svg/pupil-target-evoked.png', dpi=300)
plt.show()


"""
Statistically test pupil constriction using crossvalidation. We focus only on
the constriction period between 200 and 800 ms. Factors are ordinally coded
with -1 and 1.
"""
dm.pupil_window = dm.pupil_target[:, 138:173]  # 500 - 700 ms post-target
dm.ord_inducer = ops.replace(dm.inducer, {'blue': -1, 'red': 1})
dm.ord_bin_pupil = ops.replace(dm.bin_pupil, {0: -1, 1: 1})
dm.ord_intensity = ops.replace(dm.intensity, {100: -1, 255: 1})
dm.ord_valid = ops.replace(dm.valid, {'no': -1, 'yes': 1})
pupil_results = tst.find(dm,
    'pupil_window ~ ord_inducer + ord_bin_pupil + ord_intensity + ord_valid',
    re_formula='~ ord_inducer + ord_bin_pupil + ord_intensity + ord_valid',
    groups='subject_nr', winlen=5, suppress_convergence_warnings=True)
print(tst.summarize(pupil_results))
tst.plot(dm, dv='pupil_window', hue_factor='ord_inducer',
         linestyle_factor='ord_intensity')

"""
## Behavioral results
"""
plt.figure(figsize=(8, 4))
plt.subplots_adjust(wspace=.3)
plt.subplot(121)
plt.title('a) Response accuracy')
dm.correct = 100 * dm.accuracy
sns.barplot(x='valid', y='correct', hue='inducer', palette=[RED, BLUE],
            data=dm)
plt.legend(title='Inducer')
plt.xticks([0, 1], ['Valid', 'Invalid'])
plt.xlabel('Cue')
plt.ylabel('Accuracy (%)')
plt.ylim(55, 70)
plt.subplot(122)
plt.title('b) Response time (correct trials)')
sns.barplot(x='valid', y='response_time', hue='inducer', palette=[RED, BLUE],
            data=dm.accuracy == 1)
plt.legend(title='Inducer')
plt.xticks([0, 1], ['Valid', 'Invalid'])
plt.xlabel('Cue')
plt.ylim(550, 750)
plt.ylabel('Response time (ms)')
plt.savefig('svg/behavior.png', dpi=300)
plt.savefig('svg/behavior.svg')
plt.show()
acc_dm = dm[dm.subject_nr, dm.accuracy, dm.response_time, dm.valid, dm.inducer]
acc_dm.valid = acc_dm.valid @ (lambda a: -1 if a == 'no' else 1)
acc_dm.inducer = acc_dm.inducer @ (lambda i: -1 if i == 'blue' else 1)
io.writetxt(acc_dm, 'output/behavior.csv')
