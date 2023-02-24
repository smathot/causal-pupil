"""
# Time-frequency analyses

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual processing*


## Imports and constants
"""
from matplotlib import pyplot as plt
from datamatrix import io
import numpy as np
import itertools as it
import time_series_test as tst
from analysis_utils import *

Y_FREQS = np.array([0, 4, 9, 25])
VMIN = -.2
VMAX = .2
CMAP = 'coolwarm'


"""
## Load checkpoint
"""
dm = io.readbin(DATA_CHECKPOINT)


"""
Merge decoding results
"""
EPOCHS_KWARGS['reject_by_annotation'] = True
dm.braindecode_correct = int
dm = dm.subject_nr != {13, 15}
for subject_nr, sdm in ops.split(dm.subject_nr):
    bdm = decode_subject(subject_nr)
    for row in bdm:
        tdm = sdm.count_trial_sequence == row.count_trial_sequence
        dm.braindecode_correct[tdm] = row.braindecode_correct
dm = dm.braindecode_correct == 1


"""
## Time-frequency analyses

Create time-frequency heatmaps for the target-evoked response. This is done
for all factors.
"""
tfr_plot(dm, 'tfr')
plt.savefig(f'svg/target-tfr-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/target-tfr-{CHANNEL_GROUP}.png', dpi=300)
plt.show()
tfr_plot(dm, 'lat_tfr')
plt.savefig(f'svg/target-lat_tfr-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/target-lat_tfr-{CHANNEL_GROUP}.png', dpi=300)
plt.show()
tfr_plot(dm, 'contra_tfr')
plt.savefig(f'svg/target-contra_tfr-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/target-contra_tfr-{CHANNEL_GROUP}.png', dpi=300)
plt.show()
tfr_plot(dm, 'ipsi_tfr')
plt.savefig(f'svg/target-ipsi_tfr-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/target-ipsi_tfr-{CHANNEL_GROUP}.png', dpi=300)
plt.show()


"""
Focus on frequency bands
"""
dm.theta = dm.tfr[:, :4][:, ...]
dm.alpha = dm.tfr[:, 4:8][:, ...]
dm.beta = dm.tfr[:, 8:][:, ...]

plt.figure(figsize=(12, 4))
plt.suptitle('Theta power')
plt.subplot(141)
tst.plot(dm, dv='theta', hue_factor='inducer')
plt.subplot(142)
tst.plot(dm, dv='theta', hue_factor='bin_pupil')
plt.subplot(143)
tst.plot(dm, dv='theta', hue_factor='intensity')
plt.subplot(144)
tst.plot(dm, dv='theta', hue_factor='valid')
plt.show()

plt.figure(figsize=(12, 4))
plt.suptitle('Alpha power')
plt.subplot(141)
tst.plot(dm, dv='alpha', hue_factor='inducer')
plt.subplot(142)
tst.plot(dm, dv='alpha', hue_factor='bin_pupil')
plt.subplot(143)
tst.plot(dm, dv='alpha', hue_factor='intensity')
plt.subplot(144)
tst.plot(dm, dv='alpha', hue_factor='valid')
plt.show()

plt.figure(figsize=(12, 4))
plt.suptitle('Beta power')
plt.subplot(141)
tst.plot(dm, dv='beta', hue_factor='inducer')
plt.subplot(142)
tst.plot(dm, dv='beta', hue_factor='bin_pupil')
plt.subplot(143)
tst.plot(dm, dv='beta', hue_factor='intensity')
plt.subplot(144)
tst.plot(dm, dv='beta', hue_factor='valid')
plt.show()


"""
## Cluster-based permutation tests

Warning: This analysis takes very long to run!
"""
for dv, iv in it.product(['theta', 'alpha', 'beta'], FACTORS):
    result = tst.lmer_permutation_test(
        dm, formula=f'{dv} ~ {iv}', re_formula=f'~ {iv}',
        groups='subject_nr', winlen=2, suppress_convergence_warnings=True,
        iterations=1000)
    Path(f'output/tfr-{dv}-{iv}.txt').write_text(str(result))
