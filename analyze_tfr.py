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
## Time-frequency analyses

Create time-frequency heatmaps for the target-evoked response. This is done
for all factors.
"""
dm.tfr = dm.tgt_tfr_parietal
plt.figure(figsize=(12, 4))
plt.subplots_adjust(wspace=0)
plt.subplot(141)
tfr_red = (dm.inducer == 'red').tfr[...]
tfr_blue = (dm.inducer == 'blue').tfr[...]
plt.title('a) Induced Pupil Size (Large - Small)')
plt.imshow(tfr_red - tfr_blue, aspect='auto', vmin=VMIN, vmax=VMAX, cmap=CMAP,
           interpolation='bicubic')
plt.yticks(Y_FREQS, FULL_FREQS[Y_FREQS])
plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
 
plt.subplot(142)
tfr_large = (dm.bin_pupil == 1).tfr[...]
tfr_small = (dm.bin_pupil == 0).tfr[...]
plt.title('b) Spontaneous Pupil Size (Large - Small)')
plt.imshow(tfr_large - tfr_small, aspect='auto', vmin=VMIN, vmax=VMAX,
           cmap=CMAP, interpolation='bicubic')
plt.gca().get_yaxis().set_visible(False)
plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
plt.xlabel('Time (ms)')

plt.subplot(143)
tfr_bright = (dm.intensity == 255).tfr.mean
tfr_dim = (dm.intensity == 100).tfr.mean
plt.title('c) Stimulus Intensity (Bright - Dim)')
plt.imshow(tfr_bright - tfr_dim, aspect='auto', vmin=VMIN, vmax=VMAX,
           cmap=CMAP, interpolation='bicubic')
plt.gca().get_yaxis().set_visible(False)
plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
plt.xlabel('Time (ms)')

plt.subplot(144)
tfr_attended = (dm.valid == 'yes').tfr.mean
tfr_unattended = (dm.valid == 'no').tfr.mean
plt.title('d) Covert Visual Attention (Attended - Unattended)')
plt.imshow(tfr_attended - tfr_unattended, aspect='auto', vmin=VMIN, vmax=VMAX,
           cmap=CMAP, interpolation='bicubic')
plt.gca().get_yaxis().set_visible(False)
plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
plt.xlabel('Time (ms)')

plt.savefig(f'svg/target-tfr.svg')
plt.savefig(f'svg/target-tfr.png', dpi=300)
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
