"""
# Analysis of target-evoked pupil constriction.

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual processing*


## Imports and constants
"""
from matplotlib import pyplot as plt
from datamatrix import io
import seaborn as sns
import numpy as np
import time_series_test as tst
from pathlib import Path
from analysis_utils import *


"""
## Load checkpoint
"""
dm = io.readbin(DATA_CHECKPOINT)


"""
## Pupil analysis

Plot the pupil response to the target as a function of the various conditions.
"""
plt.figure(figsize=(12, 4))
plt.subplots_adjust(wspace=0)
plt.subplot(141)
plt.title('a) Induced Pupil size')
pupil_plot(dm, hue_factor='inducer', hues=['blue', 'red'])
plt.subplot(142)
plt.title('b) Spontaneous Pupil size')
plt.axvspan(138 + 15, 138 + 20, color='black', alpha=.1)
pupil_plot(dm, hue_factor='bin_pupil', hues=['purple', 'green'])
plt.gca().get_yaxis().set_visible(False)
plt.subplot(143)
plt.title('c) Stimulus intensity')
plt.axvspan(138 + 20, 138 + 25, color='black', alpha=.1)
pupil_plot(dm, hue_factor='intensity', hues=['gray', 'black'])
plt.gca().get_yaxis().set_visible(False)
plt.subplot(144)
plt.title('d) Covert Visual Attention')
plt.axvspan(138 + 30, 138 + 35, color='black', alpha=.1)
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
Path('output/pupil-constriction-results.txt').write_text(
    tst.summarize(pupil_results))
