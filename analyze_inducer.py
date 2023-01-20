"""
# Main analysis

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*


## Imports and constants
"""
from matplotlib import pyplot as plt
import time_series_test as tst
import numpy as np
from datamatrix import operations as ops, io, series as srs
import seaborn as sns
from scipy.stats import linregress
from analysis_utils import *
from pathlib import Path


"""
## Load checkpoint
"""
dm = io.readbin(DATA_CHECKPOINT)


"""
## Check inducer effect

For each participant separately, the effect of the inducer on pupil size and
on the ERP is determined. We inspect whether and how these correlate. Finally,
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
