"""
# ERP analyses

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual processing*


## Imports and constants
"""
from matplotlib import pyplot as plt
from datamatrix import io
from analysis_utils import *


"""
## Load checkpoint
"""
dm = io.readbin(DATA_CHECKPOINT)


"""
## Main ERP plots

Plot target-locked ERPs as a function of various factors.
"""
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
plt.figure(figsize=(12, 4))
plt.subplots_adjust(wspace=0, hspace=0)
plt.suptitle(f'{CHANNEL_GROUP} channels')
plt.subplot(141)
erp_plot(dm, hue_factor='inducer', hues=['blue', 'red'])
plt.subplot(142)
erp_plot(dm, hue_factor='bin_pupil', hues=['purple', 'green'])
plt.subplot(143)
erp_plot(dm, hue_factor='intensity', hues=['gray', 'black'])
plt.subplot(144)
erp_plot(dm, hue_factor='valid', hues=['red', 'green'])
plt.savefig(f'svg/laterp-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/laterp-{CHANNEL_GROUP}.png', dpi=300)
plt.show()


"""
Contra vs ipsilateral ERPs
"""
plt.figure(figsize=(12, 16))
plt.suptitle(f'{CHANNEL_GROUP} channels')
plt.subplot(421)
erp_plot(dm, hue_factor='inducer', hues=['blue', 'red'], dv='contra_erp')
plt.subplot(422)
erp_plot(dm, hue_factor='inducer', hues=['blue', 'red'], dv='ipsi_erp')
plt.subplot(423)
erp_plot(dm, hue_factor='bin_pupil', hues=['purple', 'green'], dv='contra_erp')
plt.subplot(424)
erp_plot(dm, hue_factor='bin_pupil', hues=['purple', 'green'], dv='ipsi_erp')
plt.subplot(425)
erp_plot(dm, hue_factor='intensity', hues=['gray', 'black'], dv='contra_erp')
plt.subplot(426)
erp_plot(dm, hue_factor='intensity', hues=['gray', 'black'], dv='ipsi_erp')
plt.subplot(427)
erp_plot(dm, hue_factor='valid', hues=['red', 'green'], dv='contra_erp')
plt.subplot(428)
erp_plot(dm, hue_factor='valid', hues=['red', 'green'], dv='ipsi_erp')
plt.savefig(f'svg/contra-ipsi-erp-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/contra-ipsi-erp-{CHANNEL_GROUP}.png', dpi=300)
plt.show()


"""
## Cluster-based permutation tests

Warning: This analysis takes very long to run!
"""
dm.lat_erp = dm.lat_erp[:, 25:]
for iv in FACTORS:
    result = tst.lmer_permutation_test(
        dm, formula=f'lat_erp ~ {iv}', re_formula=f'~ {iv}',
        groups='subject_nr', winlen=4, suppress_convergence_warnings=True,
        iterations=1000)
    Path(f'output/lmer-laterp-{iv}.txt').write_text(str(result))
