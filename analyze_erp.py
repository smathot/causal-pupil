"""
# ERP analyses

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual processing*


## Imports and constants
"""
from matplotlib import pyplot as plt
from datamatrix import io, series as srs
import seaborn as sns
from statsmodels.formula.api import mixedlm
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
plt.title('Lateralized (contra - ipsi)')
erp_plot(dm, hue_factor='inducer', hues=['blue', 'red'])
plt.subplot(422)
plt.title('Overall')
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
YLIM = -4.2e-6, 1.2e-6

plt.figure(figsize=(12, 4))
plt.subplots_adjust(wspace=0, hspace=0)
plt.suptitle(f'{CHANNEL_GROUP} channels')
plt.subplot(141)
plt.axvspan(25 + 88, 25 + 99, color='black', alpha=.1)
erp_plot(dm, hue_factor='inducer', hues=['blue', 'red'], ylim=YLIM)
plt.subplot(142)
plt.axvspan(25 + 64, 25 + 75, color='black', alpha=.1)
erp_plot(dm, hue_factor='bin_pupil', hues=['purple', 'green'], ylim=YLIM)
plt.subplot(143)
plt.axvspan(25 + 56, 25 + 59, color='black', alpha=.1)
erp_plot(dm, hue_factor='intensity', hues=['gray', 'black'], ylim=YLIM)
plt.subplot(144)
plt.axvspan(25 + 8, 25 + 125, color='black', alpha=.1)
erp_plot(dm, hue_factor='valid', hues=['red', 'green'], ylim=YLIM)
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


"""
## Non-linear effects of spontaneous pupil size

Calculate more fine-grained pupil bins and focus on the ERP during the time
period located by the cluster-based permutation test.
"""
dm.mean_pupil = srs.reduce(dm.pupil_fix)
dm.z_pupil = ''
dm.bin5_pupil = ''
for subject_nr, sdm in ops.split(dm.subject_nr):
    dm.z_pupil[sdm] = ops.z(sdm.mean_pupil)
    sdm_red, sdm_blue = ops.split(sdm.inducer, 'red', 'blue')
    for binnr, bdm in enumerate(ops.bin_split(sdm_red.mean_pupil, 5)):
        dm.bin5_pupil[bdm] = binnr
    for binnr, bdm in enumerate(ops.bin_split(sdm_blue.mean_pupil, 5)):
        dm.bin5_pupil[bdm] = binnr
dm.bin_pupil_roi = dm.lat_erp[:, 25 + 64:25 + 75][:, ...]


"""
Visualize non-linear effects
"""
plt.figure(figsize=(8, 4))
plt.subplots_adjust(wspace=.3)
plt.subplot(121)
plt.title('a) Full time course')
plt.axvspan(25 + 64, 25 + 75, alpha=.1, color='black')
erp_plot(dm, hue_factor='bin5_pupil',
         hues=['#c8e6c9', '#a5d6a7', '#66bb6a', '#43a047', '#2e7d32'])
plt.ylabel('Contralateral - Ipsilateral (µV)')
plt.subplot(122)
plt.title('b) Mean of 250 - 300 ms window')
x = []
xerr = []
y = []
yerr = []
for bin5_pupil, bdm in ops.split(dm.bin5_pupil):
    x.append(bdm.mean_pupil.mean)
    xerr.append(bdm.mean_pupil.std / np.sqrt(len(bdm)))
    y.append(bdm.bin_pupil_roi.mean)
    yerr.append(bdm.bin_pupil_roi.std / np.sqrt(len(bdm)))
plt.errorbar(x, y, yerr, xerr, 'o-')
plt.xlabel('Pupil size (mm)')
plt.ylabel('Contralateral - Ipsilateral (µV)')
plt.savefig(f'svg/nonlinear-erp-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/nonlinear-erp-{CHANNEL_GROUP}.png', dpi=300)
plt.show()


"""
Statically analyze non-linear effects
"""
dm.z_pupil2 = dm.z_pupil ** 2
dm.z_pupil3 = dm.z_pupil ** 3
valid_dm = dm.bin_pupil_roi != np.nan
valid_dm = valid_dm.mean_pupil != np.nan
model = mixedlm(formula='bin_pupil_roi ~ z_pupil + z_pupil2',
                data=valid_dm, groups='subject_nr').fit()
print(model.summary())
