"""
# Inter-trial-coherence analyses

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual processing*


## Imports and constants
"""
from analysis_utils import *
from datamatrix import convert as cnv, DataMatrix, MultiDimensionalColumn, io
from mne.time_frequency import tfr_morlet
from matplotlib import pyplot as plt
import itertools as it
from pathlib import Path


"""
## Intertial coherence

For each participant separately, create a contrast of intertrial coherence for
each factor. We store this as a DataMatrix where each row corresponds to a
participant, and each contrast is a multidimensional column.
"""
def itc(epochs):
    power, itc = tfr_morlet(
        epochs, freqs=FULL_FREQS, n_cycles=2, n_jobs=-1, return_itc=True,
        use_fft=True, average=True, decim=4,
        picks=np.arange(len(ALL_CHANNELS)))
    itc.crop(0, .5)
    return itc.data.mean(axis=0)


dm = DataMatrix(length=len(SUBJECTS))
dm.inducer = MultiDimensionalColumn(shape=(26, 32))
dm.bin_pupil = MultiDimensionalColumn(shape=(26, 32))
dm.intensity = MultiDimensionalColumn(shape=(26, 32))
dm.valid = MultiDimensionalColumn(shape=(26, 32))
for row, subject_nr in zip(dm, SUBJECTS):
    print(f'Processing subject {subject_nr}')
    raw, events, metadata = add_bin_pupil(*read_subject(subject_nr))
    epochs = get_tgt_epoch(raw, events, metadata, baseline=None, tmin=-.1,
                           tmax=1, channels=ALL_CHANNELS)
    row.inducer = \
        itc(epochs['inducer == "red"']) - itc(epochs['inducer == "blue"'])
    row.bin_pupil = \
        itc(epochs['bin_pupil == 1']) - itc(epochs['bin_pupil == 0'])
    row.intensity = \
        itc(epochs['intensity == 255']) - itc(epochs['intensity == 100'])
    row.valid = \
        itc(epochs['valid == "yes"']) - itc(epochs['valid == "no"'])
io.writebin(dm, 'output/intertrial-coherence.dm')


"""
Plot intertrial coherence.
"""
ITC_MIN = -.1
ITC_MAX = .1
plt.figure(figsize=(12, 4))
plt.subplots_adjust(wspace=0, hspace=0)
plt.subplot(141)
plt.title('a) Induced Pupil Size (Large - Small)')
plt.imshow(dm.inducer.mean, aspect='auto', vmin=ITC_MIN, vmax=ITC_MAX,
           cmap=CMAP, interpolation='bicubic')
plt.yticks(Y_FREQS, FULL_FREQS[Y_FREQS])
plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.subplot(142)
plt.title('b) Spontaneous Pupil Size (Large - Small)')
plt.imshow(dm.bin_pupil.mean, aspect='auto', vmin=ITC_MIN, vmax=ITC_MAX,
           cmap=CMAP, interpolation='bicubic')
plt.gca().get_yaxis().set_visible(False)
plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
plt.xlabel('Time (ms)')
plt.subplot(143)
plt.title('c) Stimulus Intensity (Bright - Dim)')
plt.imshow(dm.intensity.mean, aspect='auto', vmin=ITC_MIN, vmax=ITC_MAX,
           cmap=CMAP, interpolation='bicubic')
plt.gca().get_yaxis().set_visible(False)
plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
plt.xlabel('Time (ms)')
plt.subplot(144)
plt.title('d) Covert Visual Attention (Attended - Unattended)')
plt.imshow(dm.valid.mean, aspect='auto', vmin=ITC_MIN, vmax=ITC_MAX,
           cmap=CMAP, interpolation='bicubic')
plt.gca().get_yaxis().set_visible(False)
plt.xticks(np.arange(0, 31, 6.25), np.arange(0, 499, 100))
plt.xlabel('Time (ms)')
plt.savefig(f'svg/itc-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/itc-{CHANNEL_GROUP}.png', dpi=300)
plt.show()


"""
## Cluster-based permutation tests

A separate analysis of each frequency band. We're going to test the intercept
without any predictors; to accomplish this, we add a dummy predictor with a
constant value. We're also going to test subject averages, as opposed to
individual trials; to accomplish this, we're specifying no groups so that the
analysis resorts back to using a regular regression (as opposed to a linear
mixed effects analysis).
"""
dm.dummy = 0
for dv, iv in it.product(['theta', 'alpha', 'beta'], FACTORS):
    if dv == 'theta':
        dm.dv = dm[iv][:, :4][:, ...]
    elif dv == 'alpha':
        dm.dv = dm[iv][:, 4:8][:, ...]
    elif dv == 'beta':
        dm.dv = dm[iv][:, 8:][:, ...]
    else:
        raise ValueError()
    result = tst.lmer_permutation_test(
        dm, formula='dv',
        groups=None, winlen=2, suppress_convergence_warnings=True,
        iterations=1000, test_intercept=True)
    Path(f'output/itc-{dv}-{iv}.txt').write_text(str(result))
