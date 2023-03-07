"""
# Checkpoint creation

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*

This script creates a checkpoint file that serves as the data file for the
main analysis scripts.
"""
from matplotlib import pyplot as plt
import eeg_eyetracking_parser as eet
import time_series_test as tst
from datamatrix import operations as ops, series as srs, functional as fnc, \
    SeriesColumn, io, MultiDimensionalColumn
from analysis_utils import *
from pathlib import Path


"""
Read the merged data as one large datamatrix, exclude practice, and auto-type
the columns for performance.
"""
get_merged_data.clear()  # Uncomment to re-merge data
print('Reading merged data from function/ memoization')
dm = get_merged_data()
print('Excluding practice')
dm = dm.practice == 'no'
print('Auto-typing columns')
dm = ops.auto_type(dm)


"""
Print which columns are offloaded to disk. (For debugging purposes.)
"""
for name, col in dm.columns:
    if not col.loaded:
        print(name, col.loaded)


"""
To reduce the data volume, ERPs and time-frequency signals are averaged across
electrode groups.
"""
dm.erp = dm.tgt_erp[:, ALL_CHANNELS][:, ...]
dm.left_erp = dm.tgt_erp[:, LEFT_CHANNELS]
dm.right_erp = dm.tgt_erp[:, RIGHT_CHANNELS]
dm.tfr = dm.tgt_tfr[:, ALL_CHANNELS][:, ...]
dm.left_tfr = dm.tgt_tfr[:, LEFT_CHANNELS]
dm.right_tfr = dm.tgt_tfr[:, RIGHT_CHANNELS]
# If we selected more than 1 channel, we also need to average over those
# channels
if len(LEFT_CHANNELS) > 1:
    dm.left_erp = dm.left_erp[:, ...]
    dm.left_tfr = dm.left_tfr[:, ...]
if len(RIGHT_CHANNELS) > 1:
    dm.right_erp = dm.right_erp[:, ...]
    dm.right_tfr = dm.right_tfr[:, ...]


"""
Remove unused columns to save memory. (Even when offloaded they still slow
things down.)
"""
del dm.tgt_erp
del dm.tgt_tfr


"""
Normalize the lateralized ERPs such that it is contra - ipsi (as opposed to
right - left).
"""
dm.ipsi_erp = SeriesColumn(depth=dm.erp.depth)
dm.contra_erp = SeriesColumn(depth=dm.erp.depth)
dm.ipsi_tfr = MultiDimensionalColumn(shape=dm.tfr.shape[1:])
dm.contra_tfr = MultiDimensionalColumn(shape=dm.tfr.shape[1:])
for row in dm:
    if row.target_position == 'target_right':
        row.contra_erp = row.left_erp
        row.ipsi_erp = row.right_erp
        row.contra_tfr = row.left_tfr
        row.ipsi_tfr = row.right_tfr
    else:
        row.contra_erp = row.right_erp
        row.ipsi_erp = row.left_erp
        row.contra_tfr = row.right_tfr
        row.ipsi_tfr = row.left_tfr
dm.lat_erp = dm.contra_erp - dm.ipsi_erp
dm.lat_tfr = dm.contra_tfr - dm.ipsi_tfr


"""
Plot left, right, contra, and ipsi ERPs as a sanity check, because these
should show strong effects, such that the left and right erps are modulated
by target_position, while contra and ipsi should show very different patterns
from each other but not be modulated by target position.
"""
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Left channels')
tst.plot(dm, dv='left_erp', hue_factor='target_position')
plt.subplot(222)
plt.title('Right channels')
tst.plot(dm, dv='right_erp', hue_factor='target_position')
plt.subplot(223)
plt.title('Contralateral channels')
tst.plot(dm, dv='contra_erp', hue_factor='target_position')
plt.subplot(224)
plt.title('Ipsilateral channels')
tst.plot(dm, dv='ipsi_erp', hue_factor='target_position')
plt.savefig(f'svg/erp-left-right-contra-ipsi-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/erp-left-right-contra-ipsi-{CHANNEL_GROUP}.png', dpi=300)


"""
Plot left, right, and lateralized TFRs.
"""
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title('Ipsilateral channels')
plt.imshow(dm.ipsi_tfr[...])
plt.subplot(132)
plt.title('Contralateral channels')
plt.imshow(dm.contra_tfr[...])
plt.subplot(133)
plt.title('Contra - ipsi')
plt.imshow(dm.lat_tfr[...])
plt.savefig(f'svg/tfr-contra-ipsi-lat-{CHANNEL_GROUP}.svg')
plt.savefig(f'svg/tfr-contra-ipsi-lat-{CHANNEL_GROUP}.png', dpi=300)


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
Save the checkpoint in binary DataMatrix format
"""
io.writebin(dm, DATA_CHECKPOINT)
