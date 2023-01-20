"""
# Luminance calibration analysis

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual processing*


## Imports and constants
"""
from datamatrix import functional as fnc, series as srs, operations as ops
from eyelinkparser import parse, defaulttraceprocessor
import seaborn as sns
import time_series_test as tst
from analysis_utils import *


def select_stimulus_phase(phase):
    return phase == 'stimulus'
    

@fnc.memoize(persistent=True)
def get_data():
    dm = parse(traceprocessor=defaulttraceprocessor(downsample=10,
                                                    blinkreconstruct=True,
                                                    mode='advanced'),
               phasefilter=select_stimulus_phase, gaze_pos=False,
               time_trace=False, multiprocess=None, maxtracelen=300,
               folder='eyetracking_data')
    return dm.trialid < 40


dm = get_data()


"""
Preprocessing.
"""
dm = dm.subject_nr == set(SUBJECTS)
dm = dm.trial == 0
dm.constriction = srs.baseline(dm.ptrace_stimulus, dm.ptrace_stimulus, 0, 5)
dm.pupil = dm.ptrace_stimulus
for row in dm:
    row.color = row.color1 if row.color == '[color1]' else row.color2


"""
Visualize results
"""
plt.figure(figsize=(5, 4))
tst.plot(dm.trialid >= 38, dv='pupil', hue_factor='color',
         hues=[BLUE, RED])
xdata = np.linspace(0, 250, 6)
tdata = xdata / 100
plt.xticks(xdata, tdata)
plt.xlabel('Time (s)')
plt.ylabel('Pupil size (mm)')
plt.savefig('svg/inducer-calibration.png', dpi=300)
plt.savefig('svg/inducer-calibration.svg')
plt.show()


"""
Write results to file
"""
dm.pupil_95_105 = srs.reduce(dm.pupil[:, 95:105])
xdm = (dm.trialid >= 38)['pupil_95_105', 'color', 'subject_nr']
xdm = ops.sort(xdm, by=xdm.subject_nr)
xdm = ops.sort(xdm, by=xdm.color)
io.writetxt(xdm, 'output/calibration.csv')
