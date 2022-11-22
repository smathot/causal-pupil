"""
# Inducer-intensity-calibration analysis

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*

This script analyzes pupil constriction in response to the red and blue
inducers at the end of the inducer calibration.


## Imports and constants
"""
from datamatrix import functional as fnc, series as srs
from eyelinkparser import parse, defaulttraceprocessor
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
dm.pupil = srs.baseline(dm.ptrace_stimulus, dm.ptrace_stimulus, 0, 5)
for row in dm:
    row.color = row.color1 if row.color == '[color1]' else row.color2


"""
Visualize results
"""
tst.plot(dm.trialid >= 38, dv='pupil', hue_factor='color',
         hues=['blue', 'red'])
xdata = np.linspace(0, 250, 6)
tdata = xdata * 10
plt.xticks(xdata, tdata)
plt.xlabel('Time (ms)')
plt.ylabel('Pupil size (baseline corrected)')
plt.savefig('svg/inducer-calibration.png', dpi=300)
plt.savefig('svg/inducer-calibration.svg')
plt.show()
