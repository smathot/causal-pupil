"""
# Analysis of microsaccade rate.

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual processing*


## Imports and constants
"""
from datamatrix import io, operations as ops
import time_series_test as tst
import seaborn as sns
from pycrosaccade import microsaccades
from eyelinkparser import parse, defaulttraceprocessor
from datamatrix import functional as fnc, series as srs
from pathlib import Path
from analysis_utils import *


"""
Data parsing.
"""
def filter_target(name):
    return name == 4


@fnc.memoize(persistent=True)
def get_data():
    dm = parse(
        traceprocessor=defaulttraceprocessor(
            blinkreconstruct=True,
            downsample=None,
            mode="advanced",
        ),
        phasefilter=filter_target,
        folder='eyetracking_data',
        multiprocess=16,
        pupil_size=False,
        maxtracelen=1000
    )
    dm = dm.inducer != 'None'
    dm = dm.practice == 'no'
    dm = dm.subject_nr == set(SUBJECTS)
    bdm = io.readtxt('output/bin-pupil.csv')
    dm = ops.sort(dm, by=dm.subject_nr)
    bdm = ops.sort(bdm, by=bdm.subject_nr)
    dm.bin_pupil = bdm.bin_pupil
    microsaccades(dm)
    return dm


# get_data.clear()  # Uncomment to clear memoization cache
dm = get_data()


"""
Plot saccade frequency as a function of each factor.
"""
def microsaccade_plot(dm, dv='saccfreq_4', **kwargs):
    """A simple wrapper function that plots pupil size over time.
    
    Parameters
    ----------
    dm: DataMatrix
    dv: str, optional
    **kwargs: dict, optional
    """
    tst.plot(dm, dv=dv, legend_kwargs={'loc': 'lower left'},
             **kwargs)
    x = np.linspace(0, 800, 5)
    # t = [f'{int(s)}' for s in np.linspace(0, 1000, 6)]
    plt.xticks(x)
    plt.xlabel('Time (ms)')
    plt.ylim(0, .5)
    plt.xlim(0, 1000)
    plt.ylabel('Microsaccade rate (/s)')


plt.figure(figsize=(12, 4))
plt.subplots_adjust(wspace=0)
plt.subplot(141)
plt.title('a) Induced Pupil size')
microsaccade_plot(dm, hue_factor='inducer', hues=['blue', 'red'])
plt.subplot(142)
plt.title('b) Spontaneous Pupil size')
plt.axvspan(360, 480, color='black', alpha=.1)
microsaccade_plot(dm, hue_factor='bin_pupil', hues=['purple', 'green'])
plt.gca().get_yaxis().set_visible(False)
plt.subplot(143)
plt.title('c) Stimulus intensity')
microsaccade_plot(dm, hue_factor='intensity', hues=['gray', 'black'])
plt.gca().get_yaxis().set_visible(False)
plt.subplot(144)
plt.title('d) Covert Visual Attention')
microsaccade_plot(dm, hue_factor='valid', hues=['red', 'green'])
plt.gca().get_yaxis().set_visible(False)
plt.savefig('svg/microsaccade-rate.svg')
plt.savefig('svg/microsaccade-rate.png', dpi=300)
plt.show()


"""
Statistically test saccade frequency using crossvalidation. Factors are
ordinally coded with -1 and 1.
"""
dm.ord_inducer = ops.replace(dm.inducer, {'blue': -1, 'red': 1})
dm.ord_bin_pupil = ops.replace(dm.bin_pupil, {0: -1, 1: 1})
dm.ord_intensity = ops.replace(dm.intensity, {100: -1, 255: 1})
dm.ord_valid = ops.replace(dm.valid, {'no': -1, 'yes': 1})
microsaccade_results = tst.find(
    dm, 'saccfreq_4 ~ ord_inducer + ord_bin_pupil + ord_intensity + ord_valid',
    re_formula='~ ord_inducer + ord_bin_pupil + ord_intensity + ord_valid',
    groups='subject_nr', winlen=20, suppress_convergence_warnings=True)
Path('output/microsaccade-results.txt').write_text(
    tst.summarize(microsaccade_results))


"""
Analyze blink rate
"""
dm.nblink = 9 - srs.nancount(dm.blinkstlist_4)
sns.pointplot(y='nblink', hue='inducer', x='bin_pupil', data=dm)
