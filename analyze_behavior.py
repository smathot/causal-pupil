"""
# Analysis of behavioral results

This script belongs to the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual processing*


## Imports and constants
"""
from matplotlib import pyplot as plt
from datamatrix import io
import seaborn as sns
from analysis_utils import *


"""
## Load checkpoint
"""
dm = io.readbin(DATA_CHECKPOINT)


"""
## Behavioral results
"""
plt.figure(figsize=(8, 4))
plt.subplots_adjust(wspace=.3)
plt.subplot(121)
plt.title('a) Response accuracy')
dm.correct = 100 * dm.accuracy
sns.barplot(x='valid', y='correct', hue='inducer', palette=[RED, BLUE],
            data=dm)
plt.legend(title='Inducer')
plt.xticks([0, 1], ['Attended', 'Unattended'])
plt.xlabel('Covert Visual Attention')
plt.ylabel('Accuracy (%)')
plt.ylim(55, 70)
plt.subplot(122)
plt.title('b) Response time (correct trials)')
sns.barplot(x='valid', y='response_time', hue='inducer', palette=[RED, BLUE],
            data=dm.accuracy == 1)
plt.legend(title='Inducer')
plt.xticks([0, 1], ['Attended', 'Unattended'])
plt.xlabel('Covert Visual Attention')
plt.ylim(550, 750)
plt.ylabel('Response time (ms)')
plt.savefig('svg/behavior.png', dpi=300)
plt.savefig('svg/behavior.svg')
plt.show()
acc_dm = dm[dm.subject_nr, dm.accuracy, dm.response_time, dm.valid, dm.inducer]
acc_dm.valid = acc_dm.valid @ (lambda a: -1 if a == 'no' else 1)
acc_dm.inducer = acc_dm.inducer @ (lambda i: -1 if i == 'blue' else 1)
io.writetxt(acc_dm, 'output/behavior.csv')
