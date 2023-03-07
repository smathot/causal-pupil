"""
# Data-quality checks

This script contains data-quality checks for the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*
  
## Imports
"""
from analysis_utils import SUBJECTS
from datamatrix import series as srs
from pathlib import Path


"""
# Count bad channels and excluded ICA components
"""
bads = []
ica = []
for subject in SUBJECTS:
    path = Path(f'data/sub-{subject:02d}/preprocessing')
    bads += eval(
        (path / Path('Bad_channels/subject_None/bads.txt')).read_text())
    ica += eval((path / Path('ICA/subject_None/ica_removed.txt')).read_text())

print(f'{len(bads)} channels marked as bad')
print(f'{len(ica)} components removed')


"""
## Load checkpoint
"""
dm = io.readbin(DATA_CHECKPOINT)


"""
## Proportion of missing trials
"""
nan_prop = len(srs.nancount(dm.erp) > 0) / len(dm)
print(f'Proportion of empty traces: {nan_prop}')


"""
## Check bad annotations and nan-counts

For each participant, check the number of nan signals, and count how many of
the various annotations occur.
"""
for subject_nr, sdm in ops.split(dm.subject_nr):
    print(f'subject:{subject_nr}, N(trial)={len(sdm)}')
    for signal in ('erp', 'pupil_fix', 'pupil_target'):
        n_trials = len(sdm)
        n_nan = len(srs.nancount(sdm[signal]) == sdm[signal].depth)
        print(f'- missing({signal})={n_nan}')
    raw, events, metadata = read_subject(subject_nr)
    annotations = [a['description']
                   for a in raw.annotations
                   if a['description'].startswith('BAD')]
    for annotation in set(annotations):
        n = len([a for a in annotations if a == annotation])
        print(f'- N({annotation})={n}')
