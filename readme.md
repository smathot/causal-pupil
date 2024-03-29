# Causal pupil

Experimental resources for the following manuscript:

- Mathôt, S., Berberyan, H., Büchel, P., Ruuskanen, V., Vilotjević, A., & Kruijne, W. (2023). Causal effects of pupil size on visual processing. *bioRxiv*. <https://doi.org/10.1101/2023.03.08.531702>

Analysis code is stored on GitHub:

- <https://github.com/smathot/causal-pupil>

Everything else (experiment file, data, checkpoints, and output) is stored on the OSF:

- <https://osf.io/gjpkv/>

## Dependencies

The main dependencies are `eeg_eyetracking_parser` and `datamatrix` (>= 1.0) which can be installed as follows:

```
pip install eeg_eyetracking_parser datamatrix
```

See `environment.yaml` for a complete description of the Python environment used for the analysis.

The experiment file requires [OpenSesame](https://osdoc.cogsci.nl/) 3.3. The experiment requires the [`Pulse_EVT2` plug-in](https://github.com/markspan/EVT2), which needs to be installed separately.


## System requirements

Most of the analyses require 16GB of memory. To run the memoization script for multiple participants in parallel, 64 GB is recommended. To speed up the decoding analyses, a cuda-enabled graphics card is recommended.


## Running the analysis

### Explanation of files and folders on the OSF

The analysis scripts are hosted on GitHub. However, the data files, intermediate files, and output files are hosted on the OSF. You need both in order to reproduce the analyses.

- `data\` contains `.zip` archives with the raw data organized in BIDS format. There is one archive per participant, which needs to be extracted. Eye tracking data is in EyeLink `.edf` format. EEG data is in Brain Vision format (`.vhdr`, `.vmrk`, `.eeg`). For the calibration analysis, the `.edf` files need to be put together into a single subfolder called `eyetracking_data`.
- `checkpoints\` contains processed data as generated by `create_checkpoint.py`. The main checkpoint used for the analyses described in the paper is `19072023-parietal.dm`, which focused on the set of parietal electrodes as defined in `analysis_utils.py`.
- `output\` contains various output file:
  - `behavior.csv` behavioral data as used by `analyze_behavior.R`.
  - `bin-pupil.csv` contains information for each trial about whether spontaneous pupil size was small or large. This is used by the decoding analyses.
  - `calibration-wide.jasp·` contains the statistical analysis of luminance calibration.
  - `itc-[frequency band]-[factor].txt` contains the results of cluster-based permutation tests on intertrial coherence as returned by `time_series_test.lmer_permutation_test()`.
  - `tfr-[frequency band]-[factor].txt` contains the results of cluster-based permutation tests on power as returned by `time_series_test.lmer_permutation_test()`.
  - `lmer-laterp-[factor].txt` contains the results of cluster-based permutation tests on lateralized ERPs as returned by `time_series_test.lmer_permutation_test()`.


### Memoization (optional)

`memoize_all.py` is intended to be run from the command line, will run through most of the time-consuming parts of the analysis, including EEG preprocessing and decoding. The resulting cache files are saved in the `.memoize` folder. This folder can also be copied from one computer to another, for example if you want to run the memoization on a very fast computer and the rest of the (less computationally intensive) analysis on your own laptop.


### Checkpoint creation (optional)

`create_checkpoint.py` creates a single merged datafile that can be used for further processing. This datafile should be stored in a subfolder called `checkpoints/`. You can bypass this step by downloading a checkpoint directly.


### Analysis scripts

The analysis scripts are named by the type of analysis they perform. They assume that a data checkpoint is available in the `checkpoints/` folder.

- `analyze_behavior.py`
- `analyze_behavior.R`
- `analyze_calibration.py`
- `analyze_erp.py`
- `analyze_inducer.py`
- `analyze_microsaccades.py`
- `analyze_tfr.py` -- time-frequency analyses focused on power
- `analyze_itc.py` -- time-frequency analyses focused on intertrial coherence
- `analysis_utils.py` -- A module with helper functions that are used by the other analysis scripts. This file is not intended to be executed directly.
- `decode.py` -- The decoding analysis doesn't use the data checkpoint. Instead it uses the memoization cache or performs the decoding from scratch if memoization is not available.


## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
