# Causal pupil

Analysis scripts for the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual processing*
  
  
## Data

Right now the data are on the shared Google Drive. For the EEG analysis, the data needs to be re-organized using the `data2bids` utility. For the calibration analysis, the `.edf` files need to be put together into a subfolder called `eyetracking_data`.


## Dependencies

The main dependencies are `eeg_eyetracking_parser` and `datamatrix` (>= 1.0) which can be installed as follows:

```
pip install eeg_eyetracking_parser datamatrix
```

See `environment.yaml` for a complete description of the Python environment used for the analysis.


## System requirements

Most of the analyses require 16GB of memory. To run the memoization script for multiple participants in parallel, 64 GB is recommended. To speed up the decoding analyses, a cuda-enabled graphics card is recommended.


## Running the analysis


### Memoization (optional)

`memoize_all.py` is intended to be run from the command line, will run through most of the time-consuming parts of the analysis, including EEG preprocessing and decoding. The resulting cache files are saved in the `.memoize` folder. This folder can also be copied from one computer to another, for example if you want to run the memoization on a very fast computer and the rest of the (less computationally intensive) analysis on your own laptop.


### Checkpoint creation (optional)

`create_checkpoint.py` creates a single merged datafile that can be used for further processing. This datafile should be stored in a subfolder called `checkpoints/`. You can bypass this step by downloading a checkpoint directly from the Google Drive.


### Analysis scripts

The analysis scripts are named by the type of analysis they perform. They assume that a data checkpoint is available in the `checkpoints/` folder.

- `analyze_behavior.py`
- `analyze_calibration.py`
- `analyze_erp.py`
- `analyze_inducer.py`
- `analyze_tfr.py`
- `analysis_utils.py` -- A module with helper functions that are used by the other analysis scripts. This file is not intended to be executed directly.
- `decode.py` -- The decoding analysis doesn't use the data checkpoint. Instead it uses the memoization cache or performs the decoding from scratch if memoization is not available.


## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
