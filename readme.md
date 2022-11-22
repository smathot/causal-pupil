# Causal pupil

Analysis scripts for the following manuscript:

- Mathôt, Berberyan, Büchel, Ruuskanen, Vilotjević, & Kruijne (in prep.)
  *Causal effects of pupil size on visual ERPs*
  
  
## Data

Right now the data are on the shared Google Drive. For the EEG analysis, the data needs to be re-organized using the `data2bids` utility. For the calibration analysis, the `.edf` files need to be put together into a subfolder called `eyetracking_data`.


## Dependencies

The main dependency is `eeg_eyetracking_parser` which can be installed as follows:

```
pip install eeg_eyetracking_parser
```


## System requirements

Most of the analyses require 16GB of memory. To run the memoization script for multiple participants in parallel, 64 GB is recommended. To speed up the decoding analyses, a cuda-enabled graphics card is recommended.


## Running the analysis

- `memoize_all.py` -- This script, which is intended to be run from the command line, will run through most of the time-consuming parts of the analysis, including EEG preprocessing and decoding. The resulting cache files are saved in the `.memoize` folder. This folder can also be copied from one computer to another, for example if you want to run the memoization on a very fast computer and the rest of the (less computationally intensive) analysis on your own laptop.
- `analyze_calibration.py` -- Analyzes the pupil size during the calibration phase of the experiment. This script is intended to be run notebook style from a Python code editor such as Spyder or Rapunzel.
- `analyze.py` -- Performs the main analyses. This script is intended to be run notebook style from a Python code editor such as Spyder or Rapunzel.
- `decode.py` -- Performs the decoding analyses. This script is intended to be run notebook style from a Python code editor such as Spyder or Rapunzel.
- `analysis_utils.py` -- A module with helper functions that are used by the other analysis scripts. This file is not intended to be executed directly.


## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
