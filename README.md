# Multivariate functional data analysis and machine learning methods for anomaly detection in water quality data
 This repository containes the code corresponding to the functional data analysis and results parts "Multivariate functional data analysis and machine learning methods for anomaly detection in water quality data" research paper by Xurxo Rigueira, David Olivieri, Maria Araujo, Angeles Saavedra and Maria Pazo.
 <p align="center">
  <img src="https://github.com/xrigueira/w-fda/blob/main/plots/graphical_abstract.pdf" />
 </p>

## Requirements
To run the code in this repository, you need the following libraries and tools:

- Python 3.9.x
- R 4.3.1
- NumPy
- pandas
- rpy2
- fda
- fda.usc
- roahd
- fda-outlier
- statsmodels
- kneed 0.8.3
- scikit-learn
- matplotlib

## Citation
If you find this repository useful in your research, please consider citing the following paper:

```
@article{Rigueira2025,
    title={Multivariate functional data analysis and machine learning methods for anomaly detection in water quality data},
    author={Xurxo Rigueira and David Olivieri and Maria Araujo and Angeles Saavedra and Maria Pazo},
    journal={},
    volume={}
    year={2025},
    doi={}
    issn={}
}
```

## Structure
The repository contains the following relevant directories and files:

```
w-fda
   |-- data: preprocessed data.
   |-- plots: plots from the paper and their code.
   |-- preprocessors: code to preprocess the raw data.
   |-- results: prediction results and code to analyze and plot them.
   |-- raw-data: original data before preprocessing.
```

## Overview of relevant files and usage

* `preprocessing.py`: call to the preprocessors to join all variables in a single database, perform labelling, imputation, and smoothing.
* `fda.R`: contains the functions to build MMSA and the implemenation of the benchmark functional models --MUOD, MOUT, and MS--, and the Monte Carlo simulation.
* `mout_muod_ms.R`: code to execute the benchmark functional models.
* `simulation.py`: code to run the Monte Carlo simmulation.
* `outDec.py`: code of the nonparametric outlier detector of MMSA.
* `main.py`: main file to obtain MMSA results on each station.