# Semisynthetic experiments

This folder contains scripts that were used to perform the semisynthetic data studies used to evaluate CAIRE.

## Dependencies
To install the python package dependencies for these scripts,
at the versions used for the analyses in the paper,
you can install the main package with an additional option:

```
pip install -e '.[synth]'
```

## Workflows

Subfolders contain scripts for each of the semisynthetic experiments presented in the paper:
- `class/sweep_class.sh` compares several different model classes
- `strength/sweep_strength.sh` compares CAIRE and the uncorrected model across different levels of confounding
- `inject_rate/sweep_inject_rate.sh` compares CAIRE and the uncorrected model across different prevalences of the causal motif

The notebook used to summarize the results of these experiments can be found in `Semisynthetic_analysis.ipynb`

### Paths

The above scripts were anonymized, with `PACKAGE_PATH` denoting the path to this package (the folder containing causal-tcrs).

The associated configuration files 
(`class/sweep_experiment_class.ini`, `strength/sweep_experiment_strength.ini`, `inject_rate/sweep_experiment_inject_rate.ini`)
were also anonymized, 
with `EMERSON_PROCESSED_PATH` denoting the path to a folder with the preprocessed Emerson et al. data.


### Compute

The total time to run the semisynthetic experiments was several days, using an NVIDIA A100 GPU and 8 cpus.
