# Preprocessing

This folder contains scripts that were used to prepare datasets for CAIRE.
They are not written to be immediately reusable on new datasets, 
due to the lack of fully standardized data formats in the field 
(and the substantial computational costs involved in running the preprocessing).
If you would like to adapt these scripts to your own data, the best place to start
is likely by modifying the "Process metadata" section of `preprocess_igor.py` (line 254).

## Dependencies
To install the python package dependencies for these scripts, you can install the main package with an additional option:

```
pip install -e '.[pre]'
```

Preprocessing uses IGoR, which can be downloaded and installed [here](https://github.com/qmarcou/IGoR). 
It was run with IGoR version 1.4.0, patched following this [issue](https://github.com/qmarcou/IGoR/issues/68).

## Workflows

The workflow for preparing the [Snyder et al. COVID dataset](https://clients.adaptivebiotech.com/pub/covid-2020) was:

```
preprocess_snyder.sh
preprocess_snyder_patch.sh
preprocess_snyder_mira.sh
```

The workflow for preparing the [Emerson et al. CMV dataset](https://clients.adaptivebiotech.com/pub/emerson-2017-natgen), 
used to create semisynthetic data, was:
```
preprocess_emerson.sh
```

### Paths
The above scripts were anonymized as follows: 
- `PACKAGE_PATH` denotes the path to this package (the folder containing causal-tcrs)
- `EMERSON_PATH` denotes the path to a folder with the Emerson et al. data, containing 
  - the Emerson dataset repertoires and 
  - a text file `file_list.txt` containing the all the repertoire file names, separated by newlines (Keck0001_MC1.tsv\nKeck0002_MC1.tsv...)
- `SNYDER_PATH` denotes the path to the Snyder et al. data, which should contain
  - `ImmuneCODE-Repertoire-Tags-002.2.tsv` metadata on the patients and their repertoires
  - `ImmuneCODE-Review-002` a folder containing patient repertoire files from the main cohort
  - `ImmuneCODE-Repertoires-002.2` a folder containing patient repertoire files from the MIRA study cohort
  - `ImmuneCODE-MIRA-Release002.1` a folder containing the MIRA assay results
- `IGOR_PATH` denotes the path to the folder containing IGoR, which should have
  - `IGoR/igor_src/igor` the installed IGoR executable
  - `IGoR/models/human/tcr_beta` the TCR Beta model provided with IGoR
- `OUT_PATH` denotes the path to the folder where the preprocessed data is output (the folder from which `preprocess_igor.py` was run)

### Compute
The total time to run the preprocessing was roughly one week, using 40 cpus.
The time is largely dominated by inference and sampling from IGoR.