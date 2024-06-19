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

Preprocessing uses [IGoR](https://github.com/qmarcou/IGoR). 
It was run with IGoR version 1.4.0, patched following this [issue](https://github.com/qmarcou/IGoR/issues/68).



## Workflows

The workflow for preparing the Snyder et al. COVID [dataset](https://clients.adaptivebiotech.com/pub/covid-2020) was:

```
preprocess_snyder.sh
preprocess_snyder_patch.sh
preprocess_snyder_mira.sh
```

The workflow for preparing the Emerson et al. CMV [dataset](https://clients.adaptivebiotech.com/pub/emerson-2017-natgen), 
used to create semisynthetic data, was:
```
preprocess_emerson.sh
```

The total time to run the preprocessing was roughly one week, using 40 cpus.
The time is largely dominated by inference and sampling from IGoR.