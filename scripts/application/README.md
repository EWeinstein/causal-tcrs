# Application: COVID-19 Severity

This folder contains scripts and notebooks used to perform the study of TCR effects on COVID severity.

## Dependencies
To install the python package dependencies for these scripts,
at the versions used for the analyses in the paper,
you can install the main package with an additional option:

```
pip install -e '.[covid]'
```

## Workflows

The scripts for the models evaluated in the paper are:
- `cross_validate.sh` the main CAIRE model
- `cross_validate_attention.sh` the attention CAIRE model
- `cross_validate_noselect.sh` the uncorrected model
- `cross_validate_simple.sh` the simplified non-neural CAIRE model

### Paths
The above scripts were anonymized as follows: 
- `PACKAGE_PATH` denotes the path to this package (the folder containing causal-tcrs)
- `SNYDER_PATH` denotes the path to the preprocessed Snyder et al. data