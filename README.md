# Estimating the Causal Effects of T Cell Receptors
This repository contains code for the paper:

> **Estimating the Causal Effects of T Cell Receptors**  
> Eli N. Weinstein, Elizabeth B. Wood, David M. Blei  
> 2024

The code is divided into two main folders. 
`src/CausalReceptors` implements the *causal adaptive immune repertoire estimation (CAIRE)* method.
`scripts` contains additional scripts and notebooks that were used in the experiments and analyses presented in the paper. 


## Installation

The package can be installed by running from the project directory

```
pip install CausalReceptors
```

**Dependencies**: `pyproject.toml` specifies the dependency versions that were used for the experiments in the paper 
(under `[project] dependencies`), 
but also allows for compatible releases. 
The code has not been tested with other versions of these packages.

## Data

The preprocessed datasets used for the paper can be downloaded from TODO. 

The file TODO, once unzipped, contains
- `igor-snyder/preprocessed_igor.hdf5` - preprocessed Snyder et al. (2020) COVID dataset
- `igor-snyder/peptide-ci.csv`, `igor-snyder/peptide-cii.csv` - preprocessed Snyder et al. (2020) MIRA assay hit metadata
The file TODO, once unzipped, contains 
- `igor-emerson/preprocessed_igor.hdf5` - preprocessed Emerson et al. (2017) dataset, used for semisynthetic data generation

## Data format

The model takes as input an hdf5 file, such as `igor-snyder/preprocessed_igor.hdf5`.
This file must contain the fields
- `['productive_aa']` (int array) - the mature repertoire TCR sequences
  - This is an array of integers of size S x L, where S is the total number of sequences and L is the maximum sequence length plus one
  - The sequences are grouped by patients, if M0 and M1 are the total sequences in the repertoires of patients 0 and patient 1, sequences 0 through M0-1 are from patient 0, sequences M0 through M0+M1-1 are from patient 1, etc.
  - Each nonnegative integer in the array corresponds to an amino acid, in the order specified by 'aa_alphabet' in the metadata (see below)
  - The integer -1 corresponds to 'no amino acid', i.e. padding. This padding is added on the right, up to length L
- `['naive_aa']` (int array) - the preselection immature repertoire TCR sequences
  - These sequences are generated from an IGoR model fit to nonproductive patient TCRs, during preprocessing
  - This is an array of integers of size S' x L. The length L must be the same as in the mature repertoires dataset above (but the total number of sequences S' need not match S).
  - The format is otherwise the same as for the mature repertoire, `['productive_aa']`
- `['outcomes']` (float vector) - the scalar outcomes for each patient
  - This is a vector of floats of size N, the number of patients

Optionally, it may contain:
- `['productive_freq']` (float vector) - the frequencies of each mature repertoire TCR sequences
  - This is a vector of floats of size S, where S is the total number of sequences
  - The sequences are grouped by patients, as for `['productive_aa']`. Typically, they will sum to 1 for each patient (though before training, the frequencies will be rescaled to sum to one exactly)
  - This field is used for repertoire datasets in which each unique TCR sequence is recorded exactly once, along with an estimate of how common the sequence is among all productive TCRs 

It also must contain the metadata fields
- `['metadata'].attrs['nsamples']` (int) - total number of patients 
- `['metadata'].attrs['aa_alphabet']` (string) - the amino acid alphabet, in the order used for sequence encoding; the final letter should correspond to padding character, e.g. '*'
- `['metadata'].attrs['max_len']` (int) - the maximum sequence length
- `['metadata'].attrs['seq_nums']` (int vector) - the number of sequences for each mature repertoire
  - This is a vector of integers of size N, where N is the number of patients
  - The _i_ th entry is the number of sequences from the mature repertoire of the _i_ th patient
  - The sum of all the entries should match S, the size of `['productive_aa']`
- `['metadata'].attrs['naive_nums']` (int vector) - the number of sequences for each preselection repertoire
  - This is a vector of integers of size N, where N is the number of patients
  - The _i_ th entry is the number of (IGoR-simulated) sequences from the preselection repertoire of the _i_ th patient
  - The sum of all the entries should match S', the size of `['naive_aa']`

For further details, see the dataset class `RepertoiresDataset` in `src/CausalReceptors/dataloader.py`.


## Example

An example running CAIRE on the COVID dataset can be found in `example.sh`.
To run this example, you will need to set `DATA_PATH` to the location of the preprocessed Snyder et al. (2020) COVID dataset.

## Additional scripts

The scripts that were used for the analyses presented in the paper can be found in `scripts`. 
Note these are provided as a record of what was done in the paper,
and have not been prepared for reuse.
