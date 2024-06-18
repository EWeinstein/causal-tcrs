# Estimating the Causal Effects of T Cell Receptors
This repository contains code for the paper:

> **Estimating the Causal Effects of T Cell Receptors**  
> Eli N. Weinstein, Elizabeth B. Wood, David M. Blei  
> 2024

The code is divided into two main folders. 
`src/CausalReceptors` implements the *causal adaptive immune repertoire estimation (CAIRE)* method.
`scripts` contains additional scripts and notebooks that were used in the experiments and analyses presented in the paper. 


## Installation

The package can be installed by running

```
pip install CausalReceptors
```

**Dependencies**: `pyproject.toml` specifies the dependency versions that were used for the experiments in the paper 
(under `[project] dependencies`), 
but also allows for compatible releases. 
However, the code has not been tested with other versions of these packages.


## Additional scripts

Scripts used for the analyses presented in the paper can be found in
