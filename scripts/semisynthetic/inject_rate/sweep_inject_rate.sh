#! /bin/bash
python PACKAGE_PATH/causal-tcrs/semisynthetic/semisynthetic_sweep.py \
PACKAGE_PATH/causal-tcrs/src/CausalReceptors/semisynthetic.py \
PACKAGE_PATH/causal-tcrs/src/CausalReceptors/select_outcome_model.py \
PACKAGE_PATH/causal-tcrs/semisynthetic/hyper_optimization.py \
PACKAGE_PATH/causal-tcrs/semisynthetic/inject_rate/sweep_experiment_inject_rate.ini \
PACKAGE_PATH/causal-tcrs/semisynthetic/inject_rate/sweep_model_inject_rate.ini
