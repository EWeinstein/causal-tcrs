#! /bin/bash
# DATA_PATH should be set to the path of the downloaded preprocessed Snyder et al. (2020) COVID dataset
DATA_PATH=./igor-snyder/preprocessed_igor.hdf5
python ./src/CausalReceptors/select_outcome_model.py $DATA_PATH \
--outcome=continuous --flip-outcome \
--select-latent-dim=32 --repertoire-latent-dim=32 \
--select-embed-layer-dim=8 --posterior-rank=-1 --select-posterior-rank=-1 \
--selection-channels=8 --encoder-channels=8 \
--conv-kernel=9 --selection-conv-kernel=9 --encoder-conv-kernel=9 \
--no-attention --n-selection-layers=3 --pos-encode \
--encoder-no-attention \
--optimizer=Adam --unit-batch=8 --unit-batch-eval=8 --subunit-batch=16384 --subunit-batch-eval=16384 \
--lr=0.01 --epochs=1000000 --anneal=3 --anneal-time --max-time=10 \
--monitor-iter=20 --validate-iter=500 --ninit=10 \
--data-seed=0 --seed=1 \
--separate-propensity --propensity-update=10 \
--cuda --low-dtype=bfloat16 --num-workers=8 \
--weight-decay=0.01 --monitor-converge \
--intervene-frac=0.1 --uniform-sample-seq --eval-samples=1 --eval-effect-dist \
--splits=8 --split-choice=0 --stratify-cv
# Note: In the preprocessed COVID dataset, an outcome of 0 indicates mild disease and 2 indicates severe.
# We flip the sign of the outcome variable here (--flip-outcome) so that positive effects
# correspond to good patient outcomes. (The dataloader also subtracts the mean, so this is equivalent to encoding
# mild disease as +1 and severe disease as -1.)