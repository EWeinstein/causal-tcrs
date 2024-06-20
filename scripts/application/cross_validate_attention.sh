#! /bin/bash
REPS=3
SPLITS=8
OUTPUT=cir-cv/run-$(date +"%Y-%m-%d-%H-%M-%S")
echo $OUTPUT;
for j in $(seq 0 $((REPS-1))); do
  for i in $(seq 0 $((SPLITS-1))); do
      python PACKAGE_PATH/causal-tcrs/src/CausalReceptors/select_outcome_model.py \
      SNYDER_PATH/igor-snyder/preprocessed_igor.hdf5 \
      --outcome=continuous --flip-outcome --select-latent-dim=32 --repertoire-latent-dim=32 \
      --select-embed-layer-dim=8 --posterior-rank=-1 --select-posterior-rank=-1 \
      --selection-channels=8 --encoder-channels=8 \
      --conv-kernel=9 --selection-conv-kernel=9 --encoder-conv-kernel=9 \
      --n-attention-layers=2 --n-selection-layers=3 --pos-encode \
      --encoder-no-attention \
      --optimizer=Adam --unit-batch=8 --unit-batch-eval=8 --subunit-batch=16384 --subunit-batch-eval=16384 \
      --lr=0.01 --epochs=1000000 --anneal=3 --anneal-time --max-time=10 \
      --monitor-iter=20 --validate-iter=500 --ninit=10 \
      --data-seed=$j --seed=$((j*SPLITS+i+1)) \
      --separate-propensity --propensity-update=10 \
      --cuda --low-dtype=bfloat16 --num-workers=8 \
      --weight-decay=0.01 --monitor-converge \
      --intervene-frac=0.1 --eval-bind=mira --uniform-sample-seq --eval-samples=1 --eval-effect-dist \
      --splits=$SPLITS --split-choice=$i --stratify-cv --log-output=$OUTPUT
  done
done
python PACKAGE_PATH/causal-tcrs/scripts/application/cross_validate.py $OUTPUT