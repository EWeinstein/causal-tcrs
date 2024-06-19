python PACKAGE_PATH/causal-tcrs/scripts/preprocess/preprocess_igor.py \
SNYDER_PATH/ImmuneCODE-Repertoire-Tags-002.2.tsv \
IGOR_PATH/IGoR/igor_src/igor \
--MLSO --subsample=10000 --out=igor-snyder-init \
--ncut_start=1 --ncut_end=1 \
--datasource=snyder \
--repertoires-folder=SNYDER_PATH/ImmuneCODE-Review-002 \
--cpu-list=0-40 --timeout=300