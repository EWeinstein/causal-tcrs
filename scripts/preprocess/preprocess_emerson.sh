python PACKAGE_PATH/causal-tcrs/scripts/preprocess/preprocess_igor.py \
EMERSON_PATH/file_list.txt \
IGOR_PATH/IGoR/igor_src/igor \
--MLSO --subsample=10000 --out=igor-emerson \
--ncut_start=1 --ncut_end=1 \
--datasource=emerson \
--cpu-list=0-40 --timeout=300