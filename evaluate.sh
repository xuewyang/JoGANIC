BASE_FOLDER='/home/xuewyang/Xuewen/Research/expt/goodnews/tgnc_mstr_wiki'
NC='/home/xuewyang/Xuewen/Research/data/Dataminr/goodnews/name_counters.pkl'

CUDA_VISIBLE_DEVICES=1 python commands/__main__.py --param_path $BASE_FOLDER/config.yaml \
--model_path $BASE_FOLDER/serialization/model_state_epoch_84.th --evaluate True  --eval_suffix epoch_84
python compute_metrics.py -c $NC \
$BASE_FOLDER/serialization/generationsepoch_84.jsonl