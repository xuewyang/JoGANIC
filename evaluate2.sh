BASE_FOLDER='expt/nytimes/fh_a'
NC='/home/xuewyang/Xuewen/Research/data/Dataminr/nytimes/name_counters.pkl'

CUDA_VISIBLE_DEVICES=1 python commands/__main__.py --param_path $BASE_FOLDER/config.yaml \
--model_path $BASE_FOLDER/serialization/model_state_epoch_101.th --evaluate True  --eval_suffix best_gt2
python compute_metrics.py -c $NC \
$BASE_FOLDER/serialization/generationsbest_gt2.jsonl
