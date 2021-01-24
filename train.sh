#CUDA_VISIBLE_DEVICES='0' python -W ignore commands/__main__.py --param_path expt/nytimes/tgnc_mstr/config.yaml \
#--train True --force True
CUDA_VISIBLE_DEVICES='1' python -W ignore commands/__main__.py --param_path /home/xuewyang/Xuewen/Research/expt/nytimes/tgnc/config.yaml \
--train True --force True
