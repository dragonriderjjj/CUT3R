single GPU finetune:  
cd src/  
accelerate launch --num_processes 1 train.py  --config-name dpt_512_vary_4_64

evaluation:  
bash eval/mv_recon/run.sh
