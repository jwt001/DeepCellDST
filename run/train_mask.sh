NUM_PROC=4
GPUS=0,1,2,3
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py \
 --gpus $GPUS
