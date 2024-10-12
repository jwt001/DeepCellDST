NUM_PROC=2
GPUS=1,2
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_mask.py \
 --batch_size 64 \
 --num_epochs 60 \
 --gpus $GPUS
