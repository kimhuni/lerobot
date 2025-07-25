DEVICES=0
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port 29730 \
    --nproc_per_node=1 \
    ./train.py \
    --policy.path=lerobot/smolvla_base \
    --policy.repo_id=lerobot/smolvla_base \
    --dataset.repo_id=/data/piper_corn_grape_0717/lerobot_5hz_re \
    --dataset.root=/data/piper_corn_grape_0717/lerobot_5hz_re \
    --wandb.enable=true \
    --output_dir=/result/SmolVLA_20250722_piper_corn_grape \
    --job_name=SmolVLA_0722_piper_pick_corn_grape \
    --resume=false \
    --wandb.disable_artifact=true \
    --policy.device=cuda \
    --batch_size=64 \
    --num_workers=8 \
    --log_freq=10 \
    --save_freq=10000 \
    --steps=100000 \
    --optimizer.type=adamw \
    --optimizer.lr=1e-4 \
    --optimizer.weight_decay=1e-4