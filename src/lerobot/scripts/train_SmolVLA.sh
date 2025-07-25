DEVICES=1
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port 29750 \
    --nproc_per_node=1 \
    ./train.py \
    --policy.path=lerobot/smolvla_base \
    --policy.repo_id=lerobot/smolvla_base \
    --dataset.repo_id=/data/piper_grape0626/piper_grape0626_lerobot_5hz \
    --dataset.root=/data/piper_grape0626/piper_grape0626_lerobot_5hz \
    --wandb.enable=true \
    --output_dir=/result/SmolVLA_20250626_piper_pickgrape_full \
    --job_name=SmolVLA_0626_piper_pickgrape_full \
    --resume=true \
    --wandb.disable_artifact=true \
    --policy.device=cuda \
    --batch_size=64 \
    --num_workers=8 \
    --log_freq=10 \
    --save_freq=10000 \
    --steps=200000 \
    --optimizer.type=adamw \
    --optimizer.lr=1e-4 \
    --optimizer.weight_decay=1e-4