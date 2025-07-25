DEVICES=3
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port 29750 \
    --nproc_per_node=2 \
    ./train.py \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=/data/piper_grape0626/lerobot_5hz \
    --dataset.root=/data/piper_grape0626/lerobot_5hz \
    --wandb.enable=true \
    --output_dir=/result/SmolVLA_20250626_piper_pickgrape \
    --job_name=SmolVLA_0626_piper_pickgrape \
    --wandb.disable_artifact=true \
    --batch_size=16 \
    --num_workers=8 \
    --log_freq=10 \
    --save_freq=5000 \
    --steps=20000