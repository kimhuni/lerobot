DEVICES=3,4
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port 29400 \
    --nproc_per_node=2 \
    ./train_pi0.py \
    --policy.path=/ckpt/pi0 \
    --use_ddp=true \
    --use_lora=true \
    --dataset.repo_id=/data/piper_lerobot/lerobot_aligncups_5hz/train \
    --dataset.root=/data/piper_lerobot/lerobot_aligncups_5hz/train \
    --wandb.enable=false \
    --output_dir=/result/pi0_20250707_piper_pickgrape \
    --job_name=pi0_ddp_piper_pickgrape \
    --wandb.disable_artifact=true \
    --batch_size=4 \
    --num_workers=8 \
    --log_freq=10 \
    --save_freq=100 \
    --steps=200