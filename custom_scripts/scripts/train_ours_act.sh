DEVICES=3
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --nproc_per_node=1 \
    ./train_pi0_ddp.py \
    --policy.type="act" \
    --dataset.repo_id=/data/piper_grape0626/lerobot_5hz \
    --dataset.root=/data/piper_grape0626/lerobot_5hz \
    --wandb.enable=true \
    --output_dir=/result/act_20250627_piper_pickgrape \
    --job_name=act_piper_pickgrape \
    --wandb.disable_artifact=true \
    --batch_size=48 \
    --num_workers=32 \
    --log_freq=10 \
    --save_freq=100 \
    --steps=40000