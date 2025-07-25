DEVICES=2
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port 29300 \
    --nproc_per_node=1 \
    ./train_pi0_ddp.py \
    --policy.type="diffusion" \
    --policy.optimizer_lr=2.5e-5 \
    --policy.optimizer_weight_decay=1e-10 \
    --dataset.repo_id=/data/piper_grape0626/lerobot_5hz \
    --dataset.root=/data/piper_grape0626/lerobot_5hz \
    --wandb.enable=true \
    --output_dir=/result/diff_20250630_piper_pickgrape \
    --job_name=diff_piper_pickgrape \
    --wandb.disable_artifact=true \
    --batch_size=48 \
    --num_workers=32 \
    --log_freq=10 \
    --save_freq=1000 \
    --steps=40000