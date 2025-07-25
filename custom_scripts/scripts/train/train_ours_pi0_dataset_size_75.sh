DEVICES=4,5
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port 29100 \
    --nproc_per_node=2 \
    ./train_pi0_ddp.py \
    --policy.path=/ckpt/pi0 \
    --use_ddp=true \
    --dataset.repo_id=/data/piper_grape0626_75 \
    --dataset.root=/data/piper_grape0626_75 \
    --wandb.enable=true \
    --output_dir=/result/pi0_20250702_piper_pickgrape_dataset_size_75 \
    --job_name=pi0_piper_pickgrape_dataset_size_75 \
    --wandb.disable_artifact=true \
    --batch_size=6 \
    --num_workers=16 \
    --log_freq=10 \
    --save_freq=10000 \
    --steps=40000