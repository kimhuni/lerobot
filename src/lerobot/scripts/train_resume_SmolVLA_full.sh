DEVICES=0
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port 29730 \
    --nproc_per_node=1 \
    ./train.py \
    --config_path=/sdc1/SmolVLA_20250711_piper_pickgrape_full/checkpoints/last/pretrained_model/train_config.json \
    --output_dir=/result/SmolVLA_20250711_piper_pickgrape_full \
    --dataset.repo_id=/data/piper_grape0711_lerobot/lerobot_5hz \
    --dataset.root=/data/piper_grape0711_lerobot/lerobot_5hz \
    --job_name=SmolVLA_0711_piper_pickgrape_full_resume \
    --resume=true \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --batch_size=64 \
    --num_workers=8