DEVICES=0,1,2,3,4,5
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port 29400 \
    --nproc_per_node=6 \
    ./train_pi0_ddp.py \
    --policy.path=/ckpt/pi0 \
    --use_ddp=true \
    --use_lora=true \
    --dataset.repo_id=/data/piper_grape0626/lerobot_5hz \
    --dataset.root=/data/piper_grape0626/lerobot_5hz \
    --dataset.episodes=[]
    --wandb.enable=true \
    --output_dir=/result/pi0_ddp_20250627_piper_pickgrape \
    --job_name=pi0_ddp_piper_pickgrape \
    --wandb.disable_artifact=true \
    --batch_size=6 \
    --num_workers=16 \
    --log_freq=10 \
    --save_freq=10000 \
    --steps=40000