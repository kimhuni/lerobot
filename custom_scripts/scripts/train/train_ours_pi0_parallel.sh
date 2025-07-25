#!/bin/bash

# 사용할 GPU 장치 설정
DEVICES=3,4,5
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

# 훈련할 데이터셋 경로 배열
DATASET_PATHS=(
    "/data/piper_grape0711/lerobot_5hz"
    "/data/piper_grape0711_10"
)

JOB_NAMES=(
    "pi0_20250711_piper_pickgrape_original"
    "pi0_20250711_piper_pickgrape_dataset_size_20"
)

for i in "${!DATASET_PATHS[@]}"; do
    # 현재 순서의 데이터셋 경로와 작업 이름 변수 할당
    CURRENT_DATASET_PATH=${DATASET_PATHS[i]}
    CURRENT_JOB_NAME=${JOB_NAMES[i]}

    # name of the directory
    OUTPUT_DIR="/result/${CURRENT_JOB_NAME}"

    echo "================================================="
    echo "Starting training for: ${CURRENT_JOB_NAME}"
    echo "Dataset path: ${CURRENT_DATASET_PATH}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "================================================="

    CUDA_VISIBLE_DEVICES=${DEVICES} \
      torchrun \
        --master-port 29400 \
        --nproc_per_node=3 \
        ./train_pi0_ddp.py \
        --policy.path=/ckpt/pi0 \
        --use_ddp=true \
        --use_lora=true \
        --dataset.repo_id=${CURRENT_DATASET_PATH} \
        --dataset.root=${CURRENT_DATASET_PATH} \
        --wandb.enable=false \
        --output_dir=${OUTPUT_DIR} \
        --job_name=${CURRENT_JOB_NAME} \
        --wandb.disable_artifact=true \
        --batch_size=6 \
        --num_workers=8 \
        --log_freq=10 \
        --save_freq=10 \
        --steps=20

    # 훈련 성공 여부 확인 (선택 사항)
    if [ $? -eq 0 ]; then
        echo "Successfully finished training for ${CURRENT_JOB_NAME}"
    else
        echo "Error during training for ${CURRENT_JOB_NAME}. Exiting."
        exit 1
    fi
done

echo "All training jobs completed."