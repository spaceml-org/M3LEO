#!/bin/bash
export HYDRA_FULL_ERROR=1
export CLIP_BLOSC_CACHE_DIR="/mnt/disks/sardata/BLOSC"
set -o xtrace

# configs=(
#     "configs/dataset_experiments/baseline-agb/s2rgbm" #10
# )

# configs=(
#     "configs/dataset_experiments/fusion-esawc/gssic" #20
#     "configs/dataset_experiments/fusion-esawc/s1grd-vv-vh" #20
#     "configs/dataset_experiments/baseline-ghsbuilts/s1grd-vh" #10
#     "configs/dataset_experiments/baseline-ghsbuilts/gssic" #10
#     "configs/dataset_experiments/baseline-agb/s1grd-vv" #10
# )
# configs=(
#     "configs/dataset_experiments/baseline-ghsbuilts/s2rgbm" #10
#     "configs/dataset_experiments/baseline-ghsbuilts/s1grd-vv-vh" #10
#     "configs/dataset_experiments/baseline-ghsbuilts/s1grd-vv" #10
#     "configs/dataset_experiments/fusion-agb/gssic" #20
#     "configs/dataset_experiments/fusion-agb/s1grd-vv-vh" #20
# )
configs=(
    "configs/dataset_experiments/fusion-ghsbuilts/gssic" #20
    "configs/dataset_experiments/fusion-ghsbuilts/s1grd-vv-vh" #20
    "configs/dataset_experiments/baseline-esawc/s1grd-vh" #10
    "configs/dataset_experiments/baseline-esawc/gssic" #10
    "configs/dataset_experiments/baseline-agb/s1grd-vv-vh" #10
)
# configs=(
#     "configs/dataset_experiments/baseline-esawc/s2rgbm" #10
#     "configs/dataset_experiments/baseline-esawc/s1grd-vv-vh" #10
#     "configs/dataset_experiments/baseline-esawc/s1grd-vv" #10
#     "configs/dataset_experiments/baseline-agb/s1grd-vh" #10
#     "configs/dataset_experiments/baseline-agb/gssic" #10
# )


# configs_gunw=(
#     "configs/dataset_experiments/fusion-esawc/gunw" #20
#     "configs/dataset_experiments/baseline-ghsbuilts/gunw" #10
#     "configs/dataset_experiments/fusion-ghsbuilts/gunw"
#     "configs/dataset_experiments/baseline-esawc/gunw" #10
#     "configs/dataset_experiments/fusion-agb/gunw" #20
#     "configs/dataset_experiments/baseline-agb/gunw" #10
# )

filtered_configs=()
for config in "${configs[@]}"; do
    if [[ ! "$config" == *"gunw"* ]]; then #Add filtering
        filtered_configs+=("$config")
    fi
done

AOI=all
MODEL=unet
WANDB_PROJECT=data-paper-runs-redux
batch_size="8"
num_workers="8"
dataset_path=/mnt/disks/sardata/
project_path=/home/matt/work/2023-Europe-SAR/
epochs="50"
limit_val_batches="1.0"
limit_test_batches="1.0"

for config_name in "${filtered_configs[@]}"; do
    # Extract TASK and DATA
    DATA=${config_name##*/}
    path_part=${config_name%/*}
    TASK=${path_part##*-}

    # Check for 'fusion' in path and append S2rgbm if present
    if [[ "$config_name" == *"fusion"* ]]; then
        DATA="${DATA}_s2rgbm"
    fi

    tags="unet world ${DATA} ${TASK}" # Added DATA and TASK to tags

    echo "============================TRAINING CONFIG: $config_name================================="
    echo "============================TRAINING CONFIG: $config_name================================="
    echo "============================TRAINING CONFIG: $config_name================================="
    echo "============================TRAINING CONFIG: $config_name================================="
    echo "TRAINING TASK: $TASK, DATA: $DATA"

    python train.py --config-path $config_name wandb.project=$WANDB_PROJECT \
        experiment_name=$DATA.$TASK.$MODEL.$AOI.0.1.pct.$epochs.epochs.scratch.unet.seed \
        ++max_epochs=$epochs ++tags="${tags}" \
        ++dataset_path=$dataset_path ++project_path=$project_path \
        ++config_path=$config_name ++dataloader.batch_size=$batch_size ++dataloader.num_workers=$num_workers \
        ++limit_val_batches=$limit_val_batches ++limit_test_batches=$limit_test_batches
done
