project_name=$PROJECT_NAME
experiment_name=$EXPERIMENT_NAME
worker_type=$WORKER_TYPE
original_hf_model_path=local/model
ckpt_dir=checkpoints/$project_name/$experiment_name
save_dir=local/saved_ckpt/
s3_save_dir=$S3_SAVE_DIR

mkdir $save_dir


latest_ckpt_file="$ckpt_dir/latest_checkpointed_iteration.txt"
if [ ! -f "$latest_ckpt_file" ]; then
    echo "Error: File $latest_ckpt_file not found!"
    exit 1
fi

# Read the first line and store it in the variable
latest_ckpt_num=$(head -n 1 "$latest_ckpt_file")


python3 scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path $original_hf_model_path \
    --local_dir $ckpt_dir/global_step_$latest_ckpt_num/$worker_type \
    --target_dir $save_dir


aws s3 sync $save_dir $s3_save_dir