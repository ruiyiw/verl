project_name=$PROJECT_NAME
experiment_name=$EXPERIMENT_NAME
rl_type=$RL_TYPE
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

# Converting actor model
python3 scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path $original_hf_model_path/actor \
    --local_dir $ckpt_dir/global_step_$latest_ckpt_num/actor \
    --target_dir $save_dir/actor

# Upload to S3 bucket
aws s3 sync $save_dir/actor $s3_save_dir/actor


if [ "$rl_type" == "ppo" ]; then
    # Converting critic model for ppo
    python3 scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path $original_hf_model_path/critic \
    --local_dir $ckpt_dir/global_step_$latest_ckpt_num/critic \
    --target_dir $save_dir/critic

    # Upload to S3 bucket
    aws s3 sync $save_dir/critic $s3_save_dir/critic
fi

rm -rf $save_dir