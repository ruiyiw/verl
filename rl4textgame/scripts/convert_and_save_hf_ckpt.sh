project_name=best-of-n-raft-test
experiment_name=textworld
worker_type=actor
original_hf_model_path=local/model
ckpt_dir=checkpoints/$project_name/$experiment_name
save_dir=local/saved_ckpt/
s3_save_dir=s3://ruiyi-search-agents/sft_ckpt/w2-o3-q4_raft/5000_data/epoch-1/

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