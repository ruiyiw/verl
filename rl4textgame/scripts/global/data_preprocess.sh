set -x

data_start=50001
data_end=55000
game_size=w2-o3-q4
hf_repo=$HF_DATA_REPO
hf_games_dir=$HF_GAMES_DIR
hf_train_data_dir=$HF_TRAIN_DATA_DIR
local_games_dir="local/games"
local_train_data_dir="local/multiturn_ppo_data/5000_data"
local_parquet_dir="local/train_parquet"
local_schema_dir="local/schemas"
reward_method=$REWARD_METHOD

# HF download games and train data
python3 -m rl4textgame.utils.hf_download \
    --repo_id $hf_repo \
    --local_dir "local" \
    --hf_target_folder $hf_games_dir \
    --repo_type "dataset"

python3 -m rl4textgame.utils.hf_download \
    --repo_id $hf_repo \
    --local_dir "local" \
    --hf_target_folder $hf_train_data_dir \
    --repo_type "dataset"

# Untar game files
full_games_dir="$(pwd)/$local_games_dir"
echo "Looking for tar files in: $full_games_dir"
ls -la "$full_games_dir"/*.tar.gz

for tar_file in "$full_games_dir"/*.tar.gz; do
    if [ -f "$tar_file" ]; then
        echo "Extracting $(basename "$tar_file")..."
        tar -xzf "$tar_file" -C "$full_games_dir"
        echo "Deleting $(basename "$tar_file")..."
        rm "$tar_file"
    fi
done

# Add training data and preprocess them into parquet
python3 -m rl4textgame.utils.data_preprocess \
    --task textworld \
    --dataset_id "${game_size}_${data_start}-${data_end}" \
    --game_dir $local_games_dir \
    --data_dir $local_train_data_dir \
    --local_dir $local_parquet_dir \
    --reward_method $reward_method

# remove cache and redundant files
rm -rf "local/.cache/"