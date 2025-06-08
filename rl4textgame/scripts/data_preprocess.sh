data_start=50001
data_end=55000
game_size=w2-o3-q4
hf_repo="Pamela153/textworld"
local_hf_dir="local"
hf_train_data_folder="multiturn_ppo_data/5000_data"
hf_games_folder="games"
local_parquet_dir="local/train_parquet/"
local_schema_dir="local/schemas/"

# Clone private HF repo using token
git clone --filter=blob:none --no-checkout "https://user:${HF_TOKEN}@huggingface.co/datasets/${hf_repo}" $local_hf_dir
cd $local_hf_dir
git sparse-checkout init --cone
git sparse-checkout set $hf_train_data_folder $hf_games_folder
git checkout
cd ..
echo "✅ Private HF repo downloaded!"

# Add training data and preprocess them into parquet
python3 -m rl4textgame.data_process.data_preprocess \
    --task textworld \
    --dataset_id "${game_size}_${data_start}-${data_end}" \
    --game_dir "${local_hf_dir}/${hf_games_folder}" \
    --data_dir "${local_hf_dir}/${hf_train_data_folder}" \
    --local_dir $local_parquet_dir \
    --reward_method game_winning_dense

# Generate json schema for guided vllm generation
# python3 -m rl4textgame.data_process.generate_json_schema \
#     --task textworld \
#     --local_dir $local_schema_dir