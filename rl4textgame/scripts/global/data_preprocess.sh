data_start=50001
data_end=55000
game_size=w2-o3-q4
s3_train_data_dir=$S3_TRAIN_DATA_DIR
s3_games_dir=$S3_GAMES_DIR
local_train_data_dir="local/train_data/"
local_games_dir="local/games/"
local_parquet_dir="local/train_parquet/"
local_schema_dir="local/schemas/"
reward_method=$REWARD_METHOD

# Download data from S3 bucket (should contain train.jsonl, validation.jsonl, and test.jsonl)
aws s3 sync $s3_train_data_dir $local_train_data_dir --only-show-errors
# Download games from S3 bucket
aws s3 sync $s3_games_dir $local_games_dir --only-show-errors

# Add training data and preprocess them into parquet
python3 -m rl4textgame.data_process.data_preprocess \
    --task textworld \
    --dataset_id "${game_size}_${data_start}-${data_end}" \
    --game_dir $local_games_dir \
    --data_dir $local_train_data_dir \
    --local_dir $local_parquet_dir \
    --reward_method $reward_method

# Remove .jsonl file
rm -rf $local_train_data_dir

# Generate json schema for guided vllm generation
python3 -m rl4textgame.data_process.generate_json_schema \
    --task textworld \
    --local_dir $local_schema_dir