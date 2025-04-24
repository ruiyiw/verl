data_start=50001
data_end=55000
game_size=w2-o3-q4
game_dir=/workspace/data/games/tiny_search
local_dir=/workspace/local/

# Add training data and preprocess them into parquet
python3 -m rl4textgame.data_process.data_preprocess \
    --task textworld \
    --dataset_id "${game_size}_${data_start}-${data_end}" \
    --game_dir $game_dir \
    --data_dir /workspace/data/full_traj_ppo_data/tiny_search/5000_data/gold-only_depth-0_max-full-traj-len-12 \
    --local_dir $local_dir/full_traj_ppo_data

mkdir -p $local_dir/game_data/
# Add train game data
for i in $(seq $data_start $data_end); do
    cp -n "${game_dir}/${game_size}_${i}.z8" $local_dir/game_data/
done
# Add validation and test game data
for i in $(seq 1 50); do
    cp -n "${game_dir}/${game_size}_$i.z8" $local_dir/game_data/
done
for i in $(seq 41001 41050); do
    cp -n "${game_dir}/${game_size}_$i.z8" $local_dir/game_data/
done