DATA_PATH=/workspace/local/full_traj_ppo_data
RAY_DEDUP_LOGS=0

python3 -m rl4textgame.main_eval \
    data.path=$DATA_PATH/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    custom_reward_function.path=rl4textgame/reward_score.py \
    custom_reward_function.name=reward_func