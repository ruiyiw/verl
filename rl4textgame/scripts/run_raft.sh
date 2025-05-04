set -x
model_path=/local/model
data_path=/local/full_traj_ppo_parquet
adv_estimator=best_of_n_uniform
policy_loss=reinforce
rollout_num=4

project_name=best-of-n-raft-test
experiment_name=textworld
nnodes=1

python3 -m verl.trainer.main_ppo \
    data.train_files="$data_path/train.parquet" \
    data.val_files="$data_path/validation.parquet" \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.train_batch_size=256 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss=$policy_loss \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$rollout_num \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    custom_reward_function.path="rl4textgame/reward_score.py" \
    custom_reward_function.name=reward_func \
    algorithm.adv_estimator=$adv_estimator \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.nnodes=$nnodes \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.save_freq=5 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=checkpoints/$project_name/$experiment_name \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@