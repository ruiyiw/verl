set -x

hf_actor_model_path="Pamela153/w2-o3-q4-multiturn-sft"
actor_model_path=local/model/actor
critic_model_path=local/model/critic
base_model=meta-llama/Llama-3.1-8B
data_path=local/train_parquet
reward_func_path=rl4textgame/reward_score.py
adv_estimator=gae
policy_loss=ppo
rollout_num=1
ppo_mini_batch_size=256
multiturn_max_turns=8
interactive_env=textworld

SEP_TOKEN=\"'<|spec|>'\"
format_reward=0.0
dense_reward_alpha=0.8

project_name=textworld-ppo-multiturn
experiment_name=ppo-multiturn-test
nnodes=1
num_epochs=15
hf_save_freq=1
hf_save_dir="Pamela153/${project_name}_${experiment_name}"


# Check if actor model is specified
if [ -n "$hf_actor_model_path" ]; then
    # If specified, download from HF path if available
    huggingface-cli download $hf_actor_model_path --local-dir $actor_model_path
else
    # Otherwise, use base model (from HF)
    actor_model_path=$base_model
fi

# Check if critic model is specified
if [ -n "$hf_critic_model_path" ]; then
    # If specified, download from HF path if available
    huggingface-cli download $hf_critic_model_path --local-dir $actor_critic_path
else
    # Otherwise, use base model (from HF)
    critic_model_path=$base_model
fi

python3 -m rl4textgame.main_ppo \
    data.train_files="$data_path/train.parquet" \
    data.val_files="$data_path/validation.parquet" \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.train_batch_size=256 \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.is_local=False \
    actor_rollout_ref.model.base_model=$base_model \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss=$policy_loss \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.multiturn_config.is_multiturn=True \
    actor_rollout_ref.rollout.multiturn_config.max_turns=$multiturn_max_turns \
    actor_rollout_ref.rollout.multiturn_config.interactive_env=$interactive_env \
    actor_rollout_ref.rollout.multiturn_config.sep_token=$SEP_TOKEN \
    actor_rollout_ref.rollout.multiturn_config.format_reward=$format_reward \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$rollout_num \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=\"cosine\" \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    algorithm.kl_ctrl.kl_coef=0.005 \
    critic.optim.lr=1e-5 \
    critic.model.path=$critic_model_path \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.use_remove_padding=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.reward_manager=dense \
    reward_model.reward_kwargs.alpha=$dense_reward_alpha \
    custom_reward_function.path=$reward_func_path \
    custom_reward_function.name=reward_func \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=True\
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.validation_data_dir="local/val_results" \
    trainer.nnodes=$nnodes \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.hf_save_freq=$hf_save_freq \
    trainer.hf_save_dir=$hf_save_dir \
    trainer.resume_mode=auto \
    trainer.test_freq=5 \
    trainer.total_epochs=$num_epochs $@