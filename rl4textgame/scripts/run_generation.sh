set -x
MODEL_PATH=Qwen/Qwen2.5-0.5B
DATA_PATH=/workspace/local/full_traj_ppo_data
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HYDRA_FULL_ERROR=1
# Generation
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=8 \
    data.n_samples=1 \
    data.output_path=$DATA_PATH/test-output-8.parquet \
    model.path=$MODEL_PATH \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=2048 \
    rollout.response_length=512 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.max_num_batched_tokens=16384