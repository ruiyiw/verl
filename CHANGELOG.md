# Changelog

All notable changes to this forked repository will be documented in this file.
This fork is used for academic research experimentations. Changes are not intended for upstream merge.

## [2025-05-01]
### Added
- `verl/trainer/ppo/core_algos.py`: 
  - Added additional advantage computation functions for RAFT and REINFORCE: `compute_uniform_advantage_return`, `compute_best_of_n_uniform_advantage_return`
  - Added additional policy loss calculation for vanilla reinforce (without CLIP): `compute_reinforce_policy_loss`
  - Modified original `compute_policy_loss` function name to `compute_ppo_policy_loss` to better indicate the loss used

### Modified 
- New entry points for config: 
  - `algorithm.adv_estimator = uniform, best_of_n_uniform` 
  - `policy_loss = ppo, reinforce`
- `verl/trainer/ppo/ray_trainer.py`:
  - Added entry to the newly implemented advantage computations. 
- `verl/workers/actor/dp_actor.py`:
  - Added entry to the newly implemented loss functions.


## [2025-05-05]
### Modified
- Support loading models from local path and using the base model's tokenizer if not provided in local model
  - `verl/trainer/main_ppo.py`: Added condition statement when loading a tokenizer
  - `verl/workers/fsdp_workers.py`: Added additional `config` to various functions that need modification.
  - New entry points for config (`verl/rl4textgame/config/ppo_trainer.yaml`): `is_local: False`, `base_model: meta-llama/Llama-3.1-8B`
- Set chat_template to default (for base models) if `tokenizer.chat_template` is not found.
  - `verl/utils/dataset/rl_dataset.py`: Added condition statement to check if the chat template exists in the model config, provide the default template if not exists. 
- Enable vllm generating with guided json schema
  - `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` and `vllm_rollout.py`: Add additional param `guided_decoding` to the `Sampling_params`.
  - New entry points for config (`verl/rl4textgame/config/ppo_trainer.yaml`): `guided_vllm_generate: False`, `guided_json_schema: /local/template.json`
