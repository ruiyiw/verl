# Changelog

All notable changes to this forked repository will be documented in this file.
This fork is used for academic research experimentations. Changes are not intended for upstream merge.

## [2025-04-29]
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
