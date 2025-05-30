# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Ruiyi Wang, PEARLS lab, University of California, San Diego, advised by Prithviraj Ammanabrolu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

# Added by Ruiyi Wang(05/28/2025)
# Support dense reward assignment for multiturn RL
class DenseRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.alpha = kwargs.pop('alpha', 0.0)  # Extract alpha from reward_model kwargs with default value
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            score = data_item.non_tensor_batch["multiturn_final_rewards"]
            sep_pos = data_item.non_tensor_batch["multiturn_sep_pos"]

            # Assign dense reward to reward tensor [α^k r, ..., α^2 r, α^2 r, ..., α r, α r, α r, r]
            for j, pos in enumerate(reversed(sep_pos)):
                # Discard reward if pos is larger than response length
                if pos >= len(response_ids):
                    break
                if j == 0:
                    reward_tensor[i, pos] = score
                else:
                    for k in range(pos, prev_pos):
                        reward_tensor[i, k] = score
                prev_pos = pos
                score *= self.alpha
            
            # Discard reward if pos is larger than response length
            if sep_pos[0] < len(response_ids):
                for k in range(sep_pos[0]):
                    reward_tensor[i, k] = score

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            ground_truth = data_item.non_tensor_batch["extra_info"]["response"]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
