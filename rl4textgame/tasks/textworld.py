import json
import os
from typing import Dict

import textworld
import textworld.agents


def compute_score(solution_str: str, ground_truth: str, extra_info: Dict) -> float:
    try:
        # Extract action sequences:
        solution = json.loads(solution_str)
        action_seq = []
        for elem in solution["trajectory"]:
            if elem["type"] == "action":
                action_seq.append(elem["content"])
        
        # Load textworld game:
        game_file = os.path.join(extra_info["game_path"], f"{ground_truth}.z8")
        textworld_infos = textworld.EnvInfos(
            feedback=True,    # Response from the game after typing a text command.
            description=True, # Text describing the room the player is currently in.
            inventory=True    # Text describing the player's inventory.
        )
        game_agent = textworld.agents.HumanAgent()
        env = textworld.start(game_file, request_infos=textworld_infos, wrappers=game_agent.wrappers)
        game_agent.reset(env)
        game_state = env.reset()

        # Execute solution sequence in env
        for action in action_seq:
            game_state, reward, done = env.step(action)
            if done:
                return 1.0
        return 0.0

    except json.JSONDecodeError:
        return 0.0

