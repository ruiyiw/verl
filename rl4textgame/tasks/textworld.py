import json
import os
from typing import Dict

import textworld


def compute_score(solution_str: str, ground_truth: str, extra_info: Dict) -> float:
    try:
        try:
            solution = json.loads(solution_str)
        except json.JSONDecodeError:
            print("Failed to parse JSON")
            return 0.0
        
       # Extract action sequences with error handling
        action_seq = []
        try:
            for elem in solution["trajectory"]:
                if not isinstance(elem, dict):
                    print("Invalid trajectory element format")
                    return 0.0
                
                if elem.get("type") == "action":
                    if "content" not in elem:
                        print("Missing 'content' in action element")
                        return 0.0
                    action_seq.append(elem["content"])
        except Exception as e:
            print(f"Error extracting action sequence: {e}")
            return 0.0

        print(action_seq)
        
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
            game_state, reward, done = env.step(str(action))
            if done:
                return 1.0
        return 0.0

    except json.JSONDecodeError:
        print("Failed to parse generated output as it is not JSON formatted.")
        return 0.0

