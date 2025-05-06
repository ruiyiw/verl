from pydantic import BaseModel, Field
from typing import List, Literal, Union
import json
import argparse
import os


class State(BaseModel):
    type: Literal["state"] = "state"
    content: str = Field(..., description="The current state description of the game environment")

class Action(BaseModel):
    type: Literal["action"] = "action"
    content: str = Field(..., description="The action taken from this state")

TrajectoryEntry = Union[State, Action]


class TextworldSchema:
    class Trajectory(BaseModel):
        trajectory: List[TrajectoryEntry] = Field(..., 
            description="The sequence of alternating states and actions forming the game trajectory.")


def main(args):
    if args.task == "textworld":
        json_schema = TextworldSchema.Trajectory.model_json_schema()
        with open(os.path.join(args.local_dir, f"{args.task}.json"), 'w') as f:
            f.write(json.dumps(json_schema))
    else:
        raise NotImplementedError
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PPO data for verl post-training.")
    parser.add_argument("--task", type=str, required=True, choices=["textworld"], help="Choose the task.")
    parser.add_argument("--local_dir", type=str, required=True, help="Path to the local directory to save the json schema.")
    
    args = parser.parse_args()
    os.makedirs(args.local_dir, exist_ok=True)
    main(args)