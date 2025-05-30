import textworld
from textworld.core import GameState
from typing import List
import re


class GameEnv():
    def __init__(self, game_file: str):
        self.init_game_env(game_file)
        

    def init_game_env(self, game_file: str):
        # Load textworld game:
        textworld_infos = textworld.EnvInfos(
            feedback=True,    # Response from the game after typing a text command.
            description=True, # Text describing the room the player is currently in.
            inventory=True    # Text describing the player's inventory.
        )
        game_agent = textworld.agents.HumanAgent()
        self.env = textworld.start(game_file, request_infos=textworld_infos, wrappers=game_agent.wrappers)
        game_agent.reset(self.env)
        game_state = self.env.reset()


    def format_observation(self, game_state: GameState):
        """
        Get the observation at each step, consisting of `room description`, `game feedback`, `inventory`, and `last action`, according to KG-A2C paper.
        Descriptions:
            - room description: agent's current location
            - game feedback: outputs of game simulator given agent's previous action
            - inventory: agent's inventory list
        """
        room_id = None
        for s in game_state._facts:
            if s.name == "at" and s.arguments[0].name == "P":
                room_id = s.arguments[1].name
                break
        if not room_id or room_id not in game_state.game.infos:
            raise ValueError
        room_desc = f"You are now in the {game_state.game.infos[room_id].name}.\n"

        feedback = self._extract_essential_feedback(game_state.feedback) + '\n'

        inventory = game_state.inventory

        obs = room_desc + feedback + inventory

        return obs

    
    def _safe_step(self, command: str):
        """Safely execute a step, falling back to empty command on Unicode errors"""
        try:
            return self.env.step(command)
        except:
            print(f"Game backend error with command '{command}'")
            return self.env.step("")


    def replay(self, commands: List[str]):
        """
        Reset the game environment, restart game, and replay the sequence of actions.
        """
        is_winning = False
        for command in commands:
            game_state, reward, done = self._safe_step(command)
            obs = self.format_observation(game_state)
            if done:
                is_winning = True
                break  # Game completed early
        return obs, is_winning
    

    def _extract_essential_feedback(self, text):
        """
        Extract essential feedback from TextWorld output.
        This extracts room descriptions and action feedback without duplication.
        """
        result = []
        
        # Extract room descriptions - between room header and prompt
        room_pattern = r'-= (.+?) =-\n([\s\S]*?)(?=\s*>|$)'
        room_matches = re.finditer(room_pattern, text)
        
        for match in room_matches:
            room_desc = match.group(2).strip()
            if room_desc:
                # Split by lines and add non-empty ones
                lines = [line.strip() for line in room_desc.split('\n') if line.strip()]
                result.extend(lines)
        
        # Extract action feedback - lines before a prompt that aren't part of headers
        action_pattern = r'^([^-=>\n][^\n]*?)(?=\n\s*>)'
        action_matches = re.finditer(action_pattern, text, re.MULTILINE)
        
        for match in action_matches:
            action = match.group(1).strip()
            if action and action not in result:
                result.append(action)
        
        # Remove duplicates while preserving order
        return '\n'.join(dict.fromkeys(result))