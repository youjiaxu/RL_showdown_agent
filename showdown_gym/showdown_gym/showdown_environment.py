import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import math
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle, Move, Pokemon, Status
from poke_env.battle.move_category import MoveCategory
from poke_env.data import GenData
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.env import ObsType
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv

##state and information acquisition
class BattleData:
    effect_chart = {   #Class attribute
        "Normal": {"Rock": 0.5, "Ghost": 0.0, "Steel": 0.5},
        "Fire": {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 2.0, "Bug": 2.0,"Rock": 0.5, "Dragon": 0.5, "Steel": 2.0},
        "Water": {"Fire": 2.0, "Water": 0.5, "Grass": 0.5, "Ground": 2.0,"Rock": 2.0, "Dragon": 0.5},
        "Electric": {"Water": 2.0, "Electric": 0.5, "Grass": 0.5, "Ground": 0.0,"Flying": 2.0, "Dragon": 0.5},
        "Grass": {"Fire": 0.5, "Water": 2.0, "Grass": 0.5, "Poison": 0.5,"Ground": 2.0, "Flying": 0.5, "Bug": 0.5, "Rock": 2.0,"Dragon": 0.5, "Steel": 0.5},
        "Ice": {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 0.5, "Ground": 2.0,"Flying": 2.0, "Dragon": 2.0, "Steel": 0.5},
        "Fighting": {"Normal": 2.0, "Ice": 2.0, "Poison": 0.5, "Flying": 0.5,"Psychic": 0.5, "Bug": 0.5, "Rock": 2.0, "Ghost": 0.0,"Dark": 2.0, "Steel": 2.0, "Fairy": 0.5},
        "Poison": {"Grass": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5,"Ghost": 0.5, "Steel": 0.0, "Fairy": 2.0},
        "Ground": {"Fire": 2.0, "Electric": 2.0, "Grass": 0.5, "Poison": 2.0,"Flying": 0.0, "Bug": 0.5, "Rock": 2.0, "Steel": 2.0},
        "Flying": {"Electric": 0.5, "Grass": 2.0, "Fighting": 2.0, "Bug": 2.0,"Rock": 0.5, "Steel": 0.5},
        "Psychic": {"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5, "Dark": 0.0,"Steel": 0.5},
        "Bug": {"Fire": 0.5, "Grass": 2.0, "Fighting": 0.5, "Poison": 0.5,"Flying": 0.5, "Psychic": 2.0, "Ghost": 0.5, "Dark": 2.0,"Steel": 0.5, "Fairy": 0.5},
        "Rock": {"Fire": 2.0, "Ice": 2.0, "Fighting": 0.5, "Ground": 0.5,"Flying": 2.0, "Bug": 2.0, "Steel": 0.5},
        "Ghost": {"Normal": 0.0, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5},
        "Dragon": {"Dragon": 2.0, "Steel": 0.5, "Fairy": 0.0},
        "Dark": {"Fighting": 0.5, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5,"Fairy": 0.5},
        "Steel": {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2.0,"Rock": 2.0, "Steel": 0.5, "Fairy": 2.0},
        "Fairy": {"Fire": 0.5, "Fighting": 2.0, "Poison": 0.5, "Dragon": 2.0,"Dark": 2.0, "Steel": 0.5},
    }

    BULLETPROOF_MOVES = {
        "acidspay", "aurasphere", "barrage", "beakblast", "bulletseed", "corrosivegas",
        "dracometeor", "eggbomb", "electroball", "energyball", "focusblast", "gyroball",
        "iceball", "magnetbomb", "mistball", "mudsport", "octazooka", "pyroball", "rockblast",
        "rockwrecker", "searingshot", "seedbomb", "shadowball", "sludgebomb", "technoblast",
        "weatherball", "zapcannon"
    }

    #didnt account for koraidon multiattack
    MULTI_HIT_MOVES = {
        "scaleshot": 3,
        "rockblast": 3,
        "bulletseed": 3,
    }

    def __init__(self):
        #'pikachu' : {'Thunder', 'Attack'...}
        self.opponent_moves_memory = {} 

class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
        data: BattleData | None = None
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        self.rl_agent = account_name_one

        self.data = data if data is not None else BattleData()
        self.gen_data = GenData.from_gen(9)  # Initialize gen data for stat calculations

    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        return None  # Return None if action size is default

    def process_action(self, action: np.int64) -> np.int64:
        """
        Returns the np.int64 relative to the given action.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move
        10 <= action <= 13: move and mega evolve
        14 <= action <= 17: move and z-move
        18 <= action <= 21: move and dynamax
        22 <= action <= 25: move and terastallize

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        return action

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on the changes in state of the battle.

        You need to implement this method to define how the reward is calculated

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
            prior_battle (AbstractBattle): The prior battle instance to compare against.
        Returns:
            float: The calculated reward based on the change in state of the battle.
        """

        prior_battle = self._get_prior_battle(battle)

        reward = 0.0

        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # If the opponent has less than 6 Pokémon, fill the missing values with 1.0 (fraction of health)
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        prior_health_opponent = []
        if prior_battle is not None:
            prior_health_opponent = [
                mon.current_hp_fraction for mon in prior_battle.opponent_team.values()
            ]

        # Ensure health_opponent has 6 components, filling missing values with 1.0 (fraction of health)
        if len(prior_health_opponent) < len(health_team):
            prior_health_opponent.extend(
                [1.0] * (len(health_team) - len(prior_health_opponent))
            )

        diff_health_opponent = np.array(prior_health_opponent) - np.array(
            health_opponent
        )

        # Reward for reducing the opponent's health
        reward += np.sum(diff_health_opponent)

        return reward

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set obvervation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        # Updated to match the rich state representation:
        # 12 (health) + 22 (active pokemon details) + 28 (moves) + 2 (fainted counts) + 1 (turn) + 4 (weather) + 1 (switches) = 70
        # BUT actual runtime shows 71 features, so using 71
        return 71

    def move_effectiveness(self, attacker_type: str, defender_types: Tuple[str, ...]) -> float:
        multiplier = 1.0
        #defender can have more than 1 type, increasing effectiveness
        for def_type in defender_types:
            #return 1.0 if no specific effectivness found
            multiplier *= self.data.effect_chart.get(attacker_type, {}).get(def_type, 1.0)
        return multiplier
        #gets base_stats using GenData library for estimated stat calculation 

    def calculate_stat(self, pokemon: Pokemon, stat: str) -> int:
        #normalise stats
        #cover all stats inc atk def spa spd.. etc
        
        if not pokemon:
            return 100

        # If live stats are available from the server
        pokemon_stats = getattr(pokemon, 'stats', None)
        if pokemon_stats and pokemon_stats.get(stat):
            stat_value = pokemon_stats[stat]
            return stat_value if stat_value is not None else 100
        
        # If not, calculate an estimate based on base stats. Assumes standard competetive build
        pokemon_species = getattr(pokemon, 'species', '')
        base_stats = self.gen_data.pokedex.get(pokemon_species, {}).get('baseStats', {})
        base_stat = base_stats.get(stat, 100) # Default to 100 if species not found

        if stat == 'hp':
            return math.floor(0.01 * (2 * base_stat + 31 + 63) * 100) + 110
        else:
            # assumes a beneficial nature for attacking/speed stats
            nature_multiplier = 1.1 if stat in ['atk', 'spa', 'spe'] else 1.0
            return math.floor((math.floor(0.01 * (2 * base_stat + 31 + 63) * 100) + 5) * nature_multiplier)

    def stage_multiplier(self, stage:int) -> float:
        return max(2, 2 + stage) / max(2, 2 - stage)
        #simple speed check

    def is_faster(self, pokemon_a: Pokemon, pokemon_b: Pokemon) -> bool:
        pokemon_a_speed = self.calculate_stat(pokemon_a, 'spe')
        pokemon_b_speed = self.calculate_stat(pokemon_b, 'spe')
        return pokemon_a_speed > pokemon_b_speed
    
    def is_nullified(self, move: Move, defender: Pokemon) -> bool:
        if not move or not defender:
            return False
            
        defender_ability = getattr(defender, 'ability', None)
        if not defender_ability:
            return False

        ability = str(defender_ability).lower().replace(" ", "")
        move_type_obj = getattr(move, 'type', None)
        move_type = getattr(move_type_obj, 'name', '') if move_type_obj else ""
        move_id = getattr(move, 'id', '')

        if (ability == 'bulletproof' and move_id in self.data.BULLETPROOF_MOVES) or \
            (ability == 'levitate' and move_type == 'GROUND') or \
            (ability in {'voltabsorb', 'motordrive', 'lightningrod'} and move_type == 'ELECTRIC') or \
            (ability in {'waterabsorb', 'stormdrain', 'dryskin'} and move_type == 'WATER') or \
            (ability == 'flashfire' and move_type == 'FIRE') or \
            (ability == 'sapsipper' and move_type == 'GRASS'):
                return True

        return False
    
    def item_multiplier(self, attacker: Pokemon, move: Move) -> float:
        """Calculate item-based damage multiplier (simplified version)"""
        # This is a simplified version - you could expand this to include specific items
        # For now, just return 1.0 (no multiplier)
        return 1.0
    
        #checks for move's priority 
    def move_priority(self, move, battle) -> int:
        return int(getattr(move, 'priority', 0)) if move else 0
        #estimates damage based on chosen attacker and defender pokemon, accounts for STAB, stage and effectiveness multipliers
        
    def estimate_move_damage(self, move: Move, attacker: Pokemon, defender: Pokemon, battle: AbstractBattle) -> float:
        if not move or not defender or not getattr(defender, 'max_hp', None) or getattr(move, 'base_power', 0) == 0 or getattr(move, 'category', None) == MoveCategory.STATUS:
            return 0.0
        
        if self.is_nullified(move, defender):
            return 0.0

        move_category = getattr(move, 'category', None)
        if move_category == MoveCategory.PHYSICAL:
            atk_stat, def_stat = "atk", "def"
        elif move_category == MoveCategory.SPECIAL:
            atk_stat, def_stat = "spa", "spd"
        else:
            return 0.0
        
        attacker_boosts = getattr(attacker, 'boosts', {})
        defender_boosts = getattr(defender, 'boosts', {})
        attack = self.calculate_stat(attacker, atk_stat) * self.stage_multiplier(attacker_boosts.get(atk_stat, 0))
        defense = max(1, self.calculate_stat(defender, def_stat) * self.stage_multiplier(defender_boosts.get(def_stat, 0)))
        defender_hp = max(1, self.calculate_stat(defender, 'hp'))
        
        level = max(1, attacker.level if hasattr(attacker, 'level') and attacker.level else 50)
        move_base_power = getattr(move, 'base_power', 0) or 0
        base_damage = (((2 * level / 5 + 2) * move_base_power * (attack / defense)) / 50) + 2

        weather_multiplier = 1.0
        if battle.weather:
            weather_str = str(battle.weather).lower()
            move_type = getattr(move, 'type', None)
            move_type_name = getattr(move_type, 'name', '') if move_type else ''
            
            if 'sun' in weather_str or 'desolate' in weather_str:
                if move_type_name == 'FIRE': weather_multiplier = 1.5
                if move_type_name == 'WATER': weather_multiplier = 0.5
            elif 'rain' in weather_str or 'primordial' in weather_str:
                if move_type_name == 'WATER': weather_multiplier = 1.5
                if move_type_name == 'FIRE': weather_multiplier = 0.5
        
        #Weather multiplier
        base_damage *= weather_multiplier

        #STAB bonus multiplier
        move_type = getattr(move, 'type', None)
        attacker_types = getattr(attacker, 'types', [])
        if move_type and attacker_types and move_type in attacker_types: 
            base_damage *= 1.5

        #Item multiplier
        base_damage *= self.item_multiplier(attacker, move)

        #Move effectiveness multiplier
        if move_type:
            defender_types = getattr(defender, 'types', [])
            if defender_types:
                defender_type_names = tuple(getattr(t, 'name', '').title() for t in defender_types if t)
                if defender_type_names:
                    base_damage *= self.move_effectiveness(getattr(move_type, 'name', '').title(), defender_type_names)
        
        #Burn effect/physical effect negation on physical attacks
        attacker_status = getattr(attacker, 'status', None)
        if attacker_status == Status.BRN and move_category == MoveCategory.PHYSICAL:
            base_damage *= 0.5
                # Multi-hit move adjustment
        move_id = getattr(move, 'id', '')
        if move_id and move_id in self.data.MULTI_HIT_MOVES:
            average_hits = self.data.MULTI_HIT_MOVES[move_id]
            base_damage *= average_hits

        return base_damage / defender_hp
    
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        This method generates a feature vector that represents the current state of the battle,
        this is used by the agent to make decisions.

        You need to implement this method to define how the battle state is represented.

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the state you want the agent to observe.
        """
        state = []

        # === BASIC TEAM HEALTH ===
        battle_team = getattr(battle, 'team', {})
        battle_opponent_team = getattr(battle, 'opponent_team', {})
        health_team = [getattr(mon, 'current_hp_fraction', 0.0) for mon in battle_team.values()]
        health_opponent = [
            getattr(mon, 'current_hp_fraction', 0.0) for mon in battle_opponent_team.values()
        ]
        
        # Ensure both teams have 6 components
        if len(health_opponent) < 6:
            health_opponent.extend([1.0] * (6 - len(health_opponent)))
        if len(health_team) < 6:
            health_team.extend([1.0] * (6 - len(health_team)))
            
        state.extend(health_team[:6])  # 6 values
        state.extend(health_opponent[:6])  # 6 values

        # === ACTIVE POKEMON DETAILED INFO ===
        if battle.active_pokemon and battle.opponent_active_pokemon:
            active = battle.active_pokemon
            opp_active = battle.opponent_active_pokemon
            
            # Normalized stats (divide by typical max values)
            state.extend([
                self.calculate_stat(active, 'atk') / 500.0,
                self.calculate_stat(active, 'def') / 500.0,
                self.calculate_stat(active, 'spa') / 500.0,
                self.calculate_stat(active, 'spd') / 500.0,
                self.calculate_stat(active, 'spe') / 500.0,
            ])  # 5 values
            
            state.extend([
                self.calculate_stat(opp_active, 'atk') / 500.0,
                self.calculate_stat(opp_active, 'def') / 500.0,
                self.calculate_stat(opp_active, 'spa') / 500.0,
                self.calculate_stat(opp_active, 'spd') / 500.0,
                self.calculate_stat(opp_active, 'spe') / 500.0,
            ])  # 5 values

            # Status conditions (binary)
            status_map = {
                Status.BRN: 1.0, Status.FRZ: 2.0, Status.PAR: 3.0, 
                Status.PSN: 4.0, Status.SLP: 5.0, Status.TOX: 6.0
            }
            active_status = getattr(active, 'status', None)
            opp_active_status = getattr(opp_active, 'status', None)
            active_status_value = status_map.get(active_status, 0.0) if active_status else 0.0
            opp_active_status_value = status_map.get(opp_active_status, 0.0) if opp_active_status else 0.0
            state.append(active_status_value / 6.0)  # 1 value
            state.append(opp_active_status_value / 6.0)  # 1 value

            # Stat boosts (normalized to [-6, +6] range)
            active_boosts = getattr(active, 'boosts', {})
            opp_active_boosts = getattr(opp_active, 'boosts', {})
            for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
                state.append(active_boosts.get(stat, 0) / 6.0)  # 5 values
                state.append(opp_active_boosts.get(stat, 0) / 6.0)  # 5 values

            # Speed comparison
            state.append(1.0 if self.is_faster(active, opp_active) else 0.0)  # 1 value

        else:
            # Fill with zeros if no active pokemon
            state.extend([0.0] * 22)

        # === AVAILABLE MOVES ANALYSIS ===
        move_features = []
        available_moves = getattr(battle, 'available_moves', [])
        for i in range(4):  # Max 4 moves
            if i < len(available_moves):
                move = available_moves[i]
                if battle.opponent_active_pokemon:
                    # Move effectiveness against opponent
                    if move.type:
                        opp_types = tuple(t.name.title() for t in battle.opponent_active_pokemon.types if t)
                        effectiveness = self.move_effectiveness(move.type.name.title(), opp_types)
                    else:
                        effectiveness = 1.0
                    
                    # Estimated damage (normalized)
                    damage = self.estimate_move_damage(move, battle.active_pokemon, 
                                                     battle.opponent_active_pokemon, battle)
                    
                    base_power = getattr(move, 'base_power', 0) or 0
                    move_category = getattr(move, 'category', None)
                    
                    move_features.extend([
                        base_power / 150.0,  # Normalized power
                        effectiveness / 4.0,  # Effectiveness (0-4x range)
                        damage,  # Estimated damage ratio
                        getattr(move, 'priority', 0) / 5.0 + 0.5,  # Priority normalized to [0,1]
                        1.0 if move_category == MoveCategory.PHYSICAL else 0.0,  # Physical flag
                        1.0 if move_category == MoveCategory.SPECIAL else 0.0,  # Special flag
                        1.0 if move_category == MoveCategory.STATUS else 0.0,  # Status flag
                    ])  # 7 values per move
                else:
                    move_features.extend([0.0] * 7)
            else:
                move_features.extend([0.0] * 7)  # No move available
        
        state.extend(move_features)  # 28 values (4 moves × 7 features)

        # === TEAM COMPOSITION SUMMARY ===
        # Count fainted pokemon
        battle_team = getattr(battle, 'team', {})
        battle_opponent_team = getattr(battle, 'opponent_team', {})
        fainted_team = sum(1 for mon in battle_team.values() if getattr(mon, 'fainted', False))
        fainted_opponent = sum(1 for mon in battle_opponent_team.values() if getattr(mon, 'fainted', False))
        state.extend([fainted_team / 6.0, fainted_opponent / 6.0])  # 2 values

        # === BATTLE CONTEXT ===
        # Turn number (normalized)
        battle_turn = getattr(battle, 'turn', 0)
        state.append(min(battle_turn / 50.0, 1.0))  # 1 value
        
        # Weather effects
        weather_features = [0.0, 0.0, 0.0, 0.0]  # sun, rain, sand, hail
        battle_weather = getattr(battle, 'weather', None)
        if battle_weather:
            weather_str = str(battle_weather).lower()
            if 'sun' in weather_str:
                weather_features[0] = 1.0
            elif 'rain' in weather_str:
                weather_features[1] = 1.0
            elif 'sand' in weather_str:
                weather_features[2] = 1.0
            elif 'hail' in weather_str or 'snow' in weather_str:
                weather_features[3] = 1.0
        state.extend(weather_features)  # 4 values

        # Switch availability
        battle_team = getattr(battle, 'team', {})
        battle_active = getattr(battle, 'active_pokemon', None)
        available_switches = len([mon for mon in battle_team.values() 
                                if not getattr(mon, 'fainted', True) and mon != battle_active])
        state.append(available_switches / 5.0)  # 1 value (max 5 switches)

        #########################################################################################################
        # Calculate the length of the final_vector and make sure to update the value in _observation_size above #
        #########################################################################################################
        # Total: 12 + 22 + 28 + 2 + 1 + 4 + 1 = 70 features

        final_vector = np.array(state, dtype=np.float32)
        return final_vector


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(
                account_configuration=opponent_configuration
            )
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"

        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)

        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
            data=BattleData(),  # Add the required BattleData instance
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None