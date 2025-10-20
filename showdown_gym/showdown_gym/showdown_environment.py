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
        
        # V2 Enhancement: Action history and temporal context tracking
        self.action_history = []  # Store last 3 actions with context
        self.last_stat_boost_turns = {}  # Track when each stat was last boosted
        self.momentum_tracker = {'damage_dealt': [], 'damage_taken': []}  # Track recent damage trends

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
        # V2 Enhancement: Track action for temporal context
        if hasattr(self, 'battle1') and self.battle1 and getattr(self.battle1, 'turn', 0) > 0:
            self.track_action(int(action), self.battle1)
        
        return action

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add comprehensive metrics for quantitative and qualitative analysis
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            
            # === CORE BATTLE OUTCOMES ===
            info[agent]["win"] = self.battle1.won
            info[agent]["battle_finished"] = getattr(self.battle1, 'battle_finished', False)
            info[agent]["battle_turns"] = getattr(self.battle1, 'turn', 0)
            
            # === REWARD COMPONENT BREAKDOWN ===
            try:
                prior_battle = self._get_prior_battle(self.battle1)
                damage_component = self._calculate_damage_delta(self.battle1, prior_battle)
                ko_component = self._calculate_ko_reward(self.battle1, prior_battle)
                outcome_component = self._calculate_outcome_reward(self.battle1)
                
                info[agent]["reward_damage"] = float(damage_component)
                info[agent]["reward_ko"] = float(ko_component)
                info[agent]["reward_outcome"] = float(outcome_component)
                info[agent]["reward_total"] = float(damage_component + ko_component + outcome_component)
            except (AttributeError, Exception):
                info[agent]["reward_damage"] = 0.0
                info[agent]["reward_ko"] = 0.0
                info[agent]["reward_outcome"] = 0.0
                info[agent]["reward_total"] = 0.0
            
            # === TEAM PERFORMANCE METRICS ===
            battle_team = getattr(self.battle1, 'team', {})
            battle_opponent_team = getattr(self.battle1, 'opponent_team', {})
            
            # Pokemon remaining and HP
            team_pokemon_remaining = sum(1 for mon in battle_team.values() if not getattr(mon, 'fainted', True))
            opponent_pokemon_remaining = sum(1 for mon in battle_opponent_team.values() if not getattr(mon, 'fainted', True))
            
            team_total_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) for mon in battle_team.values())
            opponent_total_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) for mon in battle_opponent_team.values())
            
            info[agent]["team_pokemon_remaining"] = team_pokemon_remaining
            info[agent]["opponent_pokemon_remaining"] = opponent_pokemon_remaining
            info[agent]["team_total_hp"] = float(team_total_hp)
            info[agent]["opponent_total_hp"] = float(opponent_total_hp)
            info[agent]["hp_advantage"] = float(team_total_hp - opponent_total_hp)
            
            # Battle efficiency metrics
            if info[agent]["win"] and info[agent]["battle_turns"] > 0:
                info[agent]["victory_efficiency"] = float(6 - info[agent]["battle_turns"] / 10.0)  # Higher = faster win
                info[agent]["victory_hp_retained"] = float(team_total_hp / 6.0)  # Higher = won with more HP
            else:
                info[agent]["victory_efficiency"] = 0.0
                info[agent]["victory_hp_retained"] = 0.0
            
            # === STRATEGIC BEHAVIOR ANALYSIS ===
            # Action pattern metrics from history
            if self.action_history:
                action_types = [action.get('action_type', 'unknown') for action in self.action_history]
                total_actions = len(action_types)
                
                info[agent]["total_actions_taken"] = total_actions
                info[agent]["move_percentage"] = float(sum(1 for a in action_types if a.startswith('move')) / max(1, total_actions))
                info[agent]["switch_percentage"] = float(sum(1 for a in action_types if a == 'switch') / max(1, total_actions))
                info[agent]["mega_usage"] = float(sum(1 for a in action_types if 'mega' in a) / max(1, total_actions))
                info[agent]["z_move_usage"] = float(sum(1 for a in action_types if 'z' in a) / max(1, total_actions))
                
                # Strategy coherence (same action type clustering)
                strategy_changes = sum(1 for i in range(1, len(action_types)) 
                                    if action_types[i] != action_types[i-1])
                info[agent]["strategy_coherence"] = float(1.0 - strategy_changes / max(1, total_actions - 1))
            else:
                info[agent]["total_actions_taken"] = 0
                info[agent]["move_percentage"] = 0.0
                info[agent]["switch_percentage"] = 0.0
                info[agent]["mega_usage"] = 0.0
                info[agent]["z_move_usage"] = 0.0
                info[agent]["strategy_coherence"] = 0.0
            
            # === BATTLE PHASE ANALYSIS ===
            battle_phase = self.calculate_battle_phase(self.battle1)
            info[agent]["battle_phase"] = float(battle_phase)
            
            if battle_phase < 0.3:
                info[agent]["phase_category"] = "early"
            elif battle_phase < 0.7:
                info[agent]["phase_category"] = "mid"
            else:
                info[agent]["phase_category"] = "late"
            
            # === MOMENTUM AND ADAPTATION METRICS ===
            try:
                prior_battle = self._get_prior_battle(self.battle1)
                damage_dealt_momentum, damage_taken_momentum = self.calculate_momentum(self.battle1, prior_battle)
                info[agent]["momentum_damage_dealt"] = float(damage_dealt_momentum)
                info[agent]["momentum_damage_taken"] = float(damage_taken_momentum)
                info[agent]["momentum_ratio"] = float(damage_dealt_momentum / max(0.001, damage_taken_momentum))
            except (AttributeError, Exception):
                info[agent]["momentum_damage_dealt"] = 0.0
                info[agent]["momentum_damage_taken"] = 0.0
                info[agent]["momentum_ratio"] = 1.0
            
            # === POKEMON UTILIZATION ANALYSIS ===
            # Track which Pokemon were used and how effectively
            pokemon_used = set()
            for action in self.action_history:
                if action.get('active_pokemon'):
                    pokemon_used.add(action['active_pokemon'])
            
            info[agent]["unique_pokemon_used"] = len(pokemon_used)
            info[agent]["team_utilization"] = float(len(pokemon_used) / 6.0)  # How much of team was used
            
            # === MOVE EFFECTIVENESS ANALYSIS ===
            if self.battle1.active_pokemon and hasattr(self.battle1, 'available_moves'):
                available_moves = getattr(self.battle1, 'available_moves', [])
                if available_moves and self.battle1.opponent_active_pokemon:
                    # Calculate average move effectiveness available
                    total_effectiveness = 0.0
                    move_count = 0
                    
                    for move in available_moves:
                        try:
                            if move.type and self.battle1.opponent_active_pokemon.types:
                                opp_types = tuple(t.name.title() for t in self.battle1.opponent_active_pokemon.types 
                                                if t and hasattr(t, 'name'))
                                if opp_types:
                                    effectiveness = self.move_effectiveness(move.type.name.title(), opp_types)
                                    total_effectiveness += effectiveness
                                    move_count += 1
                        except (AttributeError, TypeError):
                            pass
                    
                    info[agent]["avg_move_effectiveness"] = float(total_effectiveness / max(1, move_count))
                else:
                    info[agent]["avg_move_effectiveness"] = 1.0
            else:
                info[agent]["avg_move_effectiveness"] = 1.0
            
            # === ENVIRONMENTAL FACTORS ===
            # Weather usage and adaptation
            weather = getattr(self.battle1, 'weather', None)
            info[agent]["weather_present"] = 1.0 if weather else 0.0
            if weather:
                info[agent]["weather_type"] = str(weather).lower()
            else:
                info[agent]["weather_type"] = "none"
            
            # === LEARNING PROGRESS INDICATORS ===
            # These help track if the agent is learning better strategies over time
            if hasattr(self.battle1, 'active_pokemon') and self.battle1.active_pokemon:
                active_boosts = getattr(self.battle1.active_pokemon, 'boosts', {})
                total_positive_boosts = sum(max(0, boost) for boost in active_boosts.values())
                total_negative_boosts = sum(min(0, boost) for boost in active_boosts.values())
                
                info[agent]["positive_stat_boosts"] = float(total_positive_boosts)
                info[agent]["negative_stat_boosts"] = float(abs(total_negative_boosts))
                info[agent]["net_stat_advantage"] = float(total_positive_boosts + total_negative_boosts)
            else:
                info[agent]["positive_stat_boosts"] = 0.0
                info[agent]["negative_stat_boosts"] = 0.0
                info[agent]["net_stat_advantage"] = 0.0
            
            # === DECISION QUALITY METRICS ===
            # Track decisions that led to good/bad outcomes
            if self.action_history:
                last_action = self.action_history[-1]
                
                # Was the last action taken when agent had type advantage?
                info[agent]["last_action_type"] = last_action.get('action_type', 'unknown')
                info[agent]["last_action_hp"] = float(last_action.get('hp_fraction', 0.0))
                
                # Action timing quality (e.g., switching when low HP)
                if last_action.get('action_type') == 'switch' and last_action.get('hp_fraction', 1.0) < 0.3:
                    info[agent]["good_defensive_switch"] = 1.0
                else:
                    info[agent]["good_defensive_switch"] = 0.0

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Conservative Strategic Reward System (Option A)
        
        Simplified tri-component reward focused on:
        1. Damage delta (dense, small magnitude)
        2. KO events (sparse, medium magnitude, scaled by game stage) 
        3. Win/Loss outcomes (sparse, large magnitude)
        
        This reduces reward fragility and exploitation while maintaining strategic guidance.
        """
        try:
            prior_battle = self._get_prior_battle(battle)
        except AttributeError:
            prior_battle = None
        
        # Component 1: Damage delta (normalized HP advantage change)
        damage_reward = self._calculate_damage_delta(battle, prior_battle)
        
        # Component 2: KO events (scaled by game stage)
        ko_reward = self._calculate_ko_reward(battle, prior_battle)
        
        # Component 3: Win/Loss outcome
        outcome_reward = self._calculate_outcome_reward(battle)
        
        # Conservative magnitudes: damage ±0.5, KO ±1.0, win/loss ±2.0
        total_reward = damage_reward + ko_reward + outcome_reward
        
        # Tight clipping for stability
        total_reward = np.clip(total_reward, -3.0, 3.0)
        
        return total_reward
    
    def _calculate_damage_delta(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """Calculate normalized HP advantage change (dense reward component)"""
        if not prior_battle:
            return 0.0
        
        # Current HP advantage
        current_advantage = self._calculate_hp_advantage(battle)
        
        # Prior HP advantage  
        prior_advantage = self._calculate_hp_advantage(prior_battle)
        
        # Delta in advantage (positive = we improved our position)
        advantage_delta = current_advantage - prior_advantage
        
        # Conservative scaling: ±0.5 max
        return np.clip(advantage_delta * 0.5, -0.5, 0.5)
    
    def _calculate_ko_reward(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """Calculate KO event rewards (sparse, scaled by game stage)"""
        if not prior_battle:
            return 0.0
        
        reward = 0.0
        
        # Count KOs
        prior_fainted_opponent = sum(1 for mon in prior_battle.opponent_team.values() if getattr(mon, 'fainted', False))
        current_fainted_opponent = sum(1 for mon in battle.opponent_team.values() if getattr(mon, 'fainted', False))
        
        prior_fainted_team = sum(1 for mon in prior_battle.team.values() if getattr(mon, 'fainted', False))
        current_fainted_team = sum(1 for mon in battle.team.values() if getattr(mon, 'fainted', False))
        
        # Our KOs (positive reward, scaled by remaining opponents)
        opponent_kos = current_fainted_opponent - prior_fainted_opponent
        if opponent_kos > 0:
            # Late-game KOs worth more (fewer remaining opponents = higher multiplier)
            remaining_opponents = 6 - current_fainted_opponent
            stage_multiplier = max(1.0, 7 - remaining_opponents)  # 1.0 to 6.0
            reward += 1.0 * stage_multiplier * opponent_kos
        
        # Our losses (negative reward)  
        team_kos = current_fainted_team - prior_fainted_team
        if team_kos > 0:
            remaining_team = 6 - current_fainted_team
            stage_multiplier = max(1.0, 7 - remaining_team)
            reward -= 1.0 * stage_multiplier * team_kos
        
        return np.clip(reward, -2.0, 2.0)
    
    def _calculate_outcome_reward(self, battle: AbstractBattle) -> float:
        """Calculate win/loss rewards (sparse, high magnitude)"""
        if not getattr(battle, 'battle_finished', False):
            return 0.0
        
        if getattr(battle, 'won', False):
            return 2.0  # Win bonus
        else:
            return -2.0  # Loss penalty
    
    def _calculate_strategic_outcomes(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Reward meaningful strategic outcomes, not just immediate damage.
        Focus on: KOs, major position changes, tempo shifts, setup completion
        """
        if not prior_battle:
            return 0.0
            
        reward = 0.0
        
        # === KNOCKOUT REWARDS (High value, sparse) ===
        prior_fainted_opponent = sum(1 for mon in prior_battle.opponent_team.values() if getattr(mon, 'fainted', False))
        current_fainted_opponent = sum(1 for mon in battle.opponent_team.values() if getattr(mon, 'fainted', False))
        
        prior_fainted_team = sum(1 for mon in prior_battle.team.values() if getattr(mon, 'fainted', False))
        current_fainted_team = sum(1 for mon in battle.team.values() if getattr(mon, 'fainted', False))
        
        # KO rewards scaled by remaining opponent Pokemon (more valuable late game)
        opponent_remaining = 6 - current_fainted_opponent
        ko_multiplier = 7 - opponent_remaining  # Higher reward for later KOs
        
        if current_fainted_opponent > prior_fainted_opponent:
            reward += 2.0 * ko_multiplier  # Major reward for KOs
        
        if current_fainted_team > prior_fainted_team:
            reward -= 2.0 * (7 - (6 - current_fainted_team))  # Penalty for losing Pokemon
            
        # === BATTLE ENDING OUTCOMES ===
        if getattr(battle, 'battle_finished', False):
            if getattr(battle, 'won', False):
                # Victory bonus scaled by performance
                hp_advantage = self._calculate_hp_advantage(battle)
                reward += 5.0 + hp_advantage  # Base win + performance bonus
            else:
                reward -= 3.0  # Clear loss penalty
                
        # === CRITICAL HEALTH THRESHOLDS ===
        # Reward bringing opponent to critical health (strategic positioning)
        if battle.opponent_active_pokemon and prior_battle.opponent_active_pokemon:
            current_hp = getattr(battle.opponent_active_pokemon, 'current_hp_fraction', 1.0)
            prior_hp = getattr(prior_battle.opponent_active_pokemon, 'current_hp_fraction', 1.0)
            
            # Crossing strategic thresholds
            if prior_hp > 0.5 and current_hp <= 0.5:
                reward += 0.8  # Brought to half health
            if prior_hp > 0.25 and current_hp <= 0.25:
                reward += 1.2  # Brought to critical health
                
        return reward
    
    def _calculate_contextual_action_reward(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Reward actions based on context - same action can be good or bad depending on situation.
        This encourages the agent to learn WHEN to use moves, not just WHICH moves to use.
        """
        if not self.action_history or not prior_battle:
            return 0.0
            
        reward = 0.0
        last_action = self.action_history[-1] if self.action_history else None
        
        if not last_action or not battle.active_pokemon:
            return 0.0
            
        action_type = last_action.get('action_type', '')
        
        # === CONTEXT-DEPENDENT MOVE EVALUATION ===
        if action_type.startswith('move'):
            reward += self._evaluate_move_context(battle, prior_battle, last_action)
            
        # === CONTEXT-DEPENDENT SWITCHING ===
        elif action_type == 'switch':
            reward += self._evaluate_switch_context(battle, prior_battle, last_action)
            
        return reward
    
    def _evaluate_move_context(self, battle: AbstractBattle, prior_battle: AbstractBattle, last_action: dict) -> float:
        """Evaluate if the move choice made sense in context"""
        reward = 0.0
        
        if not battle.active_pokemon or not prior_battle.active_pokemon:
            return 0.0
            
        # Get battle context
        my_hp = getattr(battle.active_pokemon, 'current_hp_fraction', 1.0)
        opp_hp = getattr(battle.opponent_active_pokemon, 'current_hp_fraction', 1.0) if battle.opponent_active_pokemon else 1.0
        
        battle_phase = self.calculate_battle_phase(battle)
        
        # Analyze the move that was used
        available_moves = getattr(prior_battle, 'available_moves', [])
        if available_moves:
            action_value = last_action.get('action', 6)
            move_index = action_value - 6
            
            if 0 <= move_index < len(available_moves):
                try:
                    move = available_moves[move_index]
                    move_category = getattr(move, 'category', None)
                    
                    # === CONTEXTUAL MOVE EVALUATION ===
                    
                    # Setup moves should be used early when healthy
                    if self.is_stat_boost_move(move):
                        if battle_phase < 0.3 and my_hp > 0.7:
                            reward += 0.5  # Good time to set up
                        elif battle_phase > 0.7 or my_hp < 0.4:
                            reward -= 0.8  # Poor time to set up
                            
                    # Offensive moves context
                    elif move_category in [MoveCategory.PHYSICAL, MoveCategory.SPECIAL]:
                        # Attacking when opponent is low is good
                        if opp_hp < 0.3:
                            reward += 0.3
                        # Attacking when you're low and opponent is healthy may be desperate
                        elif my_hp < 0.3 and opp_hp > 0.7:
                            if not self._can_ko_opponent(move, battle):
                                reward -= 0.2  # Likely bad trade
                                
                    # Status moves context
                    elif move_category == MoveCategory.STATUS and not self.is_stat_boost_move(move):
                        # Status moves better early-mid game
                        if battle_phase < 0.6:
                            reward += 0.2
                        else:
                            reward -= 0.1
                            
                except (IndexError, AttributeError):
                    pass
                    
        return reward
    
    def _evaluate_switch_context(self, battle: AbstractBattle, prior_battle: AbstractBattle, last_action: dict) -> float:
        """Evaluate if switching made sense in context"""
        reward = 0.0
        
        # Switching when low HP is often good
        if prior_battle.active_pokemon:
            prior_hp = getattr(prior_battle.active_pokemon, 'current_hp_fraction', 1.0)
            if prior_hp < 0.3:
                reward += 0.4  # Good defensive switch
            elif prior_hp > 0.8:
                # Switching when healthy better have a good reason (type advantage, setup, etc.)
                reward -= 0.1  # Slight penalty for switching healthy Pokemon
                
        return reward
    
    def _calculate_strategic_coherence(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Reward coherent strategic sequences rather than random actions.
        This encourages the agent to develop consistent strategies.
        """
        if not self.action_history or len(self.action_history) < 2:
            return 0.0
            
        reward = 0.0
        recent_actions = self.action_history[-3:] if len(self.action_history) >= 3 else self.action_history
        
        # === STRATEGIC COHERENCE PATTERNS ===
        
        # Setup -> Attack sequences
        setup_then_attack = self._detect_setup_attack_sequence(recent_actions)
        if setup_then_attack:
            reward += 0.8  # Strong reward for coherent strategy
            
        # Defensive sequences (switch -> heal/status)
        defensive_sequence = self._detect_defensive_sequence(recent_actions)
        if defensive_sequence:
            reward += 0.5
            
        # Sweep attempts (multiple attacks in a row when advantageous)
        sweep_sequence = self._detect_sweep_sequence(recent_actions, battle)
        if sweep_sequence:
            reward += 0.6
            
        # Penalize incoherent action patterns
        incoherent_penalty = self._calculate_incoherence_penalty(recent_actions)
        reward -= incoherent_penalty
        
        return reward
    
    def _detect_setup_attack_sequence(self, actions: list) -> bool:
        """Detect if agent set up stats then attacked"""
        if len(actions) < 2:
            return False
            
        # Look for stat boost followed by attack
        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]
            
            if (current.get('action_type', '').startswith('move') and 
                next_action.get('action_type', '').startswith('move')):
                
                # Check if first was setup, second was attack
                # This is a simplified check - in full implementation, 
                # we'd analyze the actual moves
                if (current.get('was_stat_boost', False) and 
                    not next_action.get('was_stat_boost', False)):
                    return True
                    
        return False
    
    def _detect_defensive_sequence(self, actions: list) -> bool:
        """Detect defensive strategic sequences"""
        if len(actions) < 2:
            return False
            
        # Switch followed by defensive move/healing
        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]
            
            if (current.get('action_type') == 'switch' and 
                next_action.get('action_type', '').startswith('move')):
                return True  # Switch -> move can be defensive
                
        return False
    
    def _detect_sweep_sequence(self, actions: list, battle: AbstractBattle) -> bool:
        """Detect when agent is attempting to sweep (multiple attacks)"""
        if len(actions) < 2:
            return False
            
        # Count consecutive offensive moves
        consecutive_attacks = 0
        for action in reversed(actions):
            if action.get('action_type', '').startswith('move') and not action.get('was_stat_boost', False):
                consecutive_attacks += 1
            else:
                break
                
        # Sweep attempt if 2+ consecutive attacks and we have momentum
        if consecutive_attacks >= 2:
            # Check if we have advantage (opponent's HP is low or we're boosted)
            if battle.opponent_active_pokemon:
                opp_hp = getattr(battle.opponent_active_pokemon, 'current_hp_fraction', 1.0)
                if opp_hp < 0.5:  # Opponent is weakened
                    return True
                    
        return False
    
    def _calculate_incoherence_penalty(self, actions: list) -> float:
        """Penalize clearly incoherent action patterns"""
        if len(actions) < 3:
            return 0.0
            
        penalty = 0.0
        
        # Excessive switching back and forth
        switches = sum(1 for action in actions if action.get('action_type') == 'switch')
        if switches >= 2 and len(actions) <= 3:
            penalty += 0.3  # Switching too much in short time
            
        # Setting up when at very low HP
        for action in actions:
            if (action.get('action_type', '').startswith('move') and 
                action.get('was_stat_boost', False) and 
                action.get('hp_when_used', 1.0) < 0.2):
                penalty += 0.4  # Setup when almost fainted
                
        return penalty
    
    def _can_ko_opponent(self, move: Move, battle: AbstractBattle) -> bool:
        """Estimate if move can KO opponent (simplified check)"""
        if not battle.opponent_active_pokemon:
            return False
            
        opp_hp = getattr(battle.opponent_active_pokemon, 'current_hp_fraction', 1.0)
        
        # Simplified KO check - in reality this would need damage calculation
        # High power move + low opponent HP = likely KO
        move_power = getattr(move, 'base_power', 0)
        
        if opp_hp <= 0.1:  # Very low HP
            return move_power > 40  # Most moves can KO
        elif opp_hp <= 0.25:  # Low HP
            return move_power > 80  # Strong moves can KO
        elif opp_hp <= 0.5:  # Medium HP
            return move_power > 120  # Very strong moves can KO
            
        return False  # Unlikely to KO at high HP

    def _calculate_hp_advantage(self, battle: AbstractBattle) -> float:
        """Calculate HP advantage between teams"""
        my_total_hp = 0.0
        opp_total_hp = 0.0
        
        # Calculate team HP
        if hasattr(battle, 'team') and battle.team:
            for pokemon in battle.team.values():
                if pokemon and hasattr(pokemon, 'current_hp_fraction'):
                    my_total_hp += getattr(pokemon, 'current_hp_fraction', 0.0)
                    
        if hasattr(battle, 'opponent_team') and battle.opponent_team:
            for pokemon in battle.opponent_team.values():
                if pokemon and hasattr(pokemon, 'current_hp_fraction'):
                    opp_total_hp += getattr(pokemon, 'current_hp_fraction', 0.0)
                    
        # Return advantage (-1 to +1 scale)
        total_hp = my_total_hp + opp_total_hp
        if total_hp > 0:
            return (my_total_hp - opp_total_hp) / total_hp
        return 0.0

    def _calculate_adaptation_reward(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Reward adaptation to opponent's strategy and learning from previous encounters.
        This encourages the agent to adjust its play based on what the opponent does.
        """
        if not self.action_history or len(self.action_history) < 3:
            return 0.0
            
        reward = 0.0
        
        # === ADAPTATION PATTERNS ===
        
        # Reward changing strategy when current approach isn't working
        strategy_change = self._detect_strategy_adaptation(self.action_history)
        if strategy_change:
            reward += 0.6  # Good to adapt when things aren't working
            
        # Reward counter-play to opponent's patterns
        counter_play = self._detect_counter_adaptation(battle, self.action_history)
        if counter_play:
            reward += 0.8  # Strong reward for counter-adaptation
            
        # Reward learning from successful patterns
        pattern_reinforcement = self._detect_successful_pattern_use(battle, self.action_history)
        if pattern_reinforcement:
            reward += 0.4  # Moderate reward for using what works
            
        return reward
    
    def _detect_strategy_adaptation(self, actions: list) -> bool:
        """Detect if agent changed strategy after repeated failures"""
        if len(actions) < 4:
            return False
            
        # Look for pattern changes after unsuccessful sequences
        recent_window = actions[-4:]
        
        # Check if first half used one approach, second half used different approach
        first_half = recent_window[:2]
        second_half = recent_window[2:]
        
        first_strategy = self._categorize_action_strategy(first_half)
        second_strategy = self._categorize_action_strategy(second_half)
        
        # If strategies are different, it might be adaptation
        return first_strategy != second_strategy and first_strategy != 'mixed' and second_strategy != 'mixed'
    
    def _categorize_action_strategy(self, actions: list) -> str:
        """Categorize a sequence of actions into strategic types"""
        if not actions:
            return 'none'
            
        move_count = sum(1 for a in actions if a.get('action_type', '').startswith('move'))
        switch_count = sum(1 for a in actions if a.get('action_type') == 'switch')
        setup_count = sum(1 for a in actions if a.get('was_stat_boost', False))
        
        if setup_count >= len(actions) // 2:
            return 'setup'
        elif switch_count >= len(actions) // 2:
            return 'defensive'
        elif move_count == len(actions):
            return 'aggressive'
        else:
            return 'mixed'
    
    def _detect_counter_adaptation(self, battle: AbstractBattle, actions: list) -> bool:
        """Detect if agent is adapting to opponent's strategy"""
        if len(actions) < 3:
            return False
            
        # This is a simplified version - would need opponent move tracking for full implementation
        # Look for reactive patterns like switching after opponent's strong moves
        
        recent_actions = actions[-3:]
        
        # Check for defensive reactions
        for i in range(len(recent_actions) - 1):
            current = recent_actions[i]
            next_action = recent_actions[i + 1]
            
            # If took damage then switched/used defensive move, might be adaptation
            if (current.get('damage_taken', 0) > 0 and 
                next_action.get('action_type') in ['switch', 'move']):
                return True
                
        return False
    
    def _detect_successful_pattern_use(self, battle: AbstractBattle, actions: list) -> bool:
        """Detect if agent is repeating successful patterns"""
        if len(actions) < 4:
            return False
            
        # Look for similar action patterns that led to good outcomes
        # This is simplified - would need outcome tracking for full implementation
        
        # Check if current sequence matches a previously successful pattern
        current_pattern = [a.get('action_type') for a in actions[-2:]]
        
        # Look through earlier actions for similar patterns
        for i in range(len(actions) - 4):
            if i + 1 < len(actions):
                past_pattern = [actions[i].get('action_type'), actions[i + 1].get('action_type')]
                
                if (past_pattern == current_pattern and 
                    actions[i].get('led_to_advantage', False)):  # Would need to track this
                    return True
                    
        return False

    def _evaluate_stat_boost_timing(self, battle: AbstractBattle, move: Move) -> float:
        """Evaluate whether stat boost was used at appropriate time"""
        if not battle.active_pokemon:
            return 0.0
        
        active_boosts = getattr(battle.active_pokemon, 'boosts', {})
        battle_phase = self.calculate_battle_phase(battle)
        
        # Penalize redundant boosts (already at +6)
        move_id = getattr(move, 'id', '').lower()
        affected_stats = self._get_boosted_stats(move_id)
        
        for stat in affected_stats:
            current_boost = active_boosts.get(stat, 0)
            if current_boost >= 6:  # Already at maximum
                return -1.0  #Reduced from -2.0 to prevent extreme penalties
            elif current_boost >= 4:  # Near maximum
                return -0.3  # Reduced from -0.5
        

        # Reward early game stat boosting (when it's more valuable)
        if battle_phase < 0.3:  # Early game
            return 1.0  # FIXED: Reduced from 2.0
        elif battle_phase < 0.6:  # Mid game
            return 0.5  # FIXED: Reduced from 1.0
        else:  # Late game - boosting less valuable
            return -0.2  # FIXED: Reduced from -0.5
    
    def _get_boosted_stats(self, move_id: str) -> list[str]:
        """Get which stats a move boosts"""
        boost_map = {
            'swordsdance': ['atk'], 'nastyplot': ['spa'], 'calmmind': ['spa', 'spd'],
            'dragondance': ['atk', 'spe'], 'quiverdance': ['spa', 'spd', 'spe'],
            'shellsmash': ['atk', 'spa', 'spe'], 'agility': ['spe'], 'rockpolish': ['spe'],
            'irondefense': ['def'], 'amnesia': ['spd'], 'bulkup': ['atk', 'def']
        }
        return boost_map.get(move_id, [])
    
    def _calculate_efficiency_reward(self, battle: AbstractBattle) -> float:
        """Reward efficient play (shorter battles when winning)"""
        if not getattr(battle, 'won', False):
            return 0.0
        
        turn_number = getattr(battle, 'turn', 0)
        
        # Bonus for winning quickly
        if turn_number <= 10:
            return 1.0  # FIXED: Reduced from 2.0
        elif turn_number <= 20:
            return 0.5  # FIXED: Reduced from 1.0
        elif turn_number <= 30:
            return 0.2  # FIXED: Reduced from 0.5
        else:
            return -0.1  # FIXED: Reduced from -0.2
    
    def _calculate_exploration_reward(self, battle: AbstractBattle) -> float:
        """Small bonus for trying different strategies - only if performing reasonably well"""
        if len(self.action_history) < 3:
            return 0.0
        
        # FIXED: Only provide exploration bonus if not losing badly
        # Check if we're performing poorly (losing too much health)
        battle_team = getattr(battle, 'team', {})
        health_team = [getattr(mon, 'current_hp_fraction', 0.0) for mon in battle_team.values()]
        avg_team_health = sum(health_team) / max(1, len(health_team))
        
        # Reduce exploration rewards if team is in bad shape
        if avg_team_health < 0.3:  # Team is badly hurt
            exploration_multiplier = 0.2  # Much smaller exploration rewards
        elif avg_team_health < 0.6:  # Team is moderately hurt  
            exploration_multiplier = 0.5  # Reduced exploration rewards
        else:
            exploration_multiplier = 1.0  # Full exploration rewards when doing well
        
        # Bonus for action variety
        recent_action_types = [action['action_type'] for action in self.action_history[-3:]]
        unique_types = len(set(recent_action_types))
        
        base_exploration = 0.0
        if unique_types >= 3:
            base_exploration = 0.2  # FIXED: Reduced from 0.3
        elif unique_types == 2:
            base_exploration = 0.1  # Keep same
        else:
            base_exploration = 0.0  # No bonus for repetitive actions
            
        return base_exploration * exploration_multiplier

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        V2 Enhanced state representation with temporal and strategic context:
        
        Base features (V1): 71 features  
        - 12 (health) + 22 (active pokemon details) + 28 (moves) + 2 (fainted counts) + 1 (turn) + 4 (weather) + 1 (switches) + 1 (extra from V1)
        
        V2 additions: 22 features  
        - 9 (action history: 3 actions × 3 features)
        - 1 (battle phase)
        - 2 (momentum indicators) 
        - 5 (stat boost efficiency)
        - 2 (action patterns)
        - 3 (strategic game phase indicators)
        
        Total: 71 + 22 = 93 features

        Returns:
            int: The size of the observation space.
        """
        return 93

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
    
    # V2 Enhancement: Strategic context methods
    def track_action(self, action: int, battle: AbstractBattle, action_type: str | None = None):
        """Track actions for temporal context and strategic analysis"""
        if action_type is None:
            action_type = self.get_action_type(action)
        
        action_context = {
            'action': action,
            'action_type': action_type,
            'turn': getattr(battle, 'turn', 0),
            'active_pokemon': getattr(battle.active_pokemon, 'species', '') if battle.active_pokemon else '',
            'hp_fraction': getattr(battle.active_pokemon, 'current_hp_fraction', 0.0) if battle.active_pokemon else 0.0
        }
        
        # Keep last 3 actions
        self.action_history.append(action_context)
        if len(self.action_history) > 3:
            self.action_history.pop(0)
    
    def get_action_type(self, action: int) -> str:
        """Categorize actions into strategic types"""
        if action == -2:
            return "default"
        elif action == -1:
            return "forfeit"
        elif 0 <= action <= 5:
            return "switch"
        elif 6 <= action <= 9:
            return "move"
        elif 10 <= action <= 13:
            return "move_mega"
        elif 14 <= action <= 17:
            return "move_z"
        elif 18 <= action <= 21:
            return "move_dynamax"
        elif 22 <= action <= 25:
            return "move_tera"
        else:
            return "unknown"
    
    def is_stat_boost_move(self, move: Move) -> bool:
        """Check if a move primarily boosts stats"""
        if not move or getattr(move, 'category', None) != MoveCategory.STATUS:
            return False
        
        move_id = getattr(move, 'id', '').lower()
        stat_boost_moves = {
            'swordsdance', 'nastyplot', 'calmmind', 'dragondance', 'quiverdance',
            'shellsmash', 'geomancy', 'tailglow', 'agility', 'rockpolish',
            'irondefense', 'amnesia', 'barrier', 'acidarmor', 'bulkup'
        }
        return move_id in stat_boost_moves
    
    def calculate_battle_phase(self, battle: AbstractBattle) -> float:
        """Determine battle phase: 0.0 = early, 0.5 = mid, 1.0 = late"""
        if not battle:
            return 0.0
        
        # Calculate based on turn number and remaining Pokemon
        turn_factor = min(getattr(battle, 'turn', 0) / 30.0, 1.0)
        
        battle_team = getattr(battle, 'team', {})
        battle_opponent_team = getattr(battle, 'opponent_team', {})
        fainted_us = sum(1 for mon in battle_team.values() if getattr(mon, 'fainted', False))
        fainted_them = sum(1 for mon in battle_opponent_team.values() if getattr(mon, 'fainted', False))
        
        faint_factor = (fainted_us + fainted_them) / 12.0  # Max 12 total faints
        
        return min((turn_factor + faint_factor) / 2.0, 1.0)
    
    def calculate_momentum(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> Tuple[float, float]:
        """Calculate recent momentum (damage trends)"""
        if not prior_battle:
            return 0.0, 0.0
        
        # Calculate damage dealt to opponent this turn
        health_opponent = [getattr(mon, 'current_hp_fraction', 0.0) for mon in battle.opponent_team.values()]
        prior_health_opponent = [getattr(mon, 'current_hp_fraction', 0.0) for mon in prior_battle.opponent_team.values()]
        
        if len(health_opponent) != len(prior_health_opponent):
            return 0.0, 0.0
        
        damage_dealt = sum(max(0, prior - current) for prior, current in zip(prior_health_opponent, health_opponent))
        
        # Calculate damage taken by us this turn
        health_team = [getattr(mon, 'current_hp_fraction', 0.0) for mon in battle.team.values()]
        prior_health_team = [getattr(mon, 'current_hp_fraction', 0.0) for mon in prior_battle.team.values()]
        
        damage_taken = sum(max(0, prior - current) for prior, current in zip(prior_health_team, health_team)) if len(health_team) == len(prior_health_team) else 0.0
        
        # Update momentum trackers
        self.momentum_tracker['damage_dealt'].append(damage_dealt)
        self.momentum_tracker['damage_taken'].append(damage_taken)
        
        # Keep only last 5 turns for momentum calculation
        if len(self.momentum_tracker['damage_dealt']) > 5:
            self.momentum_tracker['damage_dealt'].pop(0)
        if len(self.momentum_tracker['damage_taken']) > 5:
            self.momentum_tracker['damage_taken'].pop(0)
        
        # Calculate momentum as recent trend
        recent_damage_dealt = sum(self.momentum_tracker['damage_dealt']) / len(self.momentum_tracker['damage_dealt'])
        recent_damage_taken = sum(self.momentum_tracker['damage_taken']) / len(self.momentum_tracker['damage_taken'])
        
        # FIXED: Clamp momentum values to prevent gradient explosion
        # Momentum should be bounded to reasonable ranges
        recent_damage_dealt = np.clip(recent_damage_dealt, 0.0, 1.0)
        recent_damage_taken = np.clip(recent_damage_taken, 0.0, 1.0)
        
        return recent_damage_dealt, recent_damage_taken
    
        #checks for move's priority 
    def move_priority(self, move, battle) -> int:
        if not move:
            return 0
        if hasattr(move, 'entry') and move.entry and 'priority' in move.entry:
            try:
                return int(move.entry['priority'])
            except (KeyError, ValueError, TypeError):
                return 0
        return 0
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
        # FIXED: Add numerical stability to prevent division issues
        base_damage = (((2 * level / 5 + 2) * move_base_power * (attack / max(1, defense))) / 50) + 2

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

        #Add bounds and numerical stability to damage output
        damage_ratio = base_damage / max(1, defender_hp)
        # Clamp damage ratio to reasonable bounds to prevent extreme values
        return np.clip(damage_ratio, 0.0, 2.0)
    
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
            
            # Normalized stats (divided by max values)
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
                    try:
                        if move.type and battle.opponent_active_pokemon.types:
                            opp_types = tuple(t.name.title() for t in battle.opponent_active_pokemon.types if t and hasattr(t, 'name'))
                            if opp_types:
                                effectiveness = self.move_effectiveness(move.type.name.title(), opp_types)
                            else:
                                effectiveness = 1.0
                        else:
                            effectiveness = 1.0
                    except (AttributeError, TypeError):
                        # FIXED: Safe fallback for type effectiveness calculation
                        effectiveness = 1.0
                    
                    # Estimated damage (normalized)
                    damage = self.estimate_move_damage(move, battle.active_pokemon, 
                                                     battle.opponent_active_pokemon, battle)
                    
                    base_power = getattr(move, 'base_power', 0) or 0
                    move_category = getattr(move, 'category', None)
                    
                    # Get priority using centralized safe helper
                    move_priority = self.move_priority(move, battle)
                    
                    move_features.extend([
                        base_power / 150.0,  # Normalized power
                        effectiveness / 4.0,  # Effectiveness (0-4x range)
                        damage,  # Estimated damage ratio
                        move_priority / 5.0 + 0.5,  # Priority normalized to [0,1]
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

        # === V2 TEMPORAL CONTEXT FEATURES ===
        # Action history (last 3 actions)
        action_history_features = [0.0] * 9  # 3 actions × 3 features each
        for i, action_data in enumerate(self.action_history[-3:]):
            if i < 3:  # Safety check
                base_idx = i * 3
                # FIXED: Add bounds checking and safer normalization
                action_value = action_data.get('action', -2)
                action_history_features[base_idx] = np.clip(action_value / 26.0, -1.0, 1.0)  # Normalized action
                
                # Action type encoding
                action_type_map = {'move': 0.2, 'switch': 0.4, 'move_mega': 0.6, 'move_z': 0.8, 'move_dynamax': 1.0}
                action_history_features[base_idx + 1] = action_type_map.get(action_data.get('action_type', 'move'), 0.0)
                
                # Turns ago (recency) - with safer calculation
                current_turn = getattr(battle, 'turn', 0)
                action_turn = action_data.get('turn', current_turn)
                turns_ago = max(0, current_turn - action_turn)
                action_history_features[base_idx + 2] = min(turns_ago / 10.0, 1.0)
        
        state.extend(action_history_features)  # 9 values
        
        # === V2 STRATEGIC CONTEXT FEATURES ===
        # Battle phase
        battle_phase = self.calculate_battle_phase(battle)
        state.append(battle_phase)  # 1 value
        
        # Momentum indicators
        try:
            prior_battle = self._get_prior_battle(battle)
        except AttributeError:
            # Handle first call where prior_battle doesn't exist yet
            prior_battle = None
        damage_dealt_momentum, damage_taken_momentum = self.calculate_momentum(battle, prior_battle)
        state.extend([damage_dealt_momentum, damage_taken_momentum])  # 2 values
        
        # Stat boost efficiency (how beneficial more boosts would be)
        if battle.active_pokemon:
            active_boosts = getattr(battle.active_pokemon, 'boosts', {})
            boost_efficiency = []
            for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
                current_boost = active_boosts.get(stat, 0)
                # Efficiency decreases as we approach +6 cap
                efficiency = max(0.0, (6 - current_boost) / 6.0)
                boost_efficiency.append(efficiency)
            state.extend(boost_efficiency)  # 5 values
        else:
            state.extend([0.0] * 5)
        
        # Recent action pattern detection
        recent_move_count = sum(1 for action in self.action_history[-3:] 
                              if action.get('action_type', '').startswith('move'))
        recent_switch_count = sum(1 for action in self.action_history[-3:] 
                                if action.get('action_type') == 'switch')
        state.extend([recent_move_count / 3.0, recent_switch_count / 3.0])  # 2 values
        
        # Turn-based strategic indicators
        early_game = 1.0 if getattr(battle, 'turn', 0) <= 10 else 0.0
        mid_game = 1.0 if 10 < getattr(battle, 'turn', 0) <= 25 else 0.0
        late_game = 1.0 if getattr(battle, 'turn', 0) > 25 else 0.0
        state.extend([early_game, mid_game, late_game])  # 3 values

        #########################################################################################################
        # V2 Enhanced state calculation:
        # Original V1: 12 + 22 + 28 + 2 + 1 + 4 + 1 = 70 features (but V1 actually produced 71)
        # New V2 additions: 9 (action history) + 1 (battle phase) + 2 (momentum) + 5 (boost efficiency) + 2 (patterns) + 3 (game phase) = 22 features
        # Total: 71 + 22 = 93 features
        #########################################################################################################

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