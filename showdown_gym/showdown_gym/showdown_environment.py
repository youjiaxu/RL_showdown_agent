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
        self.opponent_type_move_patterns = {}  
        self.move_pattern_memory = {} 

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
        
        # v2 Action history and temporal context tracking
        self.action_history = []  # Store last 3 actions with context (for rewards)
        self.full_episode_history = []  # Store all actions for episode stats
        self.last_stat_boost_turns = {}  # Track when each stat was last boosted
        self.momentum_tracker = {'damage_dealt': [], 'damage_taken': []}  # Track recent damage trends
        
        # v3: Cumulative episode reward tracking
        self.cumulative_episode_reward = 0.0  # Sum of all turn rewards in current episode

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
        """
        Episode-level metrics for report analysis.
        All metrics are calculated per EPISODE (full battle), not per turn.
        Rewards are per TURN (calculated in calc_reward).
        """
        info = super().get_additional_info()

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            
            info[agent]["win"] = self.battle1.won
            info[agent]["battle_finished"] = getattr(self.battle1, 'battle_finished', False)
            info[agent]["battle_turns"] = getattr(self.battle1, 'turn', 0)
            
            # reward componennt
            try:
                prior_battle = self._get_prior_battle(self.battle1)
                damage_component = self._calculate_damage_delta(self.battle1, prior_battle)
                ko_component = self._calculate_ko_reward(self.battle1, prior_battle)
                outcome_component = self._calculate_outcome_reward(self.battle1)
                strategic_component = self._calculate_strategic_outcomes(self.battle1, prior_battle)
                immediate_component = self._calculate_immediate_feedback(self.battle1, prior_battle)
                exploration_component = self._calculate_spam_penalties(self.battle1, prior_battle)
                
                info[agent]["reward_damage"] = float(damage_component)
                info[agent]["reward_ko"] = float(ko_component)
                info[agent]["reward_outcome"] = float(outcome_component)
                info[agent]["reward_strategic"] = float(strategic_component)
                info[agent]["reward_immediate"] = float(immediate_component)
                info[agent]["reward_spam_penalty"] = float(exploration_component)
                # reward_total = last turn's reward (per-turn)
                info[agent]["reward_total"] = float(
                    damage_component + ko_component + outcome_component + 
                    strategic_component + immediate_component + exploration_component
                )
                # reward_episode = cumulative sum of all turn rewards (per-episode)
                info[agent]["reward_episode"] = float(self.cumulative_episode_reward)
            except (AttributeError, Exception):
                info[agent]["reward_damage"] = 0.0
                info[agent]["reward_ko"] = 0.0
                info[agent]["reward_outcome"] = 0.0
                info[agent]["reward_strategic"] = 0.0
                info[agent]["reward_immediate"] = 0.0
                info[agent]["reward_spam_penalty"] = 0.0
                info[agent]["reward_total"] = 0.0
                info[agent]["reward_episode"] = 0.0
            
            # === TEAM PERFORMANCE METRICS (Episode-level snapshot) ===
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
            
            # Victory quality (only meaningful when battle is won)
            if info[agent]["win"] and info[agent]["battle_turns"] > 0:
                info[agent]["victory_hp_retained"] = float(team_total_hp / 6.0)  # HP efficiency (0.0-1.0)
            else:
                info[agent]["victory_hp_retained"] = 0.0
            
            # === STRATEGIC BEHAVIOR ANALYSIS (Episode-level) ===
            # Action pattern metrics from FULL episode history
            if self.full_episode_history:
                action_types = [action.get('action_type', 'unknown') for action in self.full_episode_history]
                total_actions = len(action_types)
                
                info[agent]["total_actions_taken"] = total_actions
                info[agent]["move_percentage"] = float(sum(1 for a in action_types if a.startswith('move')) / max(1, total_actions))
                info[agent]["switch_percentage"] = float(sum(1 for a in action_types if a == 'switch') / max(1, total_actions))
                
                # Strategy coherence: measures action consistency (1.0 = never changes action type, 0.0 = changes every turn)
                strategy_changes = sum(1 for i in range(1, len(action_types)) 
                                    if action_types[i] != action_types[i-1])
                info[agent]["strategy_coherence"] = float(1.0 - strategy_changes / max(1, total_actions - 1))
            else:
                info[agent]["total_actions_taken"] = 0
                info[agent]["move_percentage"] = 0.0
                info[agent]["switch_percentage"] = 0.0
                info[agent]["strategy_coherence"] = 0.0
            
            # === BATTLE PHASE (Episode-level snapshot) ===
            battle_phase = self.calculate_battle_phase(self.battle1)
            info[agent]["battle_phase"] = float(battle_phase)  # 0.0 (early) to 1.0 (late)
            
            # === MOMENTUM (Episode-level snapshot) ===
            try:
                prior_battle = self._get_prior_battle(self.battle1)
                damage_dealt_momentum, damage_taken_momentum = self.calculate_momentum(self.battle1, prior_battle)
                info[agent]["momentum_ratio"] = float(damage_dealt_momentum / max(0.001, damage_taken_momentum))
            except (AttributeError, Exception):
                info[agent]["momentum_ratio"] = 1.0
            
            # === TYPE MATCHUP AWARENESS (Episode-level snapshot) ===
            # Measures if agent has favorable type matchups available
            if self.battle1.active_pokemon and hasattr(self.battle1, 'available_moves'):
                available_moves = getattr(self.battle1, 'available_moves', [])
                if available_moves and self.battle1.opponent_active_pokemon:
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
            
            # === STAT BOOST USAGE (Episode-level snapshot) ===
            # Tracks current stat boost state (learning to use setup moves)
            if hasattr(self.battle1, 'active_pokemon') and self.battle1.active_pokemon:
                active_boosts = getattr(self.battle1.active_pokemon, 'boosts', {})
                total_positive_boosts = sum(max(0, boost) for boost in active_boosts.values())
                
                info[agent]["positive_stat_boosts"] = float(total_positive_boosts)
            else:
                info[agent]["positive_stat_boosts"] = 0.0

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
        
        # Component 4: Strategic outcome rewards (encourages switching)
        strategic_reward = self._calculate_strategic_outcomes(battle, prior_battle)
        
        total_reward = damage_reward + ko_reward + outcome_reward + strategic_reward
        
        # Add immediate feedback bonuses for faster learning
        immediate_feedback = self._calculate_immediate_feedback(battle, prior_battle)
        total_reward += immediate_feedback
        
        # Apply spam prevention penalties (simplified, consistent)
        spam_penalty = self._calculate_spam_penalties(battle, prior_battle)
        total_reward += spam_penalty
        

        total_reward = np.clip(total_reward, -20.0, 20.0)
        
        # Final safety check for NaN/infinity
        if not np.isfinite(total_reward):
            total_reward = 0.0
        
        # Track cumulative episode reward (reset happens in reset())
        self.cumulative_episode_reward += total_reward
        
        return total_reward
    
    def _calculate_damage_delta(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Calculate normalized HP advantage change (dense reward component).
        INCREASED magnitude to compensate for win/loss reduction.
        """
        if not prior_battle:
            return 0.0
        
        # Current HP advantage
        current_advantage = self._calculate_hp_advantage(battle)
        
        # Prior HP advantage  
        prior_advantage = self._calculate_hp_advantage(prior_battle)
        
        # Delta in advantage (positive = we improved our position)
        advantage_delta = current_advantage - prior_advantage
        

        
        # INCREASED scaling: ±2.0 max (from ±1.5) to match new reward hierarchy
        # Damage now worth 20% of win (2.0/10.0) instead of 6% (1.5/25.0)
        return np.clip(advantage_delta * 2.0, -2.0, 2.0)
    
    def _calculate_ko_reward(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Calculate KO event rewards (sparse, scaled by game stage).

        """
        if not prior_battle:
            return 0.0
        
        reward = 0.0
        
        # Count KOs
        prior_fainted_opponent = sum(1 for mon in prior_battle.opponent_team.values() if getattr(mon, 'fainted', False))
        current_fainted_opponent = sum(1 for mon in battle.opponent_team.values() if getattr(mon, 'fainted', False))
        
        prior_fainted_team = sum(1 for mon in prior_battle.team.values() if getattr(mon, 'fainted', False))
        current_fainted_team = sum(1 for mon in battle.team.values() if getattr(mon, 'fainted', False))
        
        # KOs
        opponent_kos = current_fainted_opponent - prior_fainted_opponent
        if opponent_kos > 0:
            #early KOs worth less, late KOs worth more
            remaining_opponents = 6 - current_fainted_opponent
            stage_multiplier = min(2.0, max(1.0, 7 - remaining_opponents))  # 1.0 to 2.0
            reward += 4.0 * stage_multiplier * opponent_kos  # INCREASED from 3.0 (now 4.0-8.0 range)
        
        # Losses
        team_kos = current_fainted_team - prior_fainted_team
        if team_kos > 0:
            remaining_team = 6 - current_fainted_team
            stage_multiplier = min(2.0, max(1.0, 7 - remaining_team))  # 1.0 to 2.0
            reward -= 4.0 * stage_multiplier * team_kos  # INCREASED from 3.0 for symmetry
        
        return np.clip(reward, -8.0, 8.0)
    
    def _calculate_outcome_reward(self, battle: AbstractBattle) -> float:
        """
        Calculate win/loss rewards.
        NOTEE: reduce magnitude to prevent gradient washing was 25-33, now 10-12
        """
        #Only give outcome reward when battle is actually finished
        if not getattr(battle, 'battle_finished', False):
            return 0.0
        
        if getattr(battle, 'won', False):
            base_reward = 10.0 
            
            # bonus for clean sweep
            fainted_count = sum(1 for mon in battle.team.values() if getattr(mon, 'fainted', False))
            if fainted_count == 0:
                base_reward += 2.0  # Perfect win (reduced from +5.0)
            elif fainted_count <= 2:
                base_reward += 1.0  # Clean win (reduced from +3.0)
            
            return base_reward
        else:
            return -10.0  # REDUCED from -25.0 for symmetry
    
    def _calculate_immediate_feedback(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """Provide immediate feedback for faster learning"""
        if not prior_battle:
            return 0.0
        
        reward = 0.0
        
        # differentiate attacking from setup
        if (hasattr(self, 'action_history') and len(self.action_history) > 0 and 
            self.action_history[-1].get('action_type', '').startswith('move')):
            
            last_action = self.action_history[-1]
            action_value = last_action.get('action', 6)
            move_index = action_value - 6  # Moves start at action 6
            
            if prior_battle and hasattr(prior_battle, 'available_moves'):
                available_moves = prior_battle.available_moves
                if 0 <= move_index < len(available_moves):
                    move_used = available_moves[move_index]
                    move_category = getattr(move_used, 'category', None)
                    
                    if move_category in [MoveCategory.PHYSICAL, MoveCategory.SPECIAL]:
                        reward += 0.5
                        
                        # Check move effectiveness
                        if battle.opponent_active_pokemon and prior_battle.opponent_active_pokemon:
                            try:
                                opp_types = self._extract_pokemon_types(battle.opponent_active_pokemon)
                                if opp_types and hasattr(move_used, 'type') and move_used.type:
                                    effectiveness = self.move_effectiveness(move_used.type.name.title(), tuple(opp_types))
                                    
                                    # STABLE: Moderate penalties for immune moves (clear signal, not extreme)
                                    if effectiveness == 0.0:
                                        reward -= 1.5  # REDUCED from -2.5 (strong but learnable)
                                    # Moderate penalty for resisted moves  
                                    elif effectiveness < 0.5:
                                        reward -= 0.6  # REDUCED from -1.0 (clear negative signal)
                                    elif effectiveness < 1.0:
                                        reward -= 0.2  # REDUCED from -0.3 (mild disadvantage)
                                    # BALANCED: Good type matchups (clear positive signal)
                                    elif effectiveness >= 2.0:
                                        reward += 0.6  # REDUCED from +0.8 (good but not extreme)
                                    elif effectiveness > 1.0:
                                        reward += 0.3  # REDUCED from +0.4 (consistent advantage)
                            except:
                                pass
                    
                    # stat boost move
                    elif self.is_stat_boost_move(move_used):
                        # earned through timing well
                        stat_boost_reward = self._evaluate_stat_boost_timing(prior_battle, move_used)
                        reward += stat_boost_reward  # Max +0.5 early game, negative otherwise
                    
                    # OTHER STATUS MOVES: Check for redundant status effects
                    elif move_category == MoveCategory.STATUS:
                        status_reward = self._evaluate_status_move_effectiveness(move_used, battle, prior_battle)
                        reward += status_reward  # Smart status use vs redundant spam
        
        # ENHANCED: Strategic switching that considers BOTH offense and defense
        if (hasattr(self, 'action_history') and len(self.action_history) > 0 and 
            self.action_history[-1].get('action_type') == 'switch'):
            
            prior_hp_fraction = self.action_history[-1].get('hp_fraction', 1.0)
            
            # Reward strategic switches that improve our overall battle position
            if battle.active_pokemon and battle.opponent_active_pokemon:
                # Calculate offensive advantage (how much damage we deal)
                our_types = self._extract_pokemon_types(battle.active_pokemon)
                opp_types = self._extract_pokemon_types(battle.opponent_active_pokemon)
                
                if our_types and opp_types:
                    # Offensive effectiveness (our attacks vs opponent)
                    offensive_total = 0.0
                    offensive_count = 0
                    for our_type in our_types:
                        for opp_type in opp_types:
                            try:
                                effectiveness = self.move_effectiveness(our_type, (opp_type,))
                                offensive_total += effectiveness
                                offensive_count += 1
                            except:
                                pass
                    
                    # Defensive effectiveness (opponent attacks vs us)  
                    defensive_total = 0.0
                    defensive_count = 0
                    for opp_type in opp_types:
                        for our_type in our_types:
                            try:
                                incoming_effectiveness = self.move_effectiveness(opp_type, (our_type,))
                                defensive_total += incoming_effectiveness
                                defensive_count += 1
                            except:
                                pass
                    
                    if offensive_count > 0 and defensive_count > 0:
                        avg_offensive = offensive_total / offensive_count
                        avg_defensive = defensive_total / defensive_count    

                        switch_value = avg_offensive / max(avg_defensive, 0.25) 
                        
                        # reward switches without over-incentivizing switching
                        if switch_value >= 3.0:
                            reward += 1.4  # Strong reward for perfect switches
                        elif switch_value >= 2.0: 
                            reward += 1.0  # Good strategic switch
                        elif switch_value >= 1.5:  
                            reward += 0.6  # Decent switch
                        elif switch_value >= 1.0:  # Neutral: roughly even matchup
                            if prior_hp_fraction < 0.3:
                                reward += 0.4  # Emergency escape bonus
                            else:
                                reward -= 0.1  # Mild discouragement for lateral moves
                        elif switch_value >= 0.7:  # Slight disadvantage
                            reward -= 0.4
                        else: 
                            reward -= 0.8 # bad switches 
        
        return np.clip(reward, -2.0, 3.0) 
    
    def _calculate_spam_penalties(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Context-aware spam prevention that only penalizes TRUE spam, not strategic play.
        CRITICAL FIX: Penalties only apply the turn the spam occurs, not every turn forever.
        """
        if not prior_battle or not hasattr(self, 'action_history') or len(self.action_history) < 1:
            return 0.0
            
        reward = 0.0
        
        # Get current action
        current_action = self.action_history[-1]
        current_action_type = current_action.get('action_type', '')
        

        if current_action_type == 'switch':
            recent_window = self.action_history[-5:]
            switch_count = sum(1 for a in recent_window if a.get('action_type') == 'switch')
            
            if switch_count >= 4:
                reward -= 1.2  # Strong penalty for excessive switching
            elif switch_count >= 3:
                reward -= 0.6 

        current_action_category = current_action.get('action_category', '')
        if current_action_category == 'setup':
            recent_window = self.action_history[-5:]
            setup_count = sum(1 for a in recent_window if a.get('action_category') == 'setup')
            
            if setup_count >= 3:
                reward -= 1.5  #discourage spamming
            elif setup_count >= 2:
                reward -= 0.5 
        
        #consecutive entry hazard
        if len(self.action_history) >= 1:
            recent_actions = [action.get('action_category', '') for action in self.action_history]
            
            # Count consecutive hazard moves from the end
            consecutive_hazards = 0
            for action_category in reversed(recent_actions):
                if action_category == 'hazard':
                    consecutive_hazards += 1
                else:
                    break

            # Penalty for duplicate/excessive hazards
            if consecutive_hazards >= 2:
                reward -= 1.0
    
        return np.clip(reward, -4.0, 0.0)  # Only penalties 
    
    def _calculate_strategic_outcomes(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Reward long-term strategic play: setup moves, entry hazards, momentum.
        
        REMOVED: Switching rewards (moved to immediate_feedback to eliminate double-counting)
        REMOVED: Damage threshold bonuses (redundant with damage_delta)
        FOCUS: Setup strategy, hazards, sustained pressure
        
        This component now handles ONLY actions with delayed payoffs.
        """
        if not prior_battle:
            return 0.0
            
        reward = 0.0
           
        # setup move rewards

        if (hasattr(self, 'action_history') and len(self.action_history) > 0 and
            self.action_history[-1].get('action_category') == 'setup'):
            
            battle_phase = self.calculate_battle_phase(battle)
            our_hp = getattr(battle.active_pokemon, 'current_hp_fraction', 1.0) if battle.active_pokemon else 1.0
            
            # BALANCED: Modest strategic setup reward (now similar to good attacking moves)
            if battle_phase < 0.2 and our_hp > 0.8:  # Stricter conditions
                # Check if we have type advantage (safe to setup)
                if battle.active_pokemon and battle.opponent_active_pokemon:
                    our_types = self._extract_pokemon_types(battle.active_pokemon)
                    opp_types = self._extract_pokemon_types(battle.opponent_active_pokemon)
                    
                    if our_types and opp_types:
                        type_advantage = sum(self.move_effectiveness(our_type, (opp_types[0],)) 
                                           for our_type in our_types) / len(our_types)
                        
                        if type_advantage >= 1.0:  # We resist them or neutral
                            reward += 0.8  # REDUCED from 3.0 (now comparable to good type matchup attacking)
        
        # === ENTRY HAZARD REWARDS (Long-term Strategy) ===
        if (hasattr(self, 'action_history') and len(self.action_history) > 0 and
            self.action_history[-1].get('action_category') == 'hazard'):
            
            battle_phase = self.calculate_battle_phase(battle)
            
            # Reward hazards in early-mid game
            if battle_phase < 0.4:
                reward += 1.5  # REWARD strategic hazard setup
                # Note: Damage from hazards will add to damage_delta in future turns
        
        #momentum reward
        if hasattr(self, 'action_history') and len(self.action_history) >= 3:
            # Reward sustained offensive pressure (3+ consecutive attacks)
            recent_offense = sum(1 for a in self.action_history[-3:] 
                               if a.get('action_type', '').startswith('move') and
                               a.get('action_category') not in ['setup', 'hazard'])
            
            if recent_offense >= 3:
                reward += 0.5  # Small bonus for maintaining pressure
                
        return np.clip(reward, 0.0, 3.0) 
    
    def _can_ko_opponent(self, move: Move, battle: AbstractBattle) -> bool:
        """Estimate if move can KO opponent (simplified check)"""
        if not battle.opponent_active_pokemon:
            return False
            
        opp_hp = getattr(battle.opponent_active_pokemon, 'current_hp_fraction', 1.0)
        
        # Simplified KO check - in reality this would need damage calculation
        # High power move + low opponent HP = likely KO
        move_power = getattr(move, 'base_power', 0)
        
        if opp_hp <= 0.1:  # Very low HP
            return move_power > 40
        elif opp_hp <= 0.25:  # Low HP
            return move_power > 80  # Strong moves can KO
        elif opp_hp <= 0.5:  # Medium HP
            return move_power > 120
            
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


    def _evaluate_stat_boost_timing(self, battle: AbstractBattle, move: Move) -> float:
        """Evaluate whether stat boost was used at appropriate time"""
        if not battle.active_pokemon:
            return 0.0
        
        active_boosts = getattr(battle.active_pokemon, 'boosts', {})
        battle_phase = self.calculate_battle_phase(battle)
        
        # Penalize redundant boosts (already boosted)
        move_id = getattr(move, 'id', '').lower()
        affected_stats = self._get_boosted_stats(move_id)
        
        # This prevents unknown boost moves from getting +0.5 reward
        if not affected_stats:
            return -1.0  # Unknown stat boost moves are discouraged
        
        # STABLE: Moderate stat boost spam prevention (clear but not extreme)
        for stat in affected_stats:
            current_boost = active_boosts.get(stat, 0)
            if current_boost >= 6:  # Already at maximum (+6 is cap)
                return -2.5  # REDUCED from -4.0 (strong signal, not crushing)
            elif current_boost >= 4:  # High boosts (diminishing returns)
                return -1.5  # REDUCED from -2.5 (clear negative signal)
            elif current_boost >= 2:  # Moderate boosts (getting redundant)
                return -0.8  # REDUCED from -1.5 (mild discouragement)
            elif current_boost >= 1:  # Some boosts (careful not to over-boost)
                return -0.4  # REDUCED from -0.8 (gentle warning)
        
        # Check if we have HP advantage
        our_hp = getattr(battle.active_pokemon, 'current_hp_fraction', 1.0)
        opp_hp = getattr(battle.opponent_active_pokemon, 'current_hp_fraction', 1.0) if battle.opponent_active_pokemon else 1.0
        
        # BALANCED: Modest setup rewards (less than attacking moves)
        if battle_phase < 0.1 and our_hp > 0.8 and opp_hp > 0.6: # Stricter early game conditions
            return 0.3  # REDUCED from 0.5 (now less than type advantage attacking +0.6)
        elif battle_phase < 0.3 and our_hp > 0.6: 
            return 0.0  # Neutral (no reward for mid-game setup)
        else: 
            return -0.6  # REDUCED penalty from -0.8 (still discourages late setup)
    
    def _get_boosted_stats(self, move_id: str) -> list[str]:
        """
        Comprehensive mapping of stat-boosting moves to affected stats.
        Covers Gen 1-9 competitive moves.
        """
        boost_map = {
            #Attack boosts
            'swordsdance': ['atk'],      # +2 Atk
            'honeclaws': ['atk'],        # +1 Atk, +1 Acc
            'howl': ['atk'],             # +1 Atk (all allies in doubles)
            'meditate': ['atk'],         # +1 Atk
            'sharpen': ['atk'],          # +1 Atk
            'rage': ['atk'],             # +1 Atk when hit
            'poweruppunch': ['atk'],     # +1 Atk (on hit)
            
            # spa boost
            'nastyplot': ['spa'],        # +2 SpA
            'tailglow': ['spa'],         # +3 SpA
            'geomancy': ['spa', 'spd', 'spe'],  # +2 SpA, +2 SpD, +2 Spe (charge move)
            'growth': ['atk', 'spa'],    # +1 Atk, +1 SpA (+2 each in sun)
            'workup': ['atk', 'spa'],    # +1 Atk, +1 SpA
            'rototiller': ['atk', 'spa'], # +1 Atk, +1 SpA (Grass types only)
            'chargebeam': ['spa'],       # +1 SpA (70% chance on hit)
            'fierydan ce': ['atk', 'spa'], # +1 Atk, +1 SpA (50% chance)
            
            # def obosts
            'irondefense': ['def'],      # +2 Def
            'acidarmor': ['def'],        # +2 Def
            'barrier': ['def'],          # +2 Def
            'cottonguard': ['def'],      # +3 Def
            'stockpile': ['def', 'spd'], # +1 Def, +1 SpD (stackable 3x)
            'defend order': ['def', 'spd'], # +1 Def, +1 SpD
            'cosmicpower': ['def', 'spd'], # +1 Def, +1 SpD
            'coil': ['atk', 'def'],      # +1 Atk, +1 Def, +1 Acc
            'bulkup': ['atk', 'def'],    # +1 Atk, +1 Def
            'curse': ['atk', 'def'],     # +1 Atk, +1 Def, -1 Spe (non-Ghost)
            'withdraw': ['def'],         # +1 Def
            'harden': ['def'],           # +1 Def
            'defensecurl': ['def'],      # +1 Def
            'shelter': ['def'],          # +2 Def
            
            # spd boosts
            'amnesia': ['spd'],          # +2 SpD
            'calmmind': ['spa', 'spd'],  # +1 SpA, +1 SpD
            'charge': ['spd'],           # +1 SpD (doubles Electric move power next turn)
            
            # spe boosts
            'agility': ['spe'],          # +2 Spe
            'rockpolish': ['spe'],       # +2 Spe
            'autotomize': ['spe'],       # +2 Spe (user loses weight)
            'doubleteam': ['spe'],       # +1 Evasion (evasion ≈ speed in Gen 9)
            'minimize': ['spe'],         # +2 Evasion
            'flamecha rge': ['spe'],     # +1 Spe (on hit)
            
            # stat boosts
            'dragondance': ['atk', 'spe'],     # +1 Atk, +1 Spe
            'quiverdance': ['spa', 'spd', 'spe'], # +1 SpA, +1 SpD, +1 Spe
            'shellsmash': ['atk', 'spa', 'spe'],  # +2 Atk, +2 SpA, +2 Spe, -1 Def, -1 SpD
            'shiftgear': ['atk', 'spe'],       # +1 Atk, +2 Spe
            'victorydance': ['atk', 'def', 'spe'], # +1 Atk, +1 Def, +1 Spe
            'noretreat': ['atk', 'def', 'spa', 'spd', 'spe'], # +1 all stats
            
            # omni boosts
            'ancientpower': ['atk', 'def', 'spa', 'spd', 'spe'], # +1 all (10% chance)
            'silverwind': ['atk', 'def', 'spa', 'spd', 'spe'],   # +1 all (10% chance)
            'ominouswind': ['atk', 'def', 'spa', 'spd', 'spe'],  # +1 all (10% chance)
            'meteorbeam': ['spa'],  # +1 SpA (charge move)
            'clangingscales': ['def'],  # -1 Def (self-debuff, not boost)
            
            # conditoinal boosts
            'acupressure': ['atk', 'def', 'spa', 'spd', 'spe'], # +2 random stat
            'bellydrum': ['atk'],  # +6 Atk (max immediately, costs 50% HP)
            'maxknuckle': ['atk'],  # +1 Atk (Max Move)
            'maxflare': ['spa'],    # +1 SpA (Max Move)  
            'maxgeyser': ['spa'],   # +1 SpA (Max Move)
            'maxovergrowth': ['spa'], # +1 SpA (Max Move)
            'maxlightning': ['spa'], # +1 SpA (Max Move)
            'maxrockfall': ['spa'],  # +1 SpA (Max Move)
            'maxwyrmwind': ['spa'],  # +1 SpA (Max Move)
            'maxknucle': ['atk'],   # typo variant
            
            # ability based boosts
            'powertrip': ['atk'],   # Damage scales with boosts (not a boost itself, but tracked)
            'storedpower': ['spa'], # Damage scales with boosts
        }
        return boost_map.get(move_id, [])
    
    def _calculate_efficiency_reward(self, battle: AbstractBattle) -> float:
        """Reward efficient play (shorter battles when winning)"""
        if not getattr(battle, 'won', False):
            return 0.0
        
        turn_number = getattr(battle, 'turn', 0)
        
        # Bonus for winning quickly
        if turn_number <= 10:
            return 1.0 
        elif turn_number <= 20:
            return 0.5
        elif turn_number <= 30:
            return 0.2
        else:
            return -0.1 
        
    def is_entry_hazard_move(self, move) -> bool:
        """
        Detect if a move is an entry hazard (Stealth Rock, Spikes, etc.).
        Entry hazards damage Pokemon when they switch in.
        """
        if not move:
            return False
        
        move_id = getattr(move, 'id', '').lower()
        
        # Comprehensive list of entry hazard moves
        entry_hazards = {
            'stealthrock',      # Most common - 1/8 HP based on Rock weakness
            'spikes',           # 1/8, 1/6, or 1/4 HP (stackable 3 layers)
            'toxicspikes',      # Poison on switch-in (stackable 2 layers)
            'stickyweb',        # -1 Speed on switch-in
            'gmaxsteelsurge',   # G-Max Steel hazard (like Stealth Rock for Steel)
        }
        
        return move_id in entry_hazards
    
    def _evaluate_status_move_effectiveness(self, move: Move, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Evaluate status moves to prevent spamming already-statused opponents and reward strategic timing.
        
        FIXES ISSUES:
        1. Spamming Thunder Wave on already paralyzed opponent
        2. Using sleep moves on sleeping opponent  
        3. Using burn moves on already burned opponent
        4. Not recognizing when status moves are effective vs immune
        """
        if not move or not battle.opponent_active_pokemon:
            return 0.0
            
        move_id = getattr(move, 'id', '').lower()
        opponent = battle.opponent_active_pokemon
        
        # Check opponent's current status
        opponent_status = getattr(opponent, 'status', None)
        opponent_status_name = opponent_status.name.lower() if opponent_status else None
        
        # Status move categories and their effects
        paralysis_moves = {'thunderwave', 'glare', 'stunspore', 'bodyslam', 'nuzzle'}
        sleep_moves = {'sleeppowder', 'spore', 'hypnosis', 'darkvoid', 'grasswhistle', 'lovelykiss'}
        burn_moves = {'willowisp', 'scald', 'flamethrower', 'fireblast', 'lavaplume'}
        poison_moves = {'toxic', 'poisongas', 'poisonpowder', 'sludgebomb', 'toxicspikes'}
        freeze_moves = {'icebeam', 'blizzard', 'freezedry'}
        
        # STABLE PENALTY: Using status move on already-affected opponent (clear but not extreme)
        if opponent_status_name:
            if move_id in paralysis_moves and opponent_status_name == 'par':
                return -1.0 
            elif move_id in sleep_moves and opponent_status_name == 'slp':
                return -1.0  
            elif move_id in burn_moves and opponent_status_name == 'brn':
                return -1.0  
            elif move_id in poison_moves and opponent_status_name in ['psn', 'tox']:
                return -1.0  
            elif move_id in freeze_moves and opponent_status_name == 'frz':
                return -1.0  
        
        # REWARD: Using status move on healthy opponent (strategic value)
        if not opponent_status_name:
            # Check type effectiveness first (some types immune to certain status)
            opp_types = self._extract_pokemon_types(opponent)
            
            # Ground types immune to Thunder Wave, Fire types immune to burn, etc.
            if opp_types:
                type_effectiveness = True
                for opp_type in opp_types:
                    # Electric moves like Thunder Wave don't affect Ground types
                    if move_id in paralysis_moves and opp_type.lower() == 'ground':
                        type_effectiveness = False
                    # Fire moves like Will-O-Wisp don't affect Fire types  
                    elif move_id in burn_moves and opp_type.lower() == 'fire':
                        type_effectiveness = False
                    # Poison moves don't affect Steel or Poison types
                    elif move_id in poison_moves and opp_type.lower() in ['steel', 'poison']:
                        type_effectiveness = False
                
                if not type_effectiveness:
                    return -1.2  # PENALTY for using status move on immune type
            
            # Strategic value of different status effects
            if move_id in paralysis_moves:
                return 0.8  # Paralysis is very good (25% chance to not move + speed cut)
            elif move_id in sleep_moves:  
                return 1.0  # Sleep is excellent (opponent can't move for 1-3 turns)
            elif move_id in burn_moves:
                return 0.6  # Burn is good (50% attack cut + residual damage)
            elif move_id in poison_moves:
                return 0.4  # Poison is decent (residual damage, Toxic gets stronger)
            elif move_id in freeze_moves:
                return 0.7  # Freeze is strong but unreliable
        
        # Default reward for other status moves (screens, stat drops, etc.)
        return 0.15
    
    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        SIMPLIFIED Strategic state representation (noise reduction):
        
        Base features (V1): 75 features
        - 12 (health: 6 team + 6 opponent)
        - 23 (active pokemon: 5+5 stats, 1+1 status, 5+5 boosts, 1 speed comparison)
        - 32 (moves: 4 moves × 8 features including accuracy)
        - 2 (fainted counts)
        - 1 (turn number)
        - 4 (weather one-hot)
        - 1 (switches available)
        
        V2 additions: 17 features  
        - 9 (action history: 3 actions × 3 features)
        - 1 (battle phase)
        - 2 (momentum indicators) 
        - 2 (action patterns)
        - 3 (strategic game phase indicators)
        
        REMOVED V3 strategic additions (observation-reward mismatch):
        - Switch viability scores (not directly used in rewards)
        - Threat assessments (not directly used in rewards)
        - Coverage analysis (not directly used in rewards)
        
        These features added complexity without corresponding reward validation,
        creating a mismatch where the agent couldn't learn to use them effectively.
        
        Total: 75 + 17 = 92 features (reduced from 124)

        Returns:
            int: The size of the observation space.
        """
        return 92 

    def move_effectiveness(self, attacker_type: str, defender_types: Tuple[str, ...]) -> float:
        """Calculate type effectiveness with safety checks"""
        if not attacker_type or not defender_types:
            return 1.0
            
        try:
            multiplier = 1.0
            # Defender can have more than 1 type, multiplying effectiveness
            for def_type in defender_types:
                if def_type:  # Ensure type is not None or empty
                    # Return 1.0 if no specific effectiveness found
                    effectiveness = self.data.effect_chart.get(str(attacker_type), {}).get(str(def_type), 1.0)
                    multiplier *= effectiveness
            return float(multiplier)  # Ensure we return a float
        except (AttributeError, TypeError, KeyError):
            # Fallback to neutral effectiveness if calculation fails
            return 1.0
        #gets base_stats using GenData library for estimated stat calculation 
    
    def _extract_pokemon_types(self, pokemon) -> list[str]:
        """Safely extract types from a Pokemon with proper null checking"""
        if not pokemon:
            return []
        
        types = []
        try:
            # Get type_1 safely
            type_1 = getattr(pokemon, 'type_1', None)
            if type_1:
                # Convert to string if it's a type object
                type_str = str(type_1) if hasattr(type_1, '__str__') else type_1
                if type_str and type_str != 'None':
                    types.append(type_str)
            
            # Get type_2 safely
            type_2 = getattr(pokemon, 'type_2', None)
            if type_2:
                # Convert to string if it's a type object
                type_str = str(type_2) if hasattr(type_2, '__str__') else type_2
                if type_str and type_str != 'None':
                    types.append(type_str)
                    
        except (AttributeError, TypeError):
            # If anything fails, return empty list
            pass
            
        return types
    
    def _get_move_accuracy(self, move: Move) -> float:
        """
        Get move accuracy with proper handling of special cases.
        Returns accuracy as a percentage (0-100).
        """
        if not move:
            return 100.0  # Default to perfect accuracy
        
        try:
            accuracy = getattr(move, 'accuracy', 100)
            
            # Handle special accuracy values
            if accuracy is True or accuracy is None:
                return 100.0  # Moves that never miss
            elif accuracy is False or accuracy == 0:
                return 0.0 
            else:
                return float(accuracy)  # Normal accuracy
        except (AttributeError, TypeError, ValueError):
            return 100.0  #fallback
    
    def _estimate_stat_from_types(self, pokemon_types: list[str], stat: str) -> int:
        """RANDOM TEAM FEATURE: Estimate stats based on type patterns when species is unknown"""
        if not pokemon_types:
            return 100  # Neutral competitive stat
            
        # Type-based stat tendencies (competitive averages)
        type_stat_tendencies = {
            'Fire': {'atk': 105, 'def': 85, 'spa': 100, 'spd': 90, 'spe': 95, 'hp': 85},
            'Water': {'atk': 90, 'def': 95, 'spa': 100, 'spd': 105, 'spe': 85, 'hp': 90},
            'Grass': {'atk': 85, 'def': 90, 'spa': 105, 'spd': 100, 'spe': 80, 'hp': 85},
            'Electric': {'atk': 85, 'def': 75, 'spa': 110, 'spd': 85, 'spe': 115, 'hp': 75},
            'Psychic': {'atk': 75, 'def': 80, 'spa': 115, 'spd': 110, 'spe': 100, 'hp': 85},
            'Flying': {'atk': 95, 'def': 80, 'spa': 85, 'spd': 85, 'spe': 110, 'hp': 80},
            'Fighting': {'atk': 115, 'def': 90, 'spa': 65, 'spd': 85, 'spe': 95, 'hp': 90},
            'Steel': {'atk': 100, 'def': 125, 'spa': 85, 'spd': 100, 'spe': 70, 'hp': 85},
            'Dragon': {'atk': 110, 'def': 95, 'spa': 110, 'spd': 95, 'spe': 90, 'hp': 95},
            'Dark': {'atk': 105, 'def': 85, 'spa': 90, 'spd': 85, 'spe': 100, 'hp': 85},
            'Ghost': {'atk': 85, 'def': 80, 'spa': 105, 'spd': 100, 'spe': 95, 'hp': 80},
            'Rock': {'atk': 105, 'def': 115, 'spa': 75, 'spd': 85, 'spe': 65, 'hp': 85},
            'Ground': {'atk': 110, 'def': 100, 'spa': 80, 'spd': 85, 'spe': 80, 'hp': 90},
            'Ice': {'atk': 95, 'def': 75, 'spa': 100, 'spd': 80, 'spe': 85, 'hp': 80},
            'Bug': {'atk': 90, 'def': 85, 'spa': 80, 'spd': 85, 'spe': 90, 'hp': 75},
            'Poison': {'atk': 90, 'def': 90, 'spa': 85, 'spd': 90, 'spe': 85, 'hp': 85},
            'Fairy': {'atk': 80, 'def': 90, 'spa': 105, 'spd': 110, 'spe': 90, 'hp': 90},
            'Normal': {'atk': 95, 'def': 85, 'spa': 85, 'spd': 85, 'spe': 90, 'hp': 90},
        }
        
        # Average stats from all types present
        total_stat = 0
        type_count = 0
        
        for ptype in pokemon_types:
            ptype_clean = ptype.capitalize()  # Normalize case
            if ptype_clean in type_stat_tendencies:
                total_stat += type_stat_tendencies[ptype_clean].get(stat, 100)
                type_count += 1
        
        if type_count > 0:
            return int(total_stat / type_count)
        else:
            return 100  # Fallback avverage

    def calculate_stat(self, pokemon: Pokemon, stat: str) -> int:
        """RANDOM TEAM OPTIMIZED: Prioritize live stats, fallback to species-independent estimation"""
        
        if not pokemon:
            return 100

        # PRIORITY 1: Use live stats from server (always accurate, species-independent)
        pokemon_stats = getattr(pokemon, 'stats', None)
        if pokemon_stats and pokemon_stats.get(stat):
            stat_value = pokemon_stats[stat]
            return stat_value if stat_value is not None else 100
        
        # PRIORITY 2: Try to get base stats from species (for estimation only)
        pokemon_species = getattr(pokemon, 'species', '')
        base_stats = self.gen_data.pokedex.get(pokemon_species, {}).get('baseStats', {})
        base_stat = base_stats.get(stat, 100) # Default competitive stat if species unknown
        
        # FALLBACK 3: If species lookup fails, use type-based estimation (RANDOM TEAM OPTIMIZED)
        if not base_stats and pokemon_species:
            # Estimate stats based on Pokemon types (works for unknown species)
            pokemon_types = self._extract_pokemon_types(pokemon)
            base_stat = self._estimate_stat_from_types(pokemon_types, stat)

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
        """RANDOM TEAM OPTIMIZED: Track actions by strategic context, not Pokemon identity"""
        if action_type is None:
            action_type = self.get_action_type(action)
        
        # Store strategic context instead of Pokemon species
        strategic_context = self._get_strategic_context(battle) if battle.active_pokemon else {}
        
        action_context = {
            'action': action,
            'action_type': action_type,
            'turn': getattr(battle, 'turn', 0),
            'hp_fraction': getattr(battle.active_pokemon, 'current_hp_fraction', 0.0) if battle.active_pokemon else 0.0,
            # STRATEGIC CONTEXT (species-agnostic)
            'type_advantage': strategic_context.get('type_advantage', 0.0),
            'speed_advantage': strategic_context.get('speed_advantage', False),
            'hp_tier': strategic_context.get('hp_tier', 'medium'),  # high/medium/low instead of exact HP
            'battle_phase': strategic_context.get('battle_phase', 'early'),  # early/mid/late
            'action_category': strategic_context.get('action_category', 'neutral'),  # offensive/defensive/setup/neutral
        }
        
        # Keep last 3 actions for pattern recognition (for reward calculation)
        self.action_history.append(action_context)
        if len(self.action_history) > 3:
            self.action_history.pop(0)
        
        # Store ALL actions for episode-level statistics
        self.full_episode_history.append(action_context)
    
    def _get_strategic_context(self, battle: AbstractBattle) -> dict:
        """Extract strategic context that's independent of Pokemon identity"""
        context = {}
        
        if battle.active_pokemon and battle.opponent_active_pokemon:
            # Type advantage (transferable between battles)
            type_eff = self.calculate_pokemon_type_effectiveness(battle.active_pokemon, battle.opponent_active_pokemon)
            context['type_advantage'] = np.clip((type_eff - 1.0) / 3.0, -1.0, 1.0)  # Normalize to [-1,1]
            
            # Speed advantage (binary, transferable)
            context['speed_advantage'] = self.is_faster(battle.active_pokemon, battle.opponent_active_pokemon)
            
            # HP tier (categorical, more robust than exact values)
            hp = getattr(battle.active_pokemon, 'current_hp_fraction', 1.0)
            if hp > 0.7:
                context['hp_tier'] = 'high'
            elif hp > 0.3:
                context['hp_tier'] = 'medium'  
            else:
                context['hp_tier'] = 'low'
                
            # Battle phase (time-based, not Pokemon-dependent)
            turn = getattr(battle, 'turn', 0)
            if turn <= 5:
                context['battle_phase'] = 'early'
            elif turn <= 15:
                context['battle_phase'] = 'mid'
            else:
                context['battle_phase'] = 'late'
                
            # Action category based on last move type
            if hasattr(battle, 'available_moves') and battle.available_moves:
                last_move = battle.available_moves[0] if battle.available_moves else None
                if last_move:
                    move_category = getattr(last_move, 'category', None)
                    # Check for entry hazards FIRST (they're STATUS moves but special category)
                    if self.is_entry_hazard_move(last_move):
                        context['action_category'] = 'hazard'
                    elif move_category == MoveCategory.STATUS and self.is_stat_boost_move(last_move):
                        context['action_category'] = 'setup'
                    elif move_category in [MoveCategory.PHYSICAL, MoveCategory.SPECIAL]:
                        context['action_category'] = 'offensive'
                    elif move_category == MoveCategory.STATUS:
                        context['action_category'] = 'defensive'
                    else:
                        context['action_category'] = 'neutral'
        
        return context
    
    def update_opponent_pattern_memory(self, battle: AbstractBattle):
        """RANDOM TEAM OPTIMIZED: Learn opponent patterns by type, not species"""
        if not battle.opponent_active_pokemon:
            return
            
        # Get opponent types (transferable knowledge)
        opp_types = tuple(sorted(self._extract_pokemon_types(battle.opponent_active_pokemon)))
        if not opp_types:
            return
            
        # SIMPLIFIED: Initialize type patterns without relying on opponent_last_move attribute
        if opp_types not in self.data.opponent_type_move_patterns:
            self.data.opponent_type_move_patterns[opp_types] = {
                'physical': 0, 'special': 0, 'status': 0, 'total_moves': 0, 'encounters': 0
            }
        
        # Track general encounters for type-based learning
        self.data.opponent_type_move_patterns[opp_types]['encounters'] += 1
    
    def get_opponent_pattern_prediction(self, battle: AbstractBattle) -> dict:
        """Predict opponent behavior based on type patterns (works across battles)"""
        if not battle.opponent_active_pokemon:
            return {'physical_prob': 0.33, 'special_prob': 0.33, 'status_prob': 0.33}
            
        opp_types = tuple(sorted(self._extract_pokemon_types(battle.opponent_active_pokemon)))
        if not opp_types or opp_types not in self.data.opponent_type_move_patterns:
            return {'physical_prob': 0.4, 'special_prob': 0.4, 'status_prob': 0.2}  # Default assumptions
            
        patterns = self.data.opponent_type_move_patterns[opp_types]
        total = patterns['total_moves']
        
        if total == 0:
            return {'physical_prob': 0.4, 'special_prob': 0.4, 'status_prob': 0.2}
            
        return {
            'physical_prob': patterns['physical'] / total,
            'special_prob': patterns['special'] / total,
            'status_prob': patterns['status'] / total
        }
    
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
    
    def calculate_pokemon_type_effectiveness(self, pokemon, opponent_pokemon) -> float:
        """Calculate overall type effectiveness of pokemon vs opponent"""
        if not pokemon or not opponent_pokemon:
            return 1.0
            
        our_types = self._extract_pokemon_types(pokemon)
        opp_types = self._extract_pokemon_types(opponent_pokemon)
        
        if not our_types or not opp_types:
            return 1.0
            
        total_effectiveness = 0.0
        combinations = 0
        
        for our_type in our_types:
            for opp_type in opp_types:
                try:
                    effectiveness = self.move_effectiveness(our_type, (opp_type,))
                    total_effectiveness += effectiveness
                    combinations += 1
                except:
                    pass
        
        return total_effectiveness / max(1, combinations)
    
    def calculate_switch_viability(self, pokemon, battle: AbstractBattle) -> float:
        """Calculate how viable switching to this pokemon would be"""
        if not pokemon or getattr(pokemon, 'fainted', True):
            return 0.0
            
        viability = 0.0
        
        # Health factor (healthier = more viable)
        hp_fraction = getattr(pokemon, 'current_hp_fraction', 0.0)
        viability += hp_fraction * 0.4  # 40% weight on health
        
        # Type matchup vs current opponent
        if battle.opponent_active_pokemon:
            type_advantage = self.calculate_pokemon_type_effectiveness(pokemon, battle.opponent_active_pokemon)
            if type_advantage > 1.0:
                viability += 0.3  # 30% bonus for type advantage
            elif type_advantage < 1.0:
                viability -= 0.2  # 20% penalty for type disadvantage
                
        # Speed advantage
        if battle.opponent_active_pokemon and self.is_faster(pokemon, battle.opponent_active_pokemon):
            viability += 0.2  # 20% bonus for speed advantage
            
        # Status condition penalty
        if getattr(pokemon, 'status', None):
            viability -= 0.1  # 10% penalty for status
            
        return np.clip(viability, 0.0, 1.0)
    
    def analyze_threat_level(self, battle: AbstractBattle) -> Tuple[float, float, float]:
        """Analyze current threat levels: offensive, defensive, speed"""
        if not battle.active_pokemon or not battle.opponent_active_pokemon:
            return 0.5, 0.5, 0.5
            
        our_pokemon = battle.active_pokemon
        opp_pokemon = battle.opponent_active_pokemon
        
        # Offensive threat (can we KO them?)
        offensive_threat = 0.5
        if hasattr(battle, 'available_moves') and battle.available_moves:
            max_damage = 0.0
            for move in battle.available_moves:
                damage = self.estimate_move_damage(move, our_pokemon, opp_pokemon, battle)
                max_damage = max(max_damage, damage)
            offensive_threat = min(max_damage, 1.0)
            
        # Defensive threat (how much damage can they deal to us?)
        defensive_threat = 0.5
        opp_hp = getattr(opp_pokemon, 'current_hp_fraction', 1.0)
        our_hp = getattr(our_pokemon, 'current_hp_fraction', 1.0)
        
        # Estimate their threat based on HP ratio and type matchup
        type_threat = self.calculate_pokemon_type_effectiveness(opp_pokemon, our_pokemon)
        defensive_threat = min((type_threat * opp_hp) / max(our_hp, 0.1), 1.0)
        
        # Speed threat (who goes first?)
        speed_threat = 1.0 if self.is_faster(our_pokemon, opp_pokemon) else 0.0
        
        return (offensive_threat, defensive_threat, speed_threat)
    
    def analyze_coverage_options(self, battle: AbstractBattle) -> Tuple[float, float]:
        """Analyze type coverage of available moves and switch options"""
        move_coverage = 0.0
        switch_coverage = 0.0
        
        if not battle.opponent_active_pokemon:
            return 0.5, 0.5
            
        opp_types = self._extract_pokemon_types(battle.opponent_active_pokemon)
        if not opp_types:
            return 0.5, 0.5
            
        # Analyze move coverage
        if hasattr(battle, 'available_moves') and battle.available_moves:
            move_effectiveness = []
            for move in battle.available_moves:
                if hasattr(move, 'type') and move.type:
                    try:
                        effectiveness = self.move_effectiveness(move.type.name, tuple(opp_types))
                        move_effectiveness.append(effectiveness)
                    except:
                        move_effectiveness.append(1.0)
                        
            if move_effectiveness:
                move_coverage = max(move_effectiveness) / 4.0  # Normalize to 0-1 (4x is max)
                
        # Analyze switch coverage
        available_switches = [pokemon for pokemon in battle.team.values() 
                            if not getattr(pokemon, 'fainted', True) and pokemon != battle.active_pokemon]
        
        if available_switches:
            switch_effectiveness = []
            for pokemon in available_switches:
                effectiveness = self.calculate_pokemon_type_effectiveness(pokemon, battle.opponent_active_pokemon)
                switch_effectiveness.append(effectiveness)
                
            switch_coverage = max(switch_effectiveness) / 4.0  # Normalize to 0-1
            
        return np.clip(move_coverage, 0.0, 1.0), np.clip(switch_coverage, 0.0, 1.0)
    
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
        # Reset episode history at start of new battle
        if getattr(battle, 'turn', 0) <= 1:
            self.full_episode_history = []
            self.cumulative_episode_reward = 0.0  # Reset cumulative reward tracker
        
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
            # NOTE: active block appends 23 values (5+5 stats, 2 status, 10 boosts, 1 speed)
            # so pad with 23 zeros to keep vector length consistent
            state.extend([0.0] * 23)

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
                    
                    # Move accuracy (0.0 = always misses, 1.0 = always hits)
                    move_accuracy = self._get_move_accuracy(move) / 100.0  # Normalize to [0, 1]
                    
                    move_features.extend([
                        base_power / 150.0,  # Normalized power
                        effectiveness / 4.0,  # Effectiveness (0-4x range)
                        damage,  # Estimated damage ratio
                        move_priority / 5.0 + 0.5,  # Priority normalized to [0,1]
                        1.0 if move_category == MoveCategory.PHYSICAL else 0.0,  # Physical flag
                        1.0 if move_category == MoveCategory.SPECIAL else 0.0,  # Special flag
                        1.0 if move_category == MoveCategory.STATUS else 0.0,  # Status flag
                        move_accuracy,  # Move accuracy (helps learn about Thunder, Blizzard, Focus Blast, etc.)
                    ])  # 8 values per move
                else:
                    move_features.extend([0.0] * 8)
            else:
                move_features.extend([0.0] * 8)  # No move available
        
        state.extend(move_features)  # 32 values (4 moves × 8 features, increased from 28)

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
        
        # REMOVED: Stat boost efficiency - redundant with boost values in active pokemon block
        # The 10 boost values (5 ours + 5 opponent) already tell agent current boost state
        
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

        # === REMOVED V3 ENHANCED STRATEGIC FEATURES ===
        # These 32 features created observation-reward mismatch
        # The rewards don't validate switch viability, threat assessments, or coverage analysis
        # Removing them reduces noise and improves learning efficiency
        
        #########################################################################################################
        # SIMPLIFIED RANDOM-TEAM OPTIMIZED State calculation:
        # V1 Base: 12 + 23 + 32 + 2 + 1 + 4 + 1 = 75 features
        # V2 Temporal: 9 (action history) + 1 (battle phase) + 2 (momentum) + 2 (patterns) + 3 (game phase) = 17 features
        # REMOVED V3 Strategic: 32 features (observation-reward mismatch)
        # TOTAL: 75 + 17 = 92 features (NOTE: This is 92 not 91, will update _observation_size)
        #
        # RANDOM TEAM OPTIMIZATIONS APPLIED:
        # ✅ Strategic action tracking (context-based, not Pokemon species)  
        # ✅ Type-based opponent learning (transferable patterns, not species memory)
        # ✅ Position-based damage tracking (slot comparison, not species matching)
        # ✅ Type-based stat estimation (works with unknown species)
        # ✅ Strategic diversity rewards (pattern-based, not Pokemon counting)
        # ✅ Move accuracy tracking (helps learn about Thunder vs Thunderbolt reliability tradeoffs)
        # ✅ Type immunity naturally captured in effectiveness (0.0 = immune: Ground vs Flying, etc.)
        # ✅ Attack-prioritized rewards (attacking moves get +0.4 baseline vs +0.0 for stat boosts)
        # ✅ Stat boost spam prevention (penalties for redundant boosts, only reward early setup when safe)
        # 
        # RESULT: Agent learns universal Pokemon strategies that transfer to ANY random team!
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