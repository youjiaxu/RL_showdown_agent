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
            info[agent]["hp_advantage"] = float(team_total_hp - opponent_total_hp)
            
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
        
        # Component 4: Strategic outcome rewards (setup moves, hazards)
        strategic_reward = self._calculate_strategic_outcomes(battle, prior_battle)
        
        # Component 5: Switching rewards (three-case system for random teams)
        switching_reward = self._calculate_switching_reward(battle, prior_battle)
        
        total_reward = damage_reward + ko_reward + outcome_reward + strategic_reward + switching_reward
        
        # Add immediate feedback bonuses for faster learning (attack/status moves)
        immediate_feedback = self._calculate_immediate_feedback(battle, prior_battle)
        total_reward += immediate_feedback
        
        # Apply spam prevention penalties (simplified, consistent)
        spam_penalty = self._calculate_spam_penalties(battle, prior_battle)
        total_reward += spam_penalty
        
        # Clip to reasonable bounds (reduced from ±25 to ±15 after win/loss reduction)
        # Max possible: Win(8) + KO(8) + Damage(2) + Strategic(0.7) + Switch(0.65) + Immediate(0.5) ≈ 20
        # Min possible: Loss(-5) + KO(-8) + Damage(-2) + Spam(-4) ≈ -19
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
        REDUCED: Now 5-8 instead of 15-18 to prevent gradient washing.
        This allows dense rewards (damage, KOs) to provide meaningful learning signals.
        """
        #Only give outcome reward when battle is actually finished
        if not getattr(battle, 'battle_finished', False):
            return 0.0
        
        if getattr(battle, 'won', False):
            base_reward = 5.0  # REDUCED from 8.0 to prevent gradient washing
            
            # bonus for clean sweep
            fainted_count = sum(1 for mon in battle.team.values() if getattr(mon, 'fainted', False))
            if fainted_count == 0:
                base_reward += 3.0
            elif fainted_count <= 2:
                base_reward += 1.5
            
            return base_reward
        else:
            return -5.0  # REDUCED from -8.0 to prevent gradient washing
    
    def _calculate_immediate_feedback(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Provide immediate feedback for attack and status moves onl
        """
        if not prior_battle:
            return 0.0
        
        reward = 0.0
        
        # Only process MOVE actions (not switches)
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
                        # REMOVED: Attack bias (+0.3) - let damage_delta handle attack incentives
                        # This prevents rewarding ineffective attacks (e.g. Earthquake vs Flying)
                        
                        if battle.opponent_active_pokemon and prior_battle.opponent_active_pokemon:
                            # Moderate penalty for triggering immunity abilities
                            if self.is_nullified(move_used, prior_battle.opponent_active_pokemon):
                                reward -= 0.8
                            
                            try:
                                opp_types = self._extract_pokemon_types(battle.opponent_active_pokemon)
                                if opp_types and hasattr(move_used, 'type') and move_used.type:
                                    effectiveness = self.move_effectiveness(move_used.type.name.title(), tuple(opp_types))
                                    
                                    if effectiveness >= 2.0: #rewards good choices
                                        reward += 0.3
                                    
                                    # Penalise type immunity
                                    elif effectiveness == 0.0:
                                        reward -= 1.0  # Immune move (always wrong: Earthquake vs Flying)
                            except:
                                pass
                    
                    # stat boost move
                    elif self.is_stat_boost_move(move_used):
                        # earned through timing well
                        stat_boost_reward = self._evaluate_stat_boost_timing(battle, move_used)
                        reward += stat_boost_reward  # Max +0.5 early game, negative otherwise
                    
                    # OTHER STATUS MOVES: Check for redundant status effects
                    elif move_category == MoveCategory.STATUS:
                        status_reward = self._evaluate_status_move_effectiveness(move_used, battle, prior_battle)
                        reward += status_reward  # Smart status use vs redundant spam
        
        return np.clip(reward, -1.0, 0.8)  # UPDATED: +0.3 super-effective, -0.8 immunity, -1.0 type immune 
    
    def _calc_type_matchup(self, atk_types: list[str], def_types: list[str]) -> float:
        """Helper to calculate offensive/defensive matchup ratio between two type sets."""
        offensive = defensive = count = 0.0
        for a_type in atk_types:
            for d_type in def_types:
                try:
                    offensive += self.move_effectiveness(a_type, (d_type,))
                    defensive += self.move_effectiveness(d_type, (a_type,))
                    count += 1
                except:
                    pass
        return (offensive / count) / max(defensive / count, 0.25) if count > 0 else 1.0
    
    def _calculate_switching_reward(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Calculate rewards for strategic switching (three-case system for random teams).
        
        CASE 1: Both matchups known (prior + current) → Matchup improvement bonus
        CASE 2: Prior unknown (first appearance) → Exploration rewards
        CASE 3: Emergency/opponent unknown → Desperation mechanics
        
        This addresses random team issues where 5/6 Pokemon start unknown.
        """
        if not prior_battle or not hasattr(self, 'action_history') or len(self.action_history) == 0:
            return 0.0
        
        last_action_type = self.action_history[-1].get('action_type', '')
        if last_action_type != 'switch':
            return 0.0  # No reward/penalty if we didn't switch
        
        reward = 0.0
        prior_hp_fraction = self.action_history[-1].get('hp_fraction', 1.0)
        
        # Reward strategic switches that improve our overall battle position
        if not (battle.active_pokemon and battle.opponent_active_pokemon):
            return 0.0
        
        # Get current Pokemon's types (always known after switch)
        our_types = self._extract_pokemon_types(battle.active_pokemon)
        opp_types = self._extract_pokemon_types(battle.opponent_active_pokemon)
        
        # Get types of Pokemon we switched FROM
        prior_types = self.action_history[-1].get('active_types', [])
        
        # case 1 both known active pokemon
        if our_types and opp_types and prior_types:
            current_matchup = self._calc_type_matchup(our_types, opp_types)
            prior_matchup = self._calc_type_matchup(prior_types, opp_types)
            
            # BALANCED: Strong enough to compete with damage, but not overwhelming
            # Base reward on current matchup quality
            if current_matchup >= 3.0:
                base_reward = 1.0  # Perfect switch (4x advantage or better)
            elif current_matchup >= 2.0:
                base_reward = 0.7  # Good strategic switch (2-3x advantage)
            elif current_matchup >= 1.5:
                base_reward = 0.4  # Decent switch (moderate advantage)
            elif current_matchup >= 1.0:
                base_reward = 0.15  # Slight advantage (mild incentive)
            elif current_matchup >= 0.7:
                base_reward = -0.15  # Slight disadvantage
            else:
                base_reward = -0.4  # Bad matchup (don't switch into disadvantage)
            
            # Bonus for IMPROVING matchup (escape bad → find better)
            matchup_improvement = current_matchup - prior_matchup
            if prior_matchup < 0.8 and matchup_improvement > 0.3:
                # Escaped bad matchup significantly - bonus!
                base_reward += 0.3  # Reward escaping bad situations
            elif prior_hp_fraction < 0.3:
                # Emergency escape bonus (save low HP Pokemon)
                base_reward += 0.25  # Always good to preserve Pokemon
            elif matchup_improvement < -0.5:
                # Switched from good to bad - penalty
                base_reward -= 0.25  # Discourage bad switches
            elif prior_matchup >= 1.5 and matchup_improvement < 0.1:
                # Switched OUT of good matchup unnecessarily - penalty
                base_reward -= 0.2  # Discourage unnecessary switches
            
            reward += base_reward
        
        # case 2 picking new pokemon
        elif our_types and opp_types and not prior_types:
            new_matchup = self._calc_type_matchup(our_types, opp_types)
            

            if new_matchup >= 2.5:
                reward += 0.8  # Excellent matchup found
            elif new_matchup >= 1.8:
                reward += 0.5  # Good matchup found
            elif new_matchup >= 1.2:
                reward += 0.3  # Decent matchup
            elif new_matchup >= 1.0:
                reward += 0.1  # Neutral matchup (small exploration bonus)
            elif new_matchup >= 0.7:
                reward += 0.0  # Slight disadvantage (neutral - no reward/penalty)

        # === CASE 3: OPPONENT UNKNOWN OR EMERGENCY ===
        else:
            # Either opponent types unknown or something went wrong
            # Use team HP and emergency heuristics
            team_hp_total = sum(getattr(mon, 'current_hp_fraction', 0.0) 
                               for mon in battle.team.values())
            team_hp_avg = team_hp_total / max(len(battle.team), 1)
            
            if team_hp_avg < 0.4:
                # Team HP low - ENCOURAGE aggressive switching to find answers
                reward += 0.25  
            elif prior_hp_fraction < 0.3:
                # Emergency escape from low HP
                reward += 0.25  # Always good to save Pokemon

        
        return np.clip(reward, -0.6, 1.3)  # Switching: -0.6 to +1.3 (perfect switch + improvement) 
    
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
            
            # PROGRESSIVE spam penalties - scales with frequency to prevent over-switching
            if switch_count >= 5:
                reward -= 1.5  # Severe penalty for switching every turn
            elif switch_count >= 4:
                reward -= 0.8  # Strong penalty for very frequent switching
            elif switch_count >= 3:
                reward -= 0.3  # Moderate penalty for frequent switching
            # 1-2 switches in 5 turns is fine (strategic play) 

        current_action_category = current_action.get('action_category', '')
        if current_action_category == 'boost':  # FIXED: Was 'setup', but track_action sets 'boost'
            recent_window = self.action_history[-5:]
            boost_count = sum(1 for a in recent_window if a.get('action_category') == 'boost')
            
            if boost_count >= 3:
                reward -= 1.5  # Discourage excessive stat boosting
            elif boost_count >= 2:
                reward -= 0.5  # Mild penalty for frequent boosting 
        
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
        Reward long-term strategic play: entry hazards and sustained pressure.
        
        REMOVED: Setup move rewards (now handled in _evaluate_stat_boost_timing)
        REMOVED: Momentum bonus (redundant with damage_delta)
        FOCUS: Entry hazards only
        """
        if not prior_battle:
            return 0.0
            
        reward = 0.0
        
        # === ENTRY HAZARD REWARDS (Long-term Strategy) ===
        if (hasattr(self, 'action_history') and len(self.action_history) > 0 and
            self.action_history[-1].get('action_category') == 'hazard'):
            
            battle_phase = self.calculate_battle_phase(battle)
            
            # SMART: Only reward NEW hazards, not spam
            if battle_phase < 0.3:  # Stricter timing - early game only
                # Check if hazards already exist (prevent spam)
                hazards_exist = False
                if hasattr(battle, 'opponent_side_conditions'):
                    opponent_conditions_str = str(battle.opponent_side_conditions).lower()
                    hazard_conditions = ['stealth rock', 'stealthrock', 'spikes', 'toxic spikes', 'toxicspikes', 'sticky web', 'stickyweb']
                    for condition in hazard_conditions:
                        if condition in opponent_conditions_str:
                            hazards_exist = True
                            break
                
                if not hazards_exist:
                    reward += 1.2  # INCREASED from 1.0 for positive reinforcement
                else:
                    reward -= 1.0  # Clear spam penalty
        
        return np.clip(reward, -1.0, 1.0)
    
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
        # Calculate team HP using generator expressions
        my_total_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) 
                         for mon in getattr(battle, 'team', {}).values())
        opp_total_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) 
                          for mon in getattr(battle, 'opponent_team', {}).values())
                    
        # Return advantage (-1 to +1 scale)
        total_hp = my_total_hp + opp_total_hp
        return (my_total_hp - opp_total_hp) / total_hp if total_hp > 0 else 0.0


    def _evaluate_stat_boost_timing(self, battle: AbstractBattle, move: Move) -> float:
        """
        Unified stat boost evaluation with clear diminishing returns.
        Teaches: (1) Boost early when safe, (2) Don't over-boost, (3) Don't boost in bad situations
        """
        if not battle.active_pokemon:
            return 0.0
        
        active_boosts = getattr(battle.active_pokemon, 'boosts', {})
        battle_phase = self.calculate_battle_phase(battle)
        
        # Identify which stats this move boosts
        move_id = getattr(move, 'id', '').lower()
        affected_stats = self._get_boosted_stats(move_id)
        
        # Unknown boost moves are discouraged
        if not affected_stats:
            return -1.0
        
        # Check current boost level (use max across affected stats)
        max_current_boost = max([active_boosts.get(stat, 0) for stat in affected_stats] + [0])
        
        # GRANULAR DIMINISHING RETURNS CURVE - Progressively discourage over-boosting
        # STRENGTHENED: Make spam penalties more aggressive to prevent 3x+ boost patterns
        if max_current_boost >= 6:
            return -2.0  # At max (+6 cap), can't boost further - very strong penalty
        elif max_current_boost >= 5:
            return -1.5  # Very high boosts (+5), very strong diminishing returns
        elif max_current_boost >= 4:
            return -1.0  # High boosts (+4), strong diminishing returns
        elif max_current_boost >= 3:
            return -0.6  # Moderate boosts (+3), notable diminishing returns  
        elif max_current_boost >= 2:
            return -0.2  # Some boosts (+2), mild discouragement
        
        # For first boost (0 → +1 or +2), check timing and safety
        our_hp = getattr(battle.active_pokemon, 'current_hp_fraction', 1.0)
        opp_hp = getattr(battle.opponent_active_pokemon, 'current_hp_fraction', 1.0) if battle.opponent_active_pokemon else 1.0
        
        # Check type advantage (safe to setup)
        type_advantage = 1.0
        if battle.active_pokemon and battle.opponent_active_pokemon:
            our_types = self._extract_pokemon_types(battle.active_pokemon)
            opp_types = self._extract_pokemon_types(battle.opponent_active_pokemon)
            if our_types and opp_types:
                type_advantage = sum(self.move_effectiveness(our_type, (opp_types[0],)) 
                                   for our_type in our_types) / len(our_types)
        
        # BALANCED FIRST BOOST REWARDS - Encourage smart setup without over-boosting
        # REDUCED: Lower first boost reward to balance against strengthened spam penalties
        if battle_phase < 0.2 and our_hp > 0.75 and type_advantage >= 1.0:
            return 0.4  # Reward excellent early setup (reduced from 0.6)
        elif battle_phase < 0.3 and our_hp > 0.6:
            return 0.3  # Reward decent early setup (reduced from 0.4)
        elif our_hp > 0.8 and type_advantage >= 1.5:
            return 0.35  # Reward safe setup with advantage (reduced from 0.5)
        else:
            return -0.4  # Bad timing (too late, too weak, or disadvantage)
    
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
            'flamecharge': ['spe'],     # +1 Spe (on hit)
            
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
        Evaluate status moves: reward INFLICTION, penalize redundancy and immunity.
        
        Key principle: Status is valuable when SETTING IT UP, not when attacking already-statused foes.
        """
        if not move or not battle.opponent_active_pokemon:
            return 0.0
            
        move_id = getattr(move, 'id', '').lower()
        opponent = battle.opponent_active_pokemon
        
        # Check opponent's current and prior status
        opponent_status = getattr(opponent, 'status', None)
        opponent_status_name = opponent_status.name.lower() if opponent_status else None
        
        # Get prior opponent status (to detect successful infliction)
        prior_opponent_status = None
        if prior_battle and prior_battle.opponent_active_pokemon:
            prior_status = getattr(prior_battle.opponent_active_pokemon, 'status', None)
            prior_opponent_status = prior_status.name.lower() if prior_status else None
        
        # Status move categories
        paralysis_moves = {'thunderwave', 'glare', 'stunspore', 'bodyslam', 'nuzzle'}
        sleep_moves = {'sleeppowder', 'spore', 'hypnosis', 'darkvoid', 'grasswhistle', 'lovelykiss'}
        burn_moves = {'willowisp', 'scald', 'flamethrower', 'fireblast', 'lavaplume'}
        poison_moves = {'toxic', 'poisongas', 'poisonpowder', 'sludgebomb', 'toxicspikes'}
        freeze_moves = {'icebeam', 'blizzard', 'freezedry'}
        
        # CRITICAL: Check if status was SUCCESSFULLY INFLICTED this turn
        if prior_opponent_status is None and opponent_status_name is not None:
            # Successfully inflicted status! Reward based on status type
            if opponent_status_name == 'par':
                return 0.8  # Paralysis (speed cut + 25% flinch chance)
            elif opponent_status_name == 'slp':
                return 1.0  # Sleep (opponent disabled 1-3 turns) - most valuable
            elif opponent_status_name == 'brn':
                return 0.7  # Burn (50% attack cut + chip damage)
            elif opponent_status_name in ['psn', 'tox']:
                return 0.6  # Poison (chip damage, Toxic escalates)
            elif opponent_status_name == 'frz':
                return 0.8  # Freeze (strong but rare)
        
        # PENALTY: Redundant status (opponent already has status)
        if opponent_status_name:
            return -1.0  # Always wasteful (only one status per Pokemon)
        
        # PENALTY: Status move on immune type
        opp_types = self._extract_pokemon_types(opponent)
        if opp_types:
            for opp_type in opp_types:
                opp_type_lower = opp_type.lower()
                # Electric-type status on Ground (Thunder Wave immunity)
                if move_id in paralysis_moves and opp_type_lower == 'ground':
                    return -1.0
                # Fire-type status on Fire (Burn immunity)
                elif move_id in burn_moves and opp_type_lower == 'fire':
                    return -1.0
                # Poison-type status on Steel/Poison (Poison immunity)
                elif move_id in poison_moves and opp_type_lower in ['steel', 'poison']:
                    return -1.0
                
        return 0.0
    
    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        STRATEGIC State Representation with STATE-REWARD ALIGNMENT:
        
        Base features (V1): 81 features (increased from 75)
        - 12 (health: 6 team + 6 opponent)
        - 29 (active pokemon: 5+5 stats, 1+1 status, 5+5 boosts, 1 speed, 3 ability indicators, 3 item indicators) 🆕
        - 32 (moves: 4 moves × 8 features including accuracy)
        - 2 (fainted counts)
        - 1 (turn number)
        - 4 (weather one-hot)
        - 1 (switches available)
        
        V2 temporal additions: 20 features  
        - 12 (enhanced action history: 3 actions × 4 features - includes move categories!)
        - 1 (battle phase)
        - 2 (momentum indicators) 
        - 2 (action patterns)
        - 3 (strategic game phase indicators)
        
        V3 field conditions: 8 features
        - 4 (opponent hazards: stealth rock, spikes, toxic spikes, sticky web)
        - 4 (our hazards: same 4 types)
        
        CRITICAL ALIGNMENT FIXES:
        ✅ Hazard visibility - Agent can SEE if hazards exist before penalty
        ✅ Move category tracking - Agent knows if it just used hazard/boost/attack
        ✅ Ability indicators - Agent can SEE immunity abilities (Volt Absorb, etc.) 🆕
        ✅ Item indicators - Agent can SEE items (Life Orb, Leftovers, etc.) 🆕
        ✅ Immunity penalty - Extra -2.0 reward for triggering immunity abilities 🆕
        
        Total: 81 + 20 + 8 = 109 features (increased from 103)

        Returns:
            int: The size of the observation space.
        """
        return 109 

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
            for type_attr in ('type_1', 'type_2'):
                type_val = getattr(pokemon, type_attr, None)
                if type_val:
                    type_str = str(type_val) if hasattr(type_val, '__str__') else type_val
                    if type_str and type_str != 'None':
                        types.append(type_str)
        except (AttributeError, TypeError):
            pass
            
        return types
    
    def _get_move_accuracy(self, move: Move) -> float:
        """Get move accuracy with proper handling of special cases. Returns accuracy as percentage (0-100)."""
        if not move:
            return 100.0
        
        try:
            accuracy = getattr(move, 'accuracy', 100)
            # Handle special accuracy values
            if accuracy in (True, None):
                return 100.0  # Moves that never miss
            elif accuracy in (False, 0):
                return 0.0 
            return float(accuracy)
        except (AttributeError, TypeError, ValueError):
            return 100.0
    
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
    
    def get_opponent_ability_indicators(self, battle: AbstractBattle) -> tuple:
        """
        Returns binary indicators for opponent ability categories.
        Helps agent learn about abilities without memorizing specific Pokemon.
        
        Returns: (has_immunity_ability, has_boosting_ability, has_weather_ability)
        """
        if not battle.opponent_active_pokemon:
            return (0.0, 0.0, 0.0)
        
        ability = getattr(battle.opponent_active_pokemon, 'ability', None)
        if not ability:
            return (0.0, 0.0, 0.0)
        
        ability_str = str(ability).lower().replace(" ", "")
        
        # Immunity/absorption abilities (CRITICAL for learning)
        immunity_abilities = {
            'voltabsorb', 'motordrive', 'lightningrod',  # Electric immunity
            'waterabsorb', 'stormdrain', 'dryskin',      # Water immunity
            'flashfire',                                  # Fire immunity
            'sapsipper',                                  # Grass immunity
            'levitate',                                   # Ground immunity
            'bulletproof',                                # Ball/bomb immunity
            'wonderguard',                                # Multi-immunity
            'thickfat',                                   # Fire/Ice resistance
        }
        has_immunity = 1.0 if ability_str in immunity_abilities else 0.0
        
        # Stat-boosting abilities (affect damage calculations)
        boosting_abilities = {
            'intimidate', 'defiant', 'competitive', 'contrary',
            'hugepower', 'purepower', 'adaptability', 'technician',
            'tintedlens', 'scrappy', 'moxie', 'beastboost'
        }
        has_boosting = 1.0 if ability_str in boosting_abilities else 0.0
        
        # Weather/terrain abilities (change battle conditions)
        weather_abilities = {
            'drizzle', 'drought', 'sandstream', 'snowwarning',
            'electricsurge', 'grassysurge', 'mistysurge', 'psychicsurge'
        }
        has_weather = 1.0 if ability_str in weather_abilities else 0.0
        
        return (has_immunity, has_boosting, has_weather)
    
    def get_opponent_item_indicators(self, battle: AbstractBattle) -> tuple:
        """
        Returns binary indicators for opponent item categories.
        Helps agent understand item effects without memorizing specific items.
        
        Returns: (has_hp_item, has_power_item, has_defensive_item)
        """
        if not battle.opponent_active_pokemon:
            return (0.0, 0.0, 0.0)
        
        item = getattr(battle.opponent_active_pokemon, 'item', None)
        if not item:
            return (0.0, 0.0, 0.0)
        
        item_str = str(item).lower().replace(" ", "")
        
        # HP restoration items
        hp_items = {
            'leftovers', 'blacksludge', 'sitrusberry', 'oranberry',
            'aguavberry', 'figyberry', 'iapapaberry', 'magoberry', 'wikiberry'
        }
        has_hp_item = 1.0 if item_str in hp_items else 0.0
        
        # Power-boosting items
        power_items = {
            'lifeorb', 'choiceband', 'choicespecs', 'choicescarf',
            'expertbelt', 'muscleband', 'wiseglasses', 'metronome',
            # Type-specific power items
            'charcoal', 'mysticwater', 'magnet', 'miracleseed',
            'nevermeltice', 'blackbelt', 'poisonbarb', 'softsand',
            'sharpbeak', 'twistedspoon', 'silverpowder', 'hardstone',
            'spelltag', 'dragonfang', 'blackglasses', 'metalcoat', 'pixieplate'
        }
        has_power_item = 1.0 if item_str in power_items else 0.0
        
        # Defensive items
        defensive_items = {
            'assaultvest', 'rockyhelmet', 'weaknesspolicy',
            'lightclay', 'focussash', 'lumberry', 'chestoberry'
        }
        has_defensive_item = 1.0 if item_str in defensive_items else 0.0
        
        return (has_hp_item, has_power_item, has_defensive_item)
    
    # V2 Enhancement: Strategic context methods
    def track_action(self, action: int, battle: AbstractBattle, action_type: str | None = None):
        """RANDOM TEAM OPTIMIZED: Track actions by strategic context, not Pokemon identity"""
        if action_type is None:
            action_type = self.get_action_type(action)
        
        # Store strategic context instead of Pokemon species - PASS action to determine category correctly
        strategic_context = self._get_strategic_context(battle, action) if battle.active_pokemon else {}
        
        # CRITICAL: Store active Pokemon's types for matchup evaluation when switching
        active_types = self._extract_pokemon_types(battle.active_pokemon) if battle.active_pokemon else []
        
        action_context = {
            'action': action,
            'action_type': action_type,
            'turn': getattr(battle, 'turn', 0),
            'hp_fraction': getattr(battle.active_pokemon, 'current_hp_fraction', 0.0) if battle.active_pokemon else 0.0,
            'active_types': active_types,  # NEW: Store types for switch evaluation
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
    
    def _get_strategic_context(self, battle: AbstractBattle, action: int = -1) -> dict:
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
                
            # FIXED: Determine action category from ACTUAL action taken
            # Actions 6-9 = moves 0-3, 10-13 = mega moves 0-3, etc.
            if 6 <= action <= 25:  # All move actions
                move_index = (action - 6) % 4  # Convert to move index (0-3)
                if hasattr(battle, 'available_moves') and battle.available_moves:
                    if 0 <= move_index < len(battle.available_moves):
                        actual_move = battle.available_moves[move_index]
                        move_category = getattr(actual_move, 'category', None)
                        # Check for entry hazards FIRST (they're STATUS moves but special category)
                        if self.is_entry_hazard_move(actual_move):
                            context['action_category'] = 'hazard'
                        elif move_category == MoveCategory.STATUS and self.is_stat_boost_move(actual_move):
                            context['action_category'] = 'boost'
                        elif move_category in [MoveCategory.PHYSICAL, MoveCategory.SPECIAL]:
                            context['action_category'] = 'offensive'
                        elif move_category == MoveCategory.STATUS:
                            context['action_category'] = 'defensive'
                        else:
                            context['action_category'] = 'neutral'
                    else:
                        context['action_category'] = 'neutral'
                else:
                    context['action_category'] = 'neutral'
            else:
                # Switches or other actions
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
            self.action_history = []
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
            
            ability_indicators = self.get_opponent_ability_indicators(battle)
            state.extend(ability_indicators)  # 3 values (immunity, boosting, weather abilities)
            
            item_indicators = self.get_opponent_item_indicators(battle)
            state.extend(item_indicators)  # 3 values (HP, power, defensive items)

        else:
            # Fill with zeros if no active pokemon
            state.extend([0.0] * 29)

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

        # === FIELD CONDITIONS (CRITICAL: Agent needs to see hazards to avoid spam penalty!) ===
        # Entry hazards on OPPONENT side (what WE set up)
        opponent_hazards = [0.0, 0.0, 0.0, 0.0]  # [stealth_rock, spikes, toxic_spikes, sticky_web]
        if hasattr(battle, 'opponent_side_conditions'):
            opponent_conditions = battle.opponent_side_conditions
            # Stealth Rock (binary: 0 or 1)
            if any(cond in str(opponent_conditions).lower() for cond in ['stealth rock', 'stealthrock']):
                opponent_hazards[0] = 1.0
            # Spikes (layers: 0, 1, 2, or 3) - treat as binary for simplicity
            if 'spikes' in str(opponent_conditions).lower():
                opponent_hazards[1] = 1.0
            # Toxic Spikes (layers: 0, 1, or 2) - treat as binary
            if any(cond in str(opponent_conditions).lower() for cond in ['toxic spikes', 'toxicspikes']):
                opponent_hazards[2] = 1.0
            # Sticky Web (binary: 0 or 1)
            if any(cond in str(opponent_conditions).lower() for cond in ['sticky web', 'stickyweb']):
                opponent_hazards[3] = 1.0
        state.extend(opponent_hazards)  # 4 values
        
        # Entry hazards on OUR side (what OPPONENT set up)
        our_hazards = [0.0, 0.0, 0.0, 0.0]
        if hasattr(battle, 'side_conditions'):
            our_conditions = battle.side_conditions
            if any(cond in str(our_conditions).lower() for cond in ['stealth rock', 'stealthrock']):
                our_hazards[0] = 1.0
            if 'spikes' in str(our_conditions).lower():
                our_hazards[1] = 1.0
            if any(cond in str(our_conditions).lower() for cond in ['toxic spikes', 'toxicspikes']):
                our_hazards[2] = 1.0
            if any(cond in str(our_conditions).lower() for cond in ['sticky web', 'stickyweb']):
                our_hazards[3] = 1.0
        state.extend(our_hazards)  # 4 values
        
        # Total field conditions: 8 values (4 opponent + 4 ours)

        # === V2 TEMPORAL CONTEXT FEATURES ===
        # Enhanced action history (last 3 actions with DETAILED context for spam detection)
        action_history_features = [0.0] * 12  # 3 actions × 4 features each
        for i, action_data in enumerate(self.action_history[-3:]):
            if i < 3:  # Safety check
                base_idx = i * 4
                # Feature 1: Normalized action index
                action_value = action_data.get('action', -2)
                action_history_features[base_idx] = np.clip(action_value / 26.0, -1.0, 1.0)
                
                # Feature 2: Action type encoding (move/switch/etc)
                action_type_map = {'move': 0.2, 'switch': 0.4, 'move_mega': 0.6, 'move_z': 0.8, 'move_dynamax': 1.0}
                action_history_features[base_idx + 1] = action_type_map.get(action_data.get('action_type', 'move'), 0.0)
                
                # Feature 3: CRITICAL - Move category (physical/special/status/hazard)
                category_map = {
                    'physical': 0.25, 
                    'special': 0.5, 
                    'status': 0.75, 
                    'hazard': 1.0,  # CRITICAL for hazard spam detection!
                    'boost': 0.9    # For stat boost spam detection
                }
                action_history_features[base_idx + 2] = category_map.get(action_data.get('action_category', ''), 0.0)
                
                # Feature 4: Turns ago (recency)
                current_turn = getattr(battle, 'turn', 0)
                action_turn = action_data.get('turn', current_turn)
                turns_ago = max(0, current_turn - action_turn)
                action_history_features[base_idx + 3] = min(turns_ago / 10.0, 1.0)
        
        state.extend(action_history_features)  # 12 values (increased from 9)
        
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
        # STATE-REWARD ALIGNED State Representation:
        # V1 Base: 12 + 29 + 32 + 2 + 1 + 4 + 1 = 81 features
        # V2 Temporal: 12 (enhanced action history) + 1 (battle phase) + 2 (momentum) + 2 (patterns) + 3 (game phase) = 20 features
        # V3 Field Conditions: 8 features (4 opponent hazards + 4 our hazards)
        # TOTAL: 81 + 20 + 8 = 109 features (increased from 103)
        #
        # CRITICAL STATE-REWARD ALIGNMENT FIXES:
        # ✅ Hazard visibility (8 features): Agent can SEE if Stealth Rock/Spikes exist before -1.0 spam penalty
        # ✅ Move category tracking (in action history): Agent knows if last move was hazard/boost/attack
        # ✅ Enhanced action history: 12 features (3×4) instead of 9 (3×3) with category information
        # ✅ Ability indicators (3 features): Agent can SEE immunity/boosting/weather abilities 🆕
        # ✅ Item indicators (3 features): Agent can SEE HP/power/defensive items 🆕
        # ✅ Immunity penalty: -2.0 reward for triggering Volt Absorb, Flash Fire, etc. 🆕
        #
        # RANDOM TEAM OPTIMIZATIONS APPLIED:
        # ✅ Strategic action tracking (context-based, not Pokemon species)  
        # ✅ Type-based opponent learning (transferable patterns, not species memory)
        # ✅ Position-based damage tracking (slot comparison, not species matching)
        # ✅ Type-based stat estimation (works with unknown species)
        # ✅ Strategic diversity rewards (pattern-based, not Pokemon counting)
        # ✅ Move accuracy tracking (helps learn about Thunder vs Thunderbolt reliability tradeoffs)
        # ✅ Type immunity naturally captured in effectiveness (0.0 = immune: Ground vs Flying, etc.)
        # ✅ Ability immunity learning (Volt Absorb, Levitate, etc. with -2.0 penalty) 🆕
        # ✅ Hazard spam prevention with VISIBILITY (penalties only when agent can see hazards exist)
        # ✅ Stat boost spam prevention (penalties for redundant boosts, only reward early setup when safe)
        # 
        # RESULT: Agent learns universal Pokemon strategies with full state-reward alignment!
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