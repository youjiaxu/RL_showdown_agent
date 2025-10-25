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
        
        # Action history tracking for reward calculation
        self.action_history = []
        
        # STATE-REWARD SYNCHRONIZATION CACHE
        # Store matchup calculations from embed_battle() to reuse in rewards
        self.matchup_cache = {
            'our_offensive': 0.0,          # Current active Pokemon's type advantage
            'our_threat_to_them': 1.0,     # Threat we pose to opponent
            'their_threat_to_us': 1.0,     # Threat opponent poses to us
            'revealed_opponents': 0,        # How many opponent Pokemon revealed
            'total_opponents': 6            # Total opponent Pokemon (usually 6)
        }

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
                hp_component = self._calculate_hp_progress(self.battle1, prior_battle) * 0.1  # Scaled
                contextual_component = self._calculate_contextual_action_reward(self.battle1, prior_battle) * 0.05  # Scaled
                outcome_component = self._calculate_battle_outcome(self.battle1)
                if outcome_component > 0:
                    outcome_value = 1.0  # Normalized win
                elif outcome_component < 0:
                    outcome_value = -0.3  # Normalized loss
                else:
                    outcome_value = 0.0
                
                info[agent]["reward_hp"] = float(hp_component)
                info[agent]["reward_contextual"] = float(contextual_component)
                info[agent]["reward_outcome"] = float(outcome_value)
                info[agent]["reward_total"] = float(hp_component + contextual_component + outcome_value)
            except (AttributeError, Exception):
                info[agent]["reward_hp"] = 0.0
                info[agent]["reward_contextual"] = 0.0
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
            
            # === DIAGNOSTIC METRICS FOR DEBUGGING ===
            # These help identify training issues early
            
            # Reward scale check (should stay in [-1, 1.5] range)
            try:
                current_reward = self.calc_reward(self.battle1)
                info[agent]["current_step_reward"] = float(current_reward)
                info[agent]["reward_magnitude"] = float(abs(current_reward))
            except:
                info[agent]["current_step_reward"] = 0.0
                info[agent]["reward_magnitude"] = 0.0
            
            # Action distribution (detect if agent is stuck in patterns)
            if self.action_history:
                recent_actions = self.action_history[-10:] if len(self.action_history) >= 10 else self.action_history
                move_count = sum(1 for a in recent_actions if a.get('action_type', '').startswith('move'))
                switch_count = sum(1 for a in recent_actions if a.get('action_type') == 'switch')
                
                info[agent]["recent_move_percentage"] = float(move_count / max(len(recent_actions), 1))
                info[agent]["recent_switch_percentage"] = float(switch_count / max(len(recent_actions), 1))

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        NORMALIZED REWARD SYSTEM FOR STABLE Q-LEARNING
        
        Design principles:
        1. Rewards normalized to ±1 scale for stable Q-values
        2. Per-step rewards provide dense learning signal
        3. Terminal rewards dominate but don't cause instability
        4. Q-values stay in stable range [-10, +10]
        
        Per-Step Rewards (called every turn):
        1. HP Progress: ±0.01/step (gentle guidance)
        2. Move Context: ±0.005/step (timing, hazard spam)
        3. Switch Context: ±0.02/step (offensive, defensive, opponent reactions)
        4. Total per-step: ±0.035/step → ±1.4 over 40 turns
        
        Terminal Rewards (once per episode):
        5. Win: +1.0 (normalized)
        6. Loss: -0.3 (smaller penalty to encourage risk-taking)
        
        Expected Episode Totals (40 turns):
        - Good win: +1.0 + 0.5 = +1.5
        - Average win: +1.0 + 0.2 = +1.2
        - Bad win: +1.0 - 0.1 = +0.9 (still positive!)
        - Good loss: -0.3 + 0.2 = -0.1 (still negative!)
        - Bad loss: -0.3 - 0.5 = -0.8
        
        Q-value stability:
        - Rewards in [-0.8, +1.5] range → Q-values stay bounded
        - Learning rate 0.0001 × 1.5 = 0.00015 per update (stable)
        - Clear win/loss separation: worst win (+0.9) > best loss (-0.1)
        """
        try:
            prior_battle = self._get_prior_battle(battle)
        except AttributeError:
            prior_battle = None
        
        reward = 0.0
        
        # Check if battle is finished for terminal rewards
        battle_finished = getattr(battle, 'finished', False)
        
        if battle_finished:
            # TERMINAL REWARDS (normalized scale)
            outcome = self._calculate_battle_outcome(battle)
            if outcome > 0:  # Win
                reward += 1.0
            elif outcome < 0:  # Loss
                reward -= 0.3
        else:
            # PER-STEP REWARDS (dense signal for learning)
            
            # Component 1: HP Progress (±0.01 per step)
            hp_progress = self._calculate_hp_progress(battle, prior_battle)
            reward += hp_progress * 0.1  # Scale down from ±0.10 to ±0.01
            
            # Component 2: Contextual Actions (±0.025 per step)
            contextual_reward = self._calculate_contextual_action_reward(battle, prior_battle)
            reward += contextual_reward * 0.05  # Scale down significantly
        
        # Gentle clipping to prevent any outliers
        reward = np.clip(reward, -1.0, 1.5)
        
        return reward
    
    def _calculate_hp_progress(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Calculate HP advantage progress per turn (STABLE version)
        
        Uses simple difference instead of ratio to avoid division instability.
        Returns small per-turn values that accumulate over episode.
        
        Returns: ±0.10 per turn (accumulates to ±5.0 over 50 turns)
        """
        if not prior_battle:
            return 0.0
        
        # Calculate total HP for each team (0-6 range)
        current_our_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) for mon in battle.team.values())
        current_opp_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) for mon in battle.opponent_team.values())
        
        prior_our_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) for mon in prior_battle.team.values())
        prior_opp_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) for mon in prior_battle.opponent_team.values())
        
        # Calculate HP advantage (difference, not ratio)
        # Range: -6 (all ours dead) to +6 (all theirs dead)
        current_advantage = current_our_hp - current_opp_hp
        prior_advantage = prior_our_hp - prior_opp_hp
        
        # Progress in advantage
        advantage_progress = current_advantage - prior_advantage
        
        # Scale to ±0.10 per turn
        # Typical turn: ~0.15 to 0.20 HP dealt/taken = ~0.02 reward
        # Major turn: ~1.0 HP (KO) = ~0.10 reward
        return np.clip(advantage_progress * 0.8, -0.10, 0.10)
    
    def _calculate_battle_outcome(self, battle: AbstractBattle) -> float:
        """
        Calculate win/loss signal
        
        Returns: +1.0 for win, -1.0 for loss, 0.0 otherwise
        (Actual reward is +15.0/-10.0, applied in calc_reward)
        """
        
        if getattr(battle, 'won', False):
            return 1.0
        elif getattr(battle, 'lost', False):
            return -1.0
        else:
            return 0.0
    
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
        """
        Evaluate if the move choice made sense in context.
        
        Returns ±0.10 per turn (accumulates to ±5.0 over 50 turns).
        Provides feedback for attacking move timing and penalizes hazard spam.
        """
        reward = 0.0
        
        if not battle.active_pokemon or not prior_battle.active_pokemon:
            return 0.0
            
        # Get battle context
        my_hp = getattr(battle.active_pokemon, 'current_hp_fraction', 1.0)
        opp_hp = getattr(battle.opponent_active_pokemon, 'current_hp_fraction', 1.0) if battle.opponent_active_pokemon else 1.0
        
        # Analyze the move that was used
        available_moves = getattr(prior_battle, 'available_moves', [])
        if available_moves:
            action_value = last_action.get('action', 6)
            move_index = action_value - 6
            
            if 0 <= move_index < len(available_moves):
                try:
                    move = available_moves[move_index]
                    move_category = getattr(move, 'category', None)
                    move_id = getattr(move, 'id', '')
                    
                    # === PENALIZE ENTRY HAZARD SPAM ===
                    # Penalize using entry hazards when they're already active
                    if move_id and hasattr(battle, 'opponent_side_conditions') and battle.opponent_side_conditions:
                        opp_conditions = battle.opponent_side_conditions
                        
                        # Convert side conditions to strings for checking
                        # side_conditions keys are SideCondition enums, need to check via iteration
                        condition_names = set()
                        for condition in opp_conditions.keys():
                            condition_name = str(condition).lower().replace('sidecondition.', '')
                            condition_names.add(condition_name)
                        
                        # Check for redundant hazard usage
                        if move_id == 'stealthrock' and 'stealthrock' in condition_names:
                            reward -= 0.5  # Already have rocks
                        elif move_id == 'spikes':
                            # Check spikes count by iterating
                            for condition, count in opp_conditions.items():
                                if 'spikes' in str(condition).lower() and count >= 3:
                                    reward -= 0.5  # Already at max layers
                                    break
                        elif move_id == 'toxicspikes':
                            # Check toxic spikes count
                            for condition, count in opp_conditions.items():
                                if 'toxicspikes' in str(condition).lower() and count >= 2:
                                    reward -= 0.5  # Already at max layers
                                    break
                        elif move_id == 'stickyweb' and 'stickyweb' in condition_names:
                            reward -= 0.08  # Already have sticky web
                    
                    # === ONLY REWARD ATTACKING MOVES ===
                    # No rewards for setup/status moves - let agent learn those naturally
                    
                    if move_category in [MoveCategory.PHYSICAL, MoveCategory.SPECIAL]:
                        # Reward attacking when opponent is low HP (finishing)
                        if opp_hp < 0.3:
                            reward += 0.05
                        # Penalize attacking when you're low and opponent is healthy (bad trade)
                        elif my_hp < 0.3 and opp_hp > 0.7:
                            if not self._can_ko_opponent(move, battle):
                                reward -= 0.05
                            
                except (IndexError, AttributeError):
                    pass
        
        # Clip to ±0.10 per turn
        return np.clip(reward, -1.0, 1.0)
    
    def _evaluate_switch_context(self, battle: AbstractBattle, prior_battle: AbstractBattle, last_action: dict) -> float:
        """
        DECOMPOSED SWITCHING REWARDS: Direct feedback for matchup changes.
        
        Rewards are based on CURRENT active Pokemon matchup (from cache):
        - our_offensive: Current active Pokemon's type advantage
        - their_threat_to_us: Current threat opponent poses to us
        - revealed_opponents: How many opponent Pokemon we've seen
        
        These are compared to PRIOR active Pokemon to calculate improvement.
        
        State features the agent sees:
        - [57-62]: Per-Pokemon offensive matchups (one per position)
        - [63-68]: Per-Pokemon defensive matchups (one per position)
        - [69-74]: Per-Pokemon HP (one per position)
        - [75]: Revealed opponents count
        - [76]: Total opponents count
        
        Agent learns: "Position X has good matchup → switch to X → reward"
        """
        if not prior_battle.active_pokemon or not battle.active_pokemon:
            return 0.0
        
        # Get cached matchup values (synchronized with state)
        current_threat = self.matchup_cache['their_threat_to_us']
        current_offensive = self.matchup_cache['our_offensive']
        revealed_opponents = self.matchup_cache['revealed_opponents']
        total_opponents = self.matchup_cache['total_opponents']
        
        # Calculate PRIOR matchup (before switch)
        prior_active = prior_battle.active_pokemon
        opponent_active = battle.opponent_active_pokemon
        
        if not opponent_active:
            return 0.0
        
        # Prior matchup calculations
        prior_hp = getattr(prior_active, 'current_hp_fraction', 1.0)
        current_hp = getattr(battle.active_pokemon, 'current_hp_fraction', 1.0)
        prior_threat = self.calculate_threat_level(opponent_active, prior_active)
        
        prior_offensive = 0.0
        if hasattr(prior_active, 'types') and prior_active.types:
            for poke_type in prior_active.types:
                if poke_type and hasattr(poke_type, 'name'):
                    type_name = poke_type.name.title()
                    if opponent_active.types:
                        opp_types = tuple(t.name.title() for t in opponent_active.types 
                                        if t and hasattr(t, 'name'))
                        if opp_types:
                            effectiveness = self.move_effectiveness(type_name, opp_types)
                            prior_offensive = max(prior_offensive, effectiveness)
        
        # ===== COMPONENT 1: OFFENSIVE MATCHUP IMPROVEMENT =====
        # Reward switching to Pokemon with better type advantage
        # Agent sees per-Pokemon offensive values in state [57-62]
        
        offensive_delta = current_offensive - prior_offensive
        offensive_reward = np.clip(offensive_delta / 6.0, -0.08, 0.08)  # ±0.08 per turn
        
        # ===== COMPONENT 2: DEFENSIVE THREAT REDUCTION =====
        # Reward switching away from bad defensive matchups
        # Agent sees per-Pokemon defensive values in state [63-68]
        
        defensive_delta = prior_threat - current_threat  # Reduction is good
        
        # Context: Emergency switches get bonus (high threat + low HP)
        if prior_threat >= 2.0 and prior_hp < 0.5:
            # Emergency escape - reward ANY threat reduction strongly
            defensive_reward = np.clip(defensive_delta / 3.0, -0.15, 0.15)
        elif prior_threat >= 2.0 and prior_hp >= 0.5:
            # Tactical switch from bad matchup - good reward
            defensive_reward = np.clip(defensive_delta / 5.0, -0.10, 0.10)
        else:
            # Low threat - still allow some reward for optimization
            defensive_reward = np.clip(defensive_delta / 8.0, -0.05, 0.05)
        
        # ===== COMPONENT 3: INFORMATION GATHERING =====
        # Reward exploring when safe, penalize blind switches in danger
        # Agent sees revealed_opponents [75] and total_opponents [76] in state
        
        info_reward = 0.0
        
        # Reward information gathering when safe (low threat, less than half team revealed)
        if prior_threat < 1.0 and revealed_opponents < 3:
            # Safe situation, low knowledge - exploring is valuable
            info_reward = 0.02
        
        # Penalize risky switches with low information (less than 2 revealed)
        elif prior_threat >= 2.0 and revealed_opponents < 2:
            # High threat, low knowledge - switching to unknown is risky
            if prior_hp >= 0.5:  # Not emergency
                info_reward = -0.02
        
        # Reward using information when confident (most team revealed)
        elif revealed_opponents >= 5:
            # High knowledge - good switches should be rewarded more
            if offensive_delta > 0.5 or defensive_delta > 0.5:
                info_reward = 0.02  # Bonus for informed good decision
        
        # ===== TOTAL REWARD: SUM OF SEPARABLE COMPONENTS =====
        # Maximum per turn: 0.08 + 0.15 + 0.02 = 0.25
        # Good switches can now compete with HP progress rewards
        total_reward = offensive_reward + defensive_reward + info_reward
        
        # Clip to ±0.25 per turn (higher to encourage good switches)
        return np.clip(total_reward, -0.25, 0.25)
    
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

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        ENHANCED State Space with Per-Pokemon Switch Intelligence:
        
        Core Battle State: 35 features
        - 12 (team HP: 6 ours + 6 opponent)
        - 2 (fainted counts: ours + opponent) 
        - 4 (active Pokemon stats: ATK, DEF, SPA, SPD relative to opponent)
        - 2 (status conditions: ours + opponent encoded)
        - 8 (ALL stat boosts: ATK, DEF, SPA, SPD for both Pokemon)
        - 4 (speed info: speed stat ours+opp, speed boosts ours+opp)
        - 3 (type advantages: our move effectiveness, our type threat, their threat)
        
        Move Information: 16 features (4 moves × 4 features)
        - Base power (normalized)
        - Type effectiveness vs opponent
        - Estimated damage
        - Move category (physical/special/status)
        
        Strategic Context: 6 features
        - Turn number (normalized)
        - Available switches
        - Weather (encoded as single value 0-4)
        - Entry hazards (simplified advantage)
        - HP ratio (our_total_hp / opponent_total_hp)
        - Faint advantage ((6-our_fainted) / (6-opp_fainted))
        
        Per-Pokemon Switch Information: 18 features [NEW]
        - 6 offensive matchups (type advantage each Pokemon has vs opponent)
        - 6 defensive matchups (threat opponent poses to each Pokemon)
        - 6 HP fractions (HP of each Pokemon)
        This allows agent to learn: "position 2 has good defensive matchup → switch to 2"
        
        Global Switch Context: 2 features
        - Revealed opponents count (0-6)
        - Total opponents count (typically 6)
        
        Total: 35 + 16 + 6 + 18 + 2 = 77 features
        
        REMOVED (not essential for core strategy):
        - Best switch aggregates (replaced by per-Pokemon data)
        - Improvement potentials (agent can calculate from per-Pokemon data)
        - Action history (temporal patterns)
        - Stat boost efficiency
        - Action patterns
        - Game phase indicators (early/mid/late)
        - Individual hazard types (simplified to binary)
        - Ability/item indicators (learned implicitly)
        - Individual stat values (use relative comparisons)
        - Weather one-hot (use single encoded value)

        Returns:
            int: The size of the observation space.
        """
        return 77  # Per-Pokemon switch information for direct action mapping

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
        """FIXED: Check if move is nullified by defender's ability"""
        if not move or not defender:
            return False
            
        defender_ability = getattr(defender, 'ability', None)
        if not defender_ability:
            return False

        ability = str(defender_ability).lower().replace(" ", "")
        move_type_obj = getattr(move, 'type', None)

        move_type = getattr(move_type_obj, 'name', '').upper() if move_type_obj else ""
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
    
    def get_opponent_ability_indicators(self, battle: AbstractBattle) -> tuple:
        """
        NEW: Returns (has_immunity_ability, has_boosting_ability, has_weather_ability)
        CRITICAL for state-reward alignment - agent must SEE immunity abilities
        """
        if not battle.opponent_active_pokemon:
            return (0.0, 0.0, 0.0)
        
        ability = getattr(battle.opponent_active_pokemon, 'ability', None)
        if not ability:
            return (0.0, 0.0, 0.0)
        
        ability_str = str(ability).lower().replace(" ", "")
        
        # Immunity abilities (CRITICAL for learning)
        immunity_abilities = {
            'voltabsorb', 'motordrive', 'lightningrod',  # Electric immunity
            'waterabsorb', 'stormdrain', 'dryskin',      # Water immunity
            'flashfire',                                  # Fire immunity
            'sapsipper',                                  # Grass immunity
            'levitate',                                   # Ground immunity
            'bulletproof', 'wonderguard', 'thickfat',
        }
        has_immunity = 1.0 if ability_str in immunity_abilities else 0.0
        
        # Stat-boosting abilities
        boosting_abilities = {
            'intimidate', 'defiant', 'competitive', 'contrary',
            'hugepower', 'purepower', 'adaptability', 'technician',
        }
        has_boosting = 1.0 if ability_str in boosting_abilities else 0.0
        
        # Weather abilities
        weather_abilities = {
            'drizzle', 'drought', 'sandstream', 'snowwarning',
        }
        has_weather = 1.0 if ability_str in weather_abilities else 0.0
        
        return (has_immunity, has_boosting, has_weather)
    
    def get_opponent_item_indicators(self, battle: AbstractBattle) -> tuple:
        """
        NEW: Returns (has_hp_item, has_power_item, has_defensive_item)
        Helps agent understand opponent's strategy
        """
        if not battle.opponent_active_pokemon:
            return (0.0, 0.0, 0.0)
        
        item = getattr(battle.opponent_active_pokemon, 'item', None)
        if not item:
            return (0.0, 0.0, 0.0)
        
        item_str = str(item).lower().replace(" ", "")
        
        hp_items = {'leftovers', 'blacksludge', 'sitrusberry', 'oranberry'}
        has_hp_item = 1.0 if item_str in hp_items else 0.0
        
        power_items = {'lifeorb', 'choiceband', 'choicespecs', 'choicescarf', 'expertbelt'}
        has_power_item = 1.0 if item_str in power_items else 0.0
        
        defensive_items = {'assaultvest', 'rockyhelmet', 'weaknesspolicy', 'focussash'}
        has_defensive_item = 1.0 if item_str in defensive_items else 0.0
        
        return (has_hp_item, has_power_item, has_defensive_item)
    
    def _get_move_accuracy(self, move: Move) -> float:
        """
        NEW: Get move accuracy (helps learn Thunder=70% vs Thunderbolt=100%)
        """
        if not move:
            return 100.0
        
        try:
            accuracy = getattr(move, 'accuracy', 100)
            # Handle True (always hits) as 100%
            if accuracy is True:
                return 100.0
            # Handle numeric accuracy
            return float(accuracy) if accuracy else 100.0
        except (AttributeError, TypeError, ValueError):
            return 100.0
    
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
    
    def calculate_threat_level(self, attacker: Pokemon, defender: Pokemon) -> float:
        """
        Calculate comprehensive threat level from attacker to defender.
        Returns: threat multiplier (0.25 = resisted, 1.0 = neutral, 4.0 = super effective)
        """
        if not attacker or not defender:
            return 1.0
        
        max_threat = 1.0
        
        # Get defender's types
        if not hasattr(defender, 'types') or not defender.types:
            return 1.0
        
        defender_types = tuple(t.name.title() for t in defender.types if t and hasattr(t, 'name'))
        if not defender_types:
            return 1.0
        
        # Check attacker's types (STAB moves are most likely)
        if hasattr(attacker, 'types') and attacker.types:
            for atk_type in attacker.types:
                if atk_type and hasattr(atk_type, 'name'):
                    type_name = atk_type.name.title()
                    effectiveness = self.move_effectiveness(type_name, defender_types)
                    max_threat = max(max_threat, effectiveness)
        
        return max_threat
    
    def get_pokemon_switch_matchups(self, battle: AbstractBattle) -> Tuple[list, list, list]:
        """
        Calculate matchup quality for EACH Pokemon position (0-5).
        This allows agent to learn "switch to position X" → "good matchup" association.
        
        Returns: (offensive_matchups[6], defensive_matchups[6], hp_fractions[6])
        - offensive_matchups[i]: Type advantage Pokemon i has vs current opponent (0-4 scale)
        - defensive_matchups[i]: Threat opponent poses to Pokemon i (0-4 scale, lower=better)
        - hp_fractions[i]: HP of Pokemon i (0-1 scale)
        
        Invalid switches (fainted/active/empty) return (1.0, 1.0, 0.0):
        - Fainted Pokemon: HP=0.0 signals "can't switch"
        - Currently active: HP=0.0 signals "already out"
        - Empty slots: HP=0.0 signals "doesn't exist"
        
        Unknown/unrevealed Pokemon return (1.0, 1.0, actual_HP):
        - Matchups unknown (neutral 1.0) until revealed
        - HP is real → agent CAN switch to explore
        - Agent learns: "switching to unknown is risky but possible"
        """
        if not battle.opponent_active_pokemon:
            return [1.0]*6, [1.0]*6, [0.0]*6
        
        battle_team = getattr(battle, 'team', {})
        battle_active = getattr(battle, 'active_pokemon', None)
        
        # Get team as ordered list (positions 0-5)
        team_list = list(battle_team.values())
        if len(team_list) < 6:
            team_list.extend([None] * (6 - len(team_list)))
        
        offensive_matchups = []
        defensive_matchups = []
        hp_fractions = []
        
        for i, pokemon in enumerate(team_list[:6]):
            if pokemon is None or getattr(pokemon, 'fainted', True) or pokemon == battle_active:
                offensive_matchups.append(1.0)  # Neutral (no advantage)
                defensive_matchups.append(1.0)  # Neutral (no threat)
                hp_fractions.append(0.0)        # No HP (can't switch)
                continue
            
            # Handle unrevealed Pokemon (valid switch, but unknown matchup)
            if not hasattr(pokemon, 'species') or not pokemon.species:
                # Unknown matchup, but we know HP - agent can still switch!
                offensive_matchups.append(1.0)  # Unknown matchup = neutral assumption
                defensive_matchups.append(1.0)  # Unknown matchup = neutral assumption
                hp = getattr(pokemon, 'current_hp_fraction', 1.0)  # Use actual HP if available
                hp_fractions.append(hp)
                continue
            
            # Calculate offensive matchup (type advantage)
            offensive = 0.0
            if hasattr(pokemon, 'types') and pokemon.types:
                for poke_type in pokemon.types:
                    if poke_type and hasattr(poke_type, 'name'):
                        type_name = poke_type.name.title()
                        if battle.opponent_active_pokemon.types:
                            opp_types = tuple(t.name.title() for t in battle.opponent_active_pokemon.types 
                                            if t and hasattr(t, 'name'))
                            if opp_types:
                                effectiveness = self.move_effectiveness(type_name, opp_types)
                                offensive = max(offensive, effectiveness)
            
            # Calculate defensive matchup (threat from opponent)
            defensive = self.calculate_threat_level(battle.opponent_active_pokemon, pokemon)
            
            # Get HP
            hp = getattr(pokemon, 'current_hp_fraction', 0.0)
            
            offensive_matchups.append(offensive)
            defensive_matchups.append(defensive)
            hp_fractions.append(hp)
        
        return offensive_matchups, defensive_matchups, hp_fractions
    
    def get_best_switch_matchup(self, battle: AbstractBattle) -> Tuple[float, float, float, int, int]:
        """
        Find the best available switch option against current opponent.
        Also tracks information visibility of opponent team.
        
        Returns: (best_offensive_matchup, best_defensive_matchup, best_hp, 
                  revealed_opponents, total_opponents)
        """
        if not battle.opponent_active_pokemon:
            return 1.0, 1.0, 0.0, 0, 6
        
        battle_team = getattr(battle, 'team', {})
        battle_active = getattr(battle, 'active_pokemon', None)
        battle_opponent_team = getattr(battle, 'opponent_team', {})
        
        best_offensive = 0.0
        best_defensive = 4.0  # Lower is better for defense (we want to resist)
        best_hp = 0.0
        
        # Count revealed opponent Pokemon (have species information)
        revealed_opponents = 0
        total_opponents = 0
        for opp_pokemon in battle_opponent_team.values():
            total_opponents += 1
            # Pokemon is "revealed" if we know its species
            if hasattr(opp_pokemon, 'species') and opp_pokemon.species:
                revealed_opponents += 1
        
        for pokemon in battle_team.values():
            # Skip fainted and currently active Pokemon
            if getattr(pokemon, 'fainted', True) or pokemon == battle_active:
                continue
            
            # Offensive matchup: What's the best move this Pokemon could use?
            if hasattr(pokemon, 'types') and pokemon.types:
                for poke_type in pokemon.types:
                    if poke_type and hasattr(poke_type, 'name'):
                        type_name = poke_type.name.title()
                        if battle.opponent_active_pokemon.types:
                            opp_types = tuple(t.name.title() for t in battle.opponent_active_pokemon.types 
                                            if t and hasattr(t, 'name'))
                            if opp_types:
                                effectiveness = self.move_effectiveness(type_name, opp_types)
                                best_offensive = max(best_offensive, effectiveness)
            
            # Defensive matchup: How well does this Pokemon resist opponent?
            threat = self.calculate_threat_level(battle.opponent_active_pokemon, pokemon)
            best_defensive = min(best_defensive, threat)
            
            # HP of best switch option
            hp = getattr(pokemon, 'current_hp_fraction', 0.0)
            best_hp = max(best_hp, hp)
        
        return best_offensive, best_defensive, best_hp, revealed_opponents, total_opponents
    
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
        
        Philosophy: Focus on what the agent can CONTROL and what affects WIN CONDITIONS.
        Remove temporal patterns, detailed tracking, and redundant information.
        
        64 features total (expanded from 62):
        - 35 core battle state
        - 16 move information  
        - 6 strategic context
        - 7 switch matchup information (includes 2 visibility features)
        
        Args:
            battle (AbstractBattle): The current battle instance
        Returns:
            np.float32: A 1D numpy array of 64 features
        """
        state = []
        
        # Initialize variables that will be used later in switch matchup calculation
        best_effectiveness = 0.0
        our_threat_to_opponent = 1.0
        threat_level = 1.0

        # ============================================================
        # CORE BATTLE STATE (30 features)
        # ============================================================
        
        # Team HP fractions (12 features: 6 ours + 6 opponent)
        battle_team = getattr(battle, 'team', {})
        battle_opponent_team = getattr(battle, 'opponent_team', {})
        
        health_team = [getattr(mon, 'current_hp_fraction', 0.0) for mon in battle_team.values()]
        health_opponent = [getattr(mon, 'current_hp_fraction', 0.0) for mon in battle_opponent_team.values()]
        
        # Ensure exactly 6 components
        if len(health_opponent) < 6:
            health_opponent.extend([0.0] * (6 - len(health_opponent)))
        if len(health_team) < 6:
            health_team.extend([0.0] * (6 - len(health_team)))
            
        state.extend(health_team[:6])
        state.extend(health_opponent[:6])
        
        # Fainted counts (2 features)
        fainted_team = sum(1 for mon in battle_team.values() if getattr(mon, 'fainted', False))
        fainted_opponent = sum(1 for mon in battle_opponent_team.values() if getattr(mon, 'fainted', False))
        state.extend([fainted_team / 6.0, fainted_opponent / 6.0])
        
        # Active Pokemon Info (16 features)
        if battle.active_pokemon and battle.opponent_active_pokemon:
            active = battle.active_pokemon
            opp_active = battle.opponent_active_pokemon
            
            # Relative stats (4 features: compare our stats to opponent's)
            our_atk = self.calculate_stat(active, 'atk')
            our_def = self.calculate_stat(active, 'def')
            our_spa = self.calculate_stat(active, 'spa')
            our_spd = self.calculate_stat(active, 'spd')
            
            opp_atk = self.calculate_stat(opp_active, 'atk')
            opp_def = self.calculate_stat(opp_active, 'def')
            opp_spa = self.calculate_stat(opp_active, 'spa')
            opp_spd = self.calculate_stat(opp_active, 'spd')
            
            # Ratios (better than absolute values - scale invariant)
            state.extend([
                our_atk / max(1.0, opp_def),   # Physical damage potential
                our_def / max(1.0, opp_atk),   # Physical defense ratio
                our_spa / max(1.0, opp_spd),   # Special damage potential
                our_spd / max(1.0, opp_spa),   # Special defense ratio
            ])
            
            # Status conditions (2 features: encoded)
            status_map = {Status.BRN: 1.0, Status.FRZ: 2.0, Status.PAR: 3.0, 
                         Status.PSN: 4.0, Status.SLP: 5.0, Status.TOX: 6.0}
            active_status = getattr(active, 'status', None)
            opp_status = getattr(opp_active, 'status', None)
            active_status_val = status_map.get(active_status, 0.0) / 6.0 if active_status else 0.0
            opp_status_val = status_map.get(opp_status, 0.0) / 6.0 if opp_status else 0.0
            state.extend([active_status_val, opp_status_val])
            
            # ALL stat boosts (8 features: offense + defense for both Pokemon)
            active_boosts = getattr(active, 'boosts', {})
            opp_boosts = getattr(opp_active, 'boosts', {})
            state.extend([
                active_boosts.get('atk', 0) / 6.0,
                active_boosts.get('def', 0) / 6.0,
                active_boosts.get('spa', 0) / 6.0,
                active_boosts.get('spd', 0) / 6.0,
                opp_boosts.get('atk', 0) / 6.0,
                opp_boosts.get('def', 0) / 6.0,
                opp_boosts.get('spa', 0) / 6.0,
                opp_boosts.get('spd', 0) / 6.0,
            ])
            
            # Speed comparison (4 features: raw speed + boosts)
            our_spe = self.calculate_stat(active, 'spe')
            opp_spe = self.calculate_stat(opp_active, 'spe')
            state.extend([
                our_spe / 500.0,  # Normalized speed
                opp_spe / 500.0,
                active_boosts.get('spe', 0) / 6.0,
                opp_boosts.get('spe', 0) / 6.0,
            ])
            
            # Type advantage summary (3 features) - EXPANDED FROM 2
            # Our best move effectiveness against opponent
            best_effectiveness = 0.0
            available_moves = getattr(battle, 'available_moves', [])
            for move in available_moves:
                if move.type and opp_active.types:
                    opp_types = tuple(t.name.title() for t in opp_active.types if t and hasattr(t, 'name'))
                    if opp_types:
                        eff = self.move_effectiveness(move.type.name.title(), opp_types)
                        best_effectiveness = max(best_effectiveness, eff)
            
            # Our threat to opponent (how threatened should opponent be by us)
            our_threat_to_opponent = self.calculate_threat_level(active, opp_active)
            
            # Opponent's threat to us using comprehensive calculation
            threat_level = self.calculate_threat_level(opp_active, active)
            
            # CACHE for reward synchronization
            self.matchup_cache['our_offensive'] = best_effectiveness
            self.matchup_cache['our_threat_to_them'] = our_threat_to_opponent
            self.matchup_cache['their_threat_to_us'] = threat_level
            
            state.extend([
                best_effectiveness / 4.0,      # Our best move effectiveness
                our_threat_to_opponent / 4.0,  # Our type threat to them [NEW]
                threat_level / 4.0              # Their threat to us
            ])
            
        else:
            # No active Pokemon - fill with neutral values
            state.extend([1.0] * 4)  # Neutral stat ratios
            state.extend([0.0] * 2)  # No status
            state.extend([0.0] * 8)  # No boosts
            state.extend([0.5, 0.5, 0.0, 0.0])  # Neutral speed
            state.extend([0.25, 0.25, 0.25])  # Neutral type matchups (3 features)

        # ============================================================
        # MOVE INFORMATION (16 features: 4 moves × 4 features)
        # ============================================================
        
        available_moves = getattr(battle, 'available_moves', [])
        for i in range(4):
            if i < len(available_moves) and battle.opponent_active_pokemon:
                move = available_moves[i]
                
                # Type effectiveness
                try:
                    if move.type and battle.opponent_active_pokemon.types:
                        opp_types = tuple(t.name.title() for t in battle.opponent_active_pokemon.types 
                                        if t and hasattr(t, 'name'))
                        effectiveness = self.move_effectiveness(move.type.name.title(), opp_types) if opp_types else 1.0
                    else:
                        effectiveness = 1.0
                except (AttributeError, TypeError):
                    effectiveness = 1.0
                
                # Estimated damage
                damage = self.estimate_move_damage(move, battle.active_pokemon, 
                                                  battle.opponent_active_pokemon, battle)
                
                # Base power
                base_power = getattr(move, 'base_power', 0) or 0
                
                # Move category (simplified: 0=status, 0.5=physical, 1.0=special)
                move_category = getattr(move, 'category', None)
                if move_category == MoveCategory.STATUS:
                    category_value = 0.0
                elif move_category == MoveCategory.PHYSICAL:
                    category_value = 0.5
                else:  # SPECIAL
                    category_value = 1.0
                
                state.extend([
                    base_power / 150.0,
                    effectiveness / 4.0,
                    np.clip(damage, 0.0, 2.0),
                    category_value
                ])
            else:
                state.extend([0.0] * 4)
        
        
        # Turn number (1 feature)
        battle_turn = getattr(battle, 'turn', 0)
        state.append(min(battle_turn / 50.0, 1.0))
        
        # Available switches (1 feature)
        battle_active = getattr(battle, 'active_pokemon', None)
        available_switches = len([mon for mon in battle_team.values() 
                                if not getattr(mon, 'fainted', True) and mon != battle_active])
        state.append(available_switches / 5.0)
        
        # Weather (1 feature: encoded as single value)
        weather_value = 0.0
        battle_weather = getattr(battle, 'weather', None)
        if battle_weather:
            weather_str = str(battle_weather).lower()
            if 'sun' in weather_str:
                weather_value = 0.25
            elif 'rain' in weather_str:
                weather_value = 0.50
            elif 'sand' in weather_str:
                weather_value = 0.75
            elif 'hail' in weather_str or 'snow' in weather_str:
                weather_value = 1.0
        state.append(weather_value)
        
        # Entry hazards (1 feature: simplified advantage)
        # +1 if we have hazards and they don't, -1 if reverse, 0 if equal
        our_has_hazards = 0.0
        opp_has_hazards = 0.0
        
        if hasattr(battle, 'side_conditions') and battle.side_conditions:
            our_has_hazards = 1.0
        if hasattr(battle, 'opponent_side_conditions') and battle.opponent_side_conditions:
            opp_has_hazards = 1.0
        
        hazard_advantage = (opp_has_hazards - our_has_hazards + 1.0) / 2.0  # Normalize to [0, 1]
        state.append(hazard_advantage)
        
        # HP ratio (1 feature: our_total_hp / opponent_total_hp)
        our_total_hp = sum(health_team)
        opp_total_hp = sum(health_opponent)
        hp_ratio = our_total_hp / max(0.1, opp_total_hp)
        state.append(np.clip(np.log(hp_ratio + 0.01) / 2.0 + 0.5, 0.0, 1.0))  # Log scale, normalized
        
        # Pokemon advantage (1 feature: remaining Pokemon ratio)
        our_remaining = 6 - fainted_team
        opp_remaining = 6 - fainted_opponent
        pokemon_advantage = our_remaining / max(1.0, opp_remaining)
        state.append(np.clip(pokemon_advantage / 2.0, 0.0, 1.0))  # Normalize to [0, 1]
        
        # ============================================================
        # PER-POKEMON SWITCH INFORMATION (18 features) [NEW]
        # ============================================================
        # Provide matchup quality for EACH Pokemon position (0-5)
        # Agent learns: "position 2 has defensive=0.5 → switch to 2 → good outcome"
        
        (offensive_matchups, defensive_matchups, hp_fractions) = self.get_pokemon_switch_matchups(battle)
        
        # Offensive matchups for each position (6 features)
        # Shows type advantage each Pokemon would have vs current opponent
        for offensive in offensive_matchups:
            state.append(offensive / 4.0)  # Normalize 4x = 1.0
        
        # Defensive matchups for each position (6 features)
        # Shows threat opponent poses to each Pokemon (lower = better)
        for defensive in defensive_matchups:
            state.append(defensive / 4.0)  # Normalize 4x = 1.0
        
        # HP for each position (6 features)
        # Shows which Pokemon are healthy enough to switch in
        state.extend(hp_fractions)
        
        # ============================================================
        # GLOBAL SWITCH CONTEXT (2 features)
        # ============================================================
        # Opponent team visibility tracking
        
        # Get opponent visibility (reuse from best_switch calculation for cache)
        (_, _, _, revealed_opponents, total_opponents) = self.get_best_switch_matchup(battle)
        
        # CACHE for reward synchronization
        self.matchup_cache['revealed_opponents'] = revealed_opponents
        self.matchup_cache['total_opponents'] = total_opponents
        
        # Direct counts - let agent learn the relationship
        state.append(revealed_opponents / 6.0)  # How many opponents revealed (0-6)
        state.append(total_opponents / 6.0)      # How many total opponents (typically 6)
        
        # ============================================================
        # VALIDATION
        # ============================================================
        
        state_array = np.array(state, dtype=np.float32)
        
        # Ensure correct size
        assert len(state_array) == 77, f"State size mismatch! Expected 77, got {len(state_array)}"
        
        # Check for NaN/Inf
        if np.isnan(state_array).any() or np.isinf(state_array).any():
            # Replace NaN/Inf with safe values
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return state_array

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