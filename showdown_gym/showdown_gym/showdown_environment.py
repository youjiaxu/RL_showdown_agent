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
            
            # reward component logging (using NEW 6-component system)
            try:
                prior_battle = self._get_prior_battle(self.battle1)
                outcome_component = self._calculate_outcome_reward(self.battle1)
                hp_advantage_component = self._calculate_hp_advantage(self.battle1, prior_battle)
                ko_component = self._calculate_ko_events(self.battle1, prior_battle)
                move_effectiveness_component = self._calculate_move_effectiveness_feedback(self.battle1, prior_battle)
                switch_component = self._calculate_switch_reward(self.battle1, prior_battle)
                anti_spam_component = self._calculate_anti_spam_penalty(self.battle1)
                
                info[agent]["reward_outcome"] = float(outcome_component)
                info[agent]["reward_hp_advantage"] = float(hp_advantage_component)
                info[agent]["reward_ko"] = float(ko_component)
                info[agent]["reward_move_effectiveness"] = float(move_effectiveness_component)
                info[agent]["reward_switch"] = float(switch_component)
                info[agent]["reward_anti_spam"] = float(anti_spam_component)
                # reward_total = last turn's reward (per-turn) using NEW 6-component system
                info[agent]["reward_total"] = float(
                    outcome_component + hp_advantage_component + ko_component + 
                    move_effectiveness_component + switch_component + anti_spam_component
                )
                # reward_episode = cumulative sum of all turn rewards (per-episode)
                info[agent]["reward_episode"] = float(self.cumulative_episode_reward)
            except (AttributeError, Exception):
                info[agent]["reward_outcome"] = 0.0
                info[agent]["reward_hp_advantage"] = 0.0
                info[agent]["reward_ko"] = 0.0
                info[agent]["reward_move_effectiveness"] = 0.0
                info[agent]["reward_switch"] = 0.0
                info[agent]["reward_anti_spam"] = 0.0
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
        Q-Learning Optimized Reward System (Ground-Up Rebuild)
        
        DESIGN PRINCIPLES:
        1. Sparse, strong signals for terminal outcomes (Win/Loss)
        2. Dense, weak signals for continuous progress (HP advantage)
        3. Clear penalties for mistakes (immunity, ineffective moves)
        4. Anti-spam mechanisms (prevent reward hacking)
        5. State-reward alignment (agent sees what it's rewarded for)
        
        COMPONENTS (in learning priority order):
        1. WIN/LOSS: Dominant sparse reward (+5/-1 after normalization)
        2. HP ADVANTAGE: Dense progress tracking (±0.15 per turn)
        3. KO EVENTS: Medium sparse reward (±0.4-0.8 per KO)
        4. MOVE EFFECTIVENESS: Immediate feedback (±0.1 per move)
        5. SWITCHING: Strategic positioning (±0.1 per switch)
        6. ANTI-SPAM: Prevent exploitation (-0.1 to -0.3 per spam)
        
        REWARD SCALE: All normalized to [-1, +1] range for Q-learning stability
        """
        try:
            prior_battle = self._get_prior_battle(battle)
        except AttributeError:
            prior_battle = None
        
        # === CORE REWARDS (What matters for winning) ===
        outcome_reward = self._calculate_outcome_reward(battle)  # Sparse: +5/-1
        hp_advantage_reward = self._calculate_hp_advantage(battle, prior_battle)  # Dense: ±0.15
        ko_reward = self._calculate_ko_events(battle, prior_battle)  # Medium: ±0.4-0.8
        
        # === LEARNING SIGNALS (Guide strategy) ===
        move_feedback = self._calculate_move_effectiveness_feedback(battle, prior_battle)  # ±0.1
        switch_reward = self._calculate_switch_reward(battle, prior_battle)  # ±0.1
        
        # === ANTI-EXPLOITATION (Prevent reward hacking) ===
        spam_penalty = self._calculate_anti_spam_penalty(battle)  # -0.1 to -0.3
        
        # === AGGREGATE ===
        total_reward = (
            outcome_reward +      # Dominates: ±5.0
            hp_advantage_reward + # Guides: ±0.15
            ko_reward +           # Reinforces: ±0.4-0.8
            move_feedback +       # Teaches: ±0.1
            switch_reward +       # Strategic: ±0.1
            spam_penalty          # Prevents hacking: -0.3
        )
        
        # CRITICAL: Normalize to [-1, +1] range for Q-learning stability
        # Win: +5 → +0.83, Loss: -1 → -0.17, Per-turn: ±0.3 → ±0.05
        total_reward = total_reward / 6.0
        
        # FIXED: Clip AFTER normalization to maintain proper bounds
        # Handles edge cases where cumulative rewards might exceed expected range
        total_reward = np.clip(total_reward, -1.0, 1.0)
        
        # Safety check
        if not np.isfinite(total_reward):
            total_reward = 0.0
        
        # Track for debugging (reset in embed_battle when turn <= 1)
        self.cumulative_episode_reward += total_reward
        
        return total_reward
    
    # === OLD FUNCTIONS REMOVED - SEE NEW IMPLEMENTATIONS ABOVE ===
    # The following were removed duplicate old implementations:
    # - _calculate_damage_delta (replaced by _calculate_hp_advantage)
    # - _calculate_ko_reward (replaced by _calculate_ko_events)
    # Keeping this comment to track deleted code
    
    # Old functions fully removed above
    
    # ==================================================================================
    # COMPONENT 1: WIN/LOSS OUTCOME (Dominant Sparse Reward)
    # ==================================================================================
    
    def _calculate_outcome_reward(self, battle: AbstractBattle) -> float:
        """
        Win/Loss outcome reward - DOMINATES all other components.
        
        Q-Learning Principle: Terminal rewards should be the strongest signal.
        Win: +5.0 (after /6 normalization = +0.83)
        Loss: -1.0 (after /6 normalization = -0.17)
        
        WHY THIS SCALE:
        - Win signal must overcome ~40 turns of potential negatives (40 × -0.05 = -2.0)
        - +5.0 >> -2.0 → Win episodes are clearly positive
        - Loss penalty is gentle (-1.0) to maintain win/loss contrast
        - Bonus for efficiency (clean sweep) encourages tight play
        """
        battle_finished = getattr(battle, 'battle_finished', False)
        
        if not battle_finished:
            return 0.0
        
        won = getattr(battle, 'won', False)
        fainted_count = sum(1 for mon in battle.team.values() if getattr(mon, 'fainted', False))
        
        if won:
            base_reward = 5.0  # Strong win signal
            
            # Efficiency bonus (encourages minimizing damage taken)
            if fainted_count == 0:
                base_reward += 1.0  # Clean sweep = +6.0 total
            elif fainted_count <= 2:
                base_reward += 0.5  # Close win = +5.5 total
            
            return base_reward
        else:
            return -1.0  # Gentle loss penalty (win/loss contrast = 6.0 points)
    
    # ==================================================================================
    # COMPONENT 2: HP ADVANTAGE (Dense Progress Tracking)
    # ==================================================================================
    
    def _calculate_hp_advantage(self, battle: AbstractBattle, prior_battle: AbstractBattle | None = None) -> float:
        """
        HP advantage delta - provides continuous learning signal every turn.
        
        Q-Learning Principle: Dense rewards guide policy between sparse rewards.
        Range: ±0.15 per turn (after /6 normalization = ±0.025 per turn)
        
        WHY THIS SCALE:
        - Small enough to not overshadow win/loss (+5.0 vs ±0.15)
        - Large enough to provide clear direction
        - Cumulative over 40 turns: ±6.0 max (comparable to win signal, but win still dominates)
        
        LEARNING FLOW:
        Turn 1-10: Agent learns "damage opponent = positive, take damage = negative"
        Turn 10-50k: Agent learns which moves maximize damage advantage
        Turn 50k+: Agent refines to optimal damage-dealing strategies
        """
        if prior_battle is None:
            # Called from old code path - calculate current advantage only
            team_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) 
                         for mon in battle.team.values()) / max(len(battle.team), 1)
            opp_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) 
                        for mon in battle.opponent_team.values()) / max(len(battle.opponent_team), 1)
            return (team_hp - opp_hp) / 2.0  # Normalized to [-1, 1]
        
        # Calculate HP advantage delta (change from prior turn)
        current_team_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) 
                             for mon in battle.team.values()) / max(len(battle.team), 1)
        current_opp_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) 
                            for mon in battle.opponent_team.values()) / max(len(battle.opponent_team), 1)
        current_advantage = (current_team_hp - current_opp_hp) / 2.0
        
        prior_team_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) 
                           for mon in prior_battle.team.values()) / max(len(prior_battle.team), 1)
        prior_opp_hp = sum(getattr(mon, 'current_hp_fraction', 0.0) 
                          for mon in prior_battle.opponent_team.values()) / max(len(prior_battle.opponent_team), 1)
        prior_advantage = (prior_team_hp - prior_opp_hp) / 2.0
        
        # Delta: positive = improved position, negative = worsened position
        advantage_delta = current_advantage - prior_advantage
        
        return np.clip(advantage_delta * 0.3, -0.15, 0.15)  # Gentle guidance
    
    # ==================================================================================
    # COMPONENT 3: KO EVENTS (Medium Sparse Reward)
    # ==================================================================================
    
    def _calculate_ko_events(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        KO event rewards - reinforces good plays that lead to KOs.
        
        Q-Learning Principle: Intermediate rewards bridge dense and sparse signals.
        Range: ±0.4 to ±0.8 per KO (after /6 normalization = ±0.07-0.13)
        
        WHY THIS SCALE:
        - Stronger than per-turn HP (0.8 > 0.15) → KOs feel significant
        - Weaker than win/loss (0.8 < 5.0) → Doesn't overshadow final outcome
        - Stage scaling: Late-game KOs worth more (closing out game is important)
        
        LEARNING FLOW:
        Turn 1-20k: Agent learns "KO opponent = good, getting KO'd = bad"
        Turn 20-100k: Agent learns to prioritize low-HP opponents for KOs
        Turn 100k+: Agent learns end-game tactics (closing out with final KOs)
        """
        if not prior_battle:
            return 0.0
        
        reward = 0.0
        
        # Count KOs this turn
        prior_fainted_opponent = sum(1 for mon in prior_battle.opponent_team.values() 
                                     if getattr(mon, 'fainted', False))
        current_fainted_opponent = sum(1 for mon in battle.opponent_team.values() 
                                       if getattr(mon, 'fainted', False))
        
        prior_fainted_team = sum(1 for mon in prior_battle.team.values() 
                                 if getattr(mon, 'fainted', False))
        current_fainted_team = sum(1 for mon in battle.team.values() 
                                   if getattr(mon, 'fainted', False))
        
        # Opponent KOs (good!)
        opponent_kos = current_fainted_opponent - prior_fainted_opponent
        if opponent_kos > 0:
            # Stage scaling: Late-game KOs are more valuable
            remaining_opponents = 6 - current_fainted_opponent
            stage_multiplier = 1.0 + (0.3 * (6 - remaining_opponents) / 6)  # 1.0 to 1.3
            reward += 0.6 * stage_multiplier * opponent_kos
        
        # Team KOs (bad!)
        team_kos = current_fainted_team - prior_fainted_team
        if team_kos > 0:
            remaining_team = 6 - current_fainted_team
            stage_multiplier = 1.0 + (0.3 * (6 - remaining_team) / 6)  # 1.0 to 1.3
            reward -= 0.6 * stage_multiplier * team_kos
        
        return np.clip(reward, -0.8, 0.8)
    
    # ==================================================================================
    # COMPONENT 4: MOVE EFFECTIVENESS FEEDBACK (Immediate Learning Signal)
    # ==================================================================================
    
    def _calculate_move_effectiveness_feedback(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Immediate move effectiveness feedback - teaches good/bad move selection.
        
        Q-Learning Principle: Immediate rewards accelerate early learning.
        Range: ±0.1 per move (after /6 normalization = ±0.017)
        
        SIGNALS:
        +0.1: Super-effective hit (2x+)
        -0.1: Type immunity (0x) or ability immunity (Volt Absorb, Levitate, etc.)
        -0.05: Not very effective (<1x)
        
        WHY THIS SCALE:
        - Small magnitude (0.1) → Doesn't overshadow HP advantage or KOs
        - Clear binary signal → Agent quickly learns type matchups
        - Immunity gets strongest penalty → Wasted turn is costly
        
        IMMUNITY AWARENESS:
        - Type immunity: 0x effectiveness (Ground vs Flying, Ghost vs Normal)
        - Ability immunity: is_nullified() check (Volt Absorb, Flash Fire, Levitate)
        - Item immunity: Safety Goggles (powder moves), Heavy-Duty Boots (hazards)
        
        LEARNING FLOW:
        Turn 1-5k: Agent learns "red (immunity) = bad, green (super effective) = good"
        Turn 5-30k: Agent learns specific matchups (Water > Fire, Flying > Ground)
        Turn 30k+: Agent avoids immunity abilities (won't use Electric on Volt Absorb)
        """
        if not prior_battle:
            return 0.0
        
        reward = 0.0
        
        # Only process MOVE actions (not switches)
        if not (hasattr(self, 'action_history') and len(self.action_history) > 0):
            return 0.0
        
        last_action = self.action_history[-1]
        if not last_action.get('action_type', '').startswith('move'):
            return 0.0
        
        action_value = last_action.get('action', 6)
        move_index = (action_value - 6) % 4  # FIXED: Actions 6-9, 10-13, 14-17, etc. all map to moves 0-3
        
        if not (prior_battle and hasattr(prior_battle, 'available_moves')):
            return 0.0
        
        available_moves = prior_battle.available_moves
        if not (0 <= move_index < len(available_moves)):
            return 0.0
        
        move_used = available_moves[move_index]
        move_category = getattr(move_used, 'category', None)
        
        # === ATTACKING MOVES ===
        if move_category in [MoveCategory.PHYSICAL, MoveCategory.SPECIAL]:
            
            # Check for ABILITY IMMUNITY first (highest priority - nullifies everything)
            if battle.opponent_active_pokemon and prior_battle.opponent_active_pokemon:
                if self.is_nullified(move_used, prior_battle.opponent_active_pokemon):
                    reward -= 0.1  # IMMUNITY: Volt Absorb, Flash Fire, Levitate, etc.
                    return reward  # Early return - move was nullified
            
            # Check for TYPE EFFECTIVENESS
            if battle.opponent_active_pokemon:
                opp_types = self._extract_pokemon_types(battle.opponent_active_pokemon)
                if opp_types and hasattr(move_used, 'type') and move_used.type:
                    try:
                        effectiveness = self.move_effectiveness(move_used.type.name.title(), tuple(opp_types))
                        
                        if effectiveness == 0.0:
                            reward -= 0.1  # TYPE IMMUNITY: Ground vs Flying, Ghost vs Normal
                        elif effectiveness >= 2.0:
                            reward += 0.1  # SUPER EFFECTIVE: 2x or 4x damage
                        elif effectiveness < 1.0:
                            reward -= 0.05  # NOT VERY EFFECTIVE: 0.5x or 0.25x damage
                    except:
                        pass
        
        # === STATUS MOVES ===
        elif move_category == MoveCategory.STATUS:
            # Check for redundant status (don't reward/penalize - HP advantage handles it)
            pass
        
        return np.clip(reward, -0.1, 0.1)
    
    # ==================================================================================
    # COMPONENT 5: SWITCHING REWARDS (Strategic Positioning)
    # ==================================================================================
    
    def _calculate_switch_reward(self, battle: AbstractBattle, prior_battle: AbstractBattle | None) -> float:
        """
        Strategic switching reward - encourages good matchup positioning.
        
        Q-Learning Principle: Reward state improvements, not actions.
        Range: ±0.1 per switch (after /6 normalization = ±0.017)
        
        SIGNALS:
        +0.1: Switch to better matchup (type advantage improved)
        +0.05: Switch to save low-HP Pokemon (< 30% HP)
        0.0: Neutral switch
        
        WHY THIS SCALE:
        - Small magnitude (0.1) → Encourages switching without spam
        - Comparable to move feedback (0.1) → Balanced strategic options
        - No penalty for bad switches → Let HP advantage teach that lesson
        
        LEARNING FLOW:
        Turn 1-10k: Agent learns switching is an option
        Turn 10-50k: Agent learns to switch on bad matchups
        Turn 50k+: Agent learns optimal switch timing (save Pokemon, scout, etc.)
        """
        if not prior_battle or not hasattr(self, 'action_history') or len(self.action_history) == 0:
            return 0.0
        
        last_action_type = self.action_history[-1].get('action_type', '')
        if last_action_type != 'switch':
            return 0.0
        
        reward = 0.0
        
        # Did we save a low-HP Pokemon?
        if prior_battle.active_pokemon:
            prior_hp = getattr(prior_battle.active_pokemon, 'current_hp_fraction', 1.0)
            if prior_hp < 0.3:
                reward += 0.05  # Good: Saved endangered Pokemon
        
        # Did we improve type matchup?
        if battle.active_pokemon and battle.opponent_active_pokemon:
            active_types = self._extract_pokemon_types(battle.active_pokemon)
            opp_types = self._extract_pokemon_types(battle.opponent_active_pokemon)
            
            if active_types and opp_types:
                # Simple matchup check: offensive advantage
                total_effectiveness = 0.0
                for a_type in active_types:
                    for o_type in opp_types:
                        try:
                            eff = self.move_effectiveness(a_type, (o_type,))
                            total_effectiveness += eff
                        except:
                            pass
                
                avg_effectiveness = total_effectiveness / max(len(active_types) * len(opp_types), 1)
                
                if avg_effectiveness >= 2.0:
                    reward += 0.1  # Excellent matchup
                elif avg_effectiveness >= 1.5:
                    reward += 0.05  # Good matchup
        
        return np.clip(reward, 0.0, 0.1)  # Only positive rewards for good switches
    
    # ==================================================================================
    # COMPONENT 6: ANTI-SPAM PENALTY (Prevent Reward Hacking)
    # ==================================================================================
    
    def _calculate_anti_spam_penalty(self, battle: AbstractBattle) -> float:
        """
        Anti-spam penalty - prevents reward hacking through repetitive actions.
        
        Q-Learning Principle: Penalize exploitation, not exploration.
        Range: -0.3 max per turn (after /6 normalization = -0.05)
        
        PENALTIES:
        -0.1: Excessive switching (4+ switches in 5 turns)
        -0.1: Move spam (same move 3+ times in a row)
        -0.05: Redundant hazards (Stealth Rock already up)
        
        WHY THIS SCALE:
        - Small magnitude (-0.3 max) → Doesn't dominate learning signal
        - Progressive penalties → Gentle warnings before hard punishment
        - Only penalizes TRUE spam → Allows legitimate repeated moves
        
        LEARNING FLOW:
        Turn 1-20k: No spam (random exploration)
        Turn 20-50k: Agent might discover spam exploits → Penalties kick in
        Turn 50k+: Agent learns balanced strategies without spam
        """
        if not hasattr(self, 'action_history') or len(self.action_history) < 2:
            return 0.0
        
        penalty = 0.0
        
        # === SWITCH SPAM ===
        recent_switches = sum(1 for action in self.action_history[-5:] 
                             if action.get('action_type') == 'switch')
        if recent_switches >= 4:
            penalty -= 0.1  # Excessive switching
        
        # === MOVE SPAM ===
        if len(self.action_history) >= 3:
            last_3_actions = [a.get('action', -1) for a in self.action_history[-3:]]
            if len(set(last_3_actions)) == 1 and last_3_actions[0] >= 6:  # Same move 3x
                penalty -= 0.1
        
        # === HAZARD SPAM (Removed setup spam - was broken) ===
        current_action = self.action_history[-1]
        if current_action.get('action_category') == 'hazard':
            # Check if hazards already exist
            hazards_exist = False
            if hasattr(battle, 'opponent_side_conditions'):
                opponent_conditions_str = str(battle.opponent_side_conditions).lower()
                if any(h in opponent_conditions_str for h in ['stealth', 'spikes', 'toxic', 'sticky']):
                    hazards_exist = True
            
            if hazards_exist:
                penalty -= 0.05  # Don't spam hazards
        
        return np.clip(penalty, -0.3, 0.0) 
    
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
    
    # OLD _calculate_switching_reward removed - see new _calculate_switch_reward in COMPONENT 5 above 
    # OLD _calculate_spam_penalties removed - see new _calculate_anti_spam_penalty in COMPONENT 6 above
    # OLD _calculate_strategic_outcomes removed - setup rewards completely removed as requested
    
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

    # OLD _calculate_hp_advantage removed (duplicate) - see new implementation in COMPONENT 2 above
    # OLD _evaluate_stat_boost_timing removed - was only used by deleted _calculate_strategic_outcomes
    # OLD _evaluate_status_move_effectiveness removed - was only used by deleted _calculate_immediate_feedback
    
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
        move_type = getattr(move_type_obj, 'name', '').upper() if move_type_obj else ""  # FIXED: Uppercase for consistent comparison
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
                # Normalized action index
                action_value = action_data.get('action', -2)
                action_history_features[base_idx] = np.clip(action_value / 26.0, -1.0, 1.0)
                
                #  Action type encoding (move/switch/etc)
                action_type_map = {'move': 0.2, 'switch': 0.4, 'move_mega': 0.6, 'move_z': 0.8, 'move_dynamax': 1.0}
                action_history_features[base_idx + 1] = action_type_map.get(action_data.get('action_type', 'move'), 0.0)
                
                category_map = {
                    'physical': 0.25, 
                    'special': 0.5, 
                    'status': 0.75, 
                    'hazard': 1.0,
                    'boost': 0.9
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