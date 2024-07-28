import random
import itertools
import numpy as np
from engine.constants import SCORING_CATEGORIES
from engine.game_engine import GameEngine, State, Action
from functools import lru_cache


class YahtzeeState(State): 
    ## Optimized version of Yahtzee state for memory efficiency
    __slots__ = ['dice_code', 'score_card_mask', 'remaining_rolls', 'total_score']

    def __init__(self, dice: tuple, score_card: tuple, remaining_rolls: int, total_score: int = 0):
        self.dice_code = self.encode_dice(dice)
        scores = [score if score is not None else -1 for score in score_card]
        self.score_card = tuple(scores)
        self.remaining_rolls = np.int8(remaining_rolls)
        self.total_score = np.int16(total_score)
        self.is_final = min(self.score_card) >= 0

    def encode_dice(self, dice: tuple):
        """Encode dice tuple into a single integer."""
        code = 0
        for die in sorted(dice):
            code = code * 7 + die
        return code

    def decode_dice(self):
        """Decode the single integer back into a sorted dice tuple."""
        result = []
        code = self.dice_code
        for _ in range(5):
            result.append(code % 7)
            code //= 7
        return tuple(sorted(result))

    def __hash__(self):
        return hash((self.dice_code, tuple(self.score_card), self.remaining_rolls))

    def __eq__(self, other):
        return (self.dice_code, tuple(self.score_card), self.remaining_rolls) == (other.dice_code, tuple(other.score_card), other.remaining_rolls)

    def __repr__(self):
        return (f"YahtzeeState(dice={self.decode_dice()}, score_card={list(self.score_card)}, "
                f"remaining_rolls={self.remaining_rolls}, total_score={self.total_score})")


class YahtzeeAction(Action):
    def __init__(self, action_type, details):
        self.action_type = action_type
        self.details = details

    def __repr__(self):
        return f"Action(type={self.action_type}, details={self.details})"


class YahtzeeEngine(GameEngine):
    def roll_dice(self, state: YahtzeeState, keep_indices: tuple):
        """
        Perfom dice roll given the initial state and indices of dices to be kept
        """
        new_dice = list(state.decode_dice())
        for i in range(5):
            if i not in keep_indices:
                new_dice[i] = random.randint(1, 6)
        return tuple(sorted(new_dice))

    @lru_cache(maxsize=512)
    def calculate_score(self, dice: tuple, category: str):
        """
        Calculate score given a scoring category and a dice combination
        """
        category_value_map = {
            'ones': 1,
            'twos': 2,
            'threes': 3,
            'fours': 4,
            'fives': 5,
            'sixes': 6
        }

        dice_count = {x: dice.count(x) for x in set(dice)}

        if category in ['ones', 'twos', 'threes', 'fours', 'fives', 'sixes']:
            target_value = category_value_map[category]
            dice_count = {x: dice.count(x) for x in range(1, 7)}
            return dice_count.get(target_value, 0) * target_value

        elif category == 'three_of_a_kind':
            return sum(dice) if any(c >= 3 for c in dice_count.values()) else 0
        elif category == 'four_of_a_kind':
            return sum(dice) if any(c >= 4 for c in dice_count.values()) else 0
        elif category == 'full_house':
            if set(dice_count.values()) == {2, 3}:
                return 25
            return 0
        elif category == 'small_straight':
            straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
            return 30 if any(s.issubset(dice) for s in straights) else 0
        elif category == 'large_straight':
            straights = [{1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}]
            return 40 if any(s.issubset(dice) for s in straights) else 0
        elif category == 'yahtzee':
            return 50 if len(set(dice)) == 1 else 0
        elif category == 'chance':
            return sum(dice)

        return 0

    def apply_action(self, state: YahtzeeState, action: YahtzeeAction):
        """
        Apply action to current state (equivalent to transitioning into a new state)
        """
        if state.is_final:
                return state

        if action.action_type == 'roll':
            new_dice = self.roll_dice(state, action.details)
            new_state = YahtzeeState(new_dice, state.score_card, state.remaining_rolls - 1, state.total_score)

        elif action.action_type == 'score':
            dice = state.decode_dice()
            score_card = list(state.score_card)
            index = SCORING_CATEGORIES.index(action.details)

            if score_card[index] != -1:  # This category is already scored
                return state

            score = self.calculate_score(dice, action.details)
            score_card[index] = score  # Update score card
            new_total_score = self.get_total_score(tuple(score_card)) # Update total score
            new_state = YahtzeeState((0,0,0,0,0), score_card, 3, new_total_score)  # Reset dice and rolls

        return new_state

    def calculate_reward(self, old_state: YahtzeeState, action: YahtzeeAction, new_state: YahtzeeState):
        """
        Calculate the immediate reward of an action
        """
        if action.action_type == 'score':
            return new_state.total_score - old_state.total_score
        return 0

    @lru_cache(maxsize=512)
    def get_possible_actions(self, is_final: bool, remaining_rolls: int, score_card: tuple) -> tuple[YahtzeeAction]:
        """
        Generate space of possible actions given the a state
        """
        actions = []

        if is_final:
            return []

        if remaining_rolls == 3:
            return [YahtzeeAction('roll', ())]

        if remaining_rolls > 0:
            actions.append(YahtzeeAction('roll', ())) # Add possibility for re-rolling all dices
            indices = list(range(5))
            for r in range(1, 6):
                for subset in itertools.combinations(indices, r):
                    actions.append(YahtzeeAction('roll', subset))

        # Alternative version : Do not allow scoring until last turn

        if remaining_rolls == 0:
            actions = [
                        YahtzeeAction('score', category)
                        for category in SCORING_CATEGORIES
                        if score_card[SCORING_CATEGORIES.index(category)] == -1
                    ]

        return tuple(actions)

    def get_transition_probabilities(self, state: YahtzeeState, action: YahtzeeAction) -> dict:
        """
        Given a state and an action, compute the possible new states and their probabilities.
        """
        if action.action_type == 'roll':
            return self._transition_for_roll(state, action)
        elif action.action_type == 'score':
            return self._transition_for_score(state.score_card, state.decode_dice(), action.details)

    def _transition_for_roll(self, state: YahtzeeState, action: YahtzeeAction):
        """
        Calculate the transition probabilities for a dice roll action.
        """
        new_states = {}
        dice_to_roll = [i for i in range(5) if i not in action.details]
        outcomes = itertools.product(range(1, 7), repeat=len(dice_to_roll))
        
        for outcome in outcomes:
            new_dice = list(state.decode_dice())

            for i, die_index in zip(outcome, dice_to_roll):
                new_dice[die_index] = i

            new_dice = tuple(sorted(new_dice))
            new_state = YahtzeeState(new_dice, state.score_card, state.remaining_rolls - 1, state.total_score)
            probability = (1/6) ** len(dice_to_roll)   # Each die has an equal chance of landing on 1-6

            if new_state in new_states:
                new_states[new_state] += probability
            else:
                new_states[new_state] = probability
        
        return new_states

    @lru_cache(maxsize=512)
    def _transition_for_score(self, score_card: tuple, dice: tuple, details: str):
        """
        Calculate the transition probabilities for a scoring action.
        """
        new_score_card = list(score_card)
        category_index = SCORING_CATEGORIES.index(details)
        new_score_card[category_index] = self.calculate_score(dice, details)

        new_total_score = self.get_total_score(tuple(new_score_card)) # Get new total score
        
        new_state = YahtzeeState((0,0,0,0,0), tuple(new_score_card), 3, new_total_score)  # Reset the roll count after scoring

        return {new_state: 1}  # Deterministic transition

    @lru_cache(maxsize=512)
    def get_total_score(self, score_card: tuple):
        """
        Calculate the total score across the board for a given state. Include the upper score board bonus (if > 63)
        """
        upper_board_score = sum(x for x in score_card[0:6] if x != -1)
        lower_board_score = sum(x for x in score_card[6:] if x != -1)

        if upper_board_score >= 63:
            return upper_board_score + lower_board_score + 35 # Bonus for upper board score greater than or equal to 63
        else:
            return upper_board_score + lower_board_score
