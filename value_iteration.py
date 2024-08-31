import numpy as np
import itertools
from engine.yahtzee_engine import YahtzeeState, YahtzeeAction, YahtzeeEngine
import time


class ValueIteration:
    def __init__(self, state_space: tuple, actions_space: tuple, game_engine: YahtzeeEngine, gamma: float = 0.99, tolerance: float = 1, max_iterations: int = 1000):
        """
        Initialize Value Iteration algorithm
        """


        self.state_space = state_space
        self.state_index = {state: idx for idx, state in enumerate(self.state_space)}
        self.actions_space = actions_space

        self.game_engine = game_engine
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.gamma = gamma

        self.values = np.zeros(len(state_space))
        self.policy = [None] * len(state_space)

    def run_iteration(self):
        """
        Run the value iteration
        """
        delta = 0
        iteration_count = 1
        start_time = time.time()

        print("Starting iteration")
        print(f"State space size :{len(self.state_space)}")

        while delta > self.tolerance or iteration_count < self.max_iterations:
            
            delta = 0

            for count, state in enumerate(self.state_space):
                value = self.values[count]
                possible_actions = self.game_engine.get_possible_actions(state.is_final, state.remaining_rolls, state.score_card)
                self.values[count] =  max(self.calculate_value(state, action) for action in possible_actions)
                delta = max(delta, abs(value - self.values[count]))

                print(f"State {count} done")

            print(f"Iteration {iteration_count}: Max Value Change (Delta) = {delta:.6f}")
            print(f"Elapsed time : {time.time() - start_time}")

            iteration_count += 1

        self.extract_policy()

    def calculate_value(self, state: YahtzeeState, action: YahtzeeAction):
        """
        Calculate the value of taking an action in a given state.
        """
        total_value = 0

        # Retrieve transition probabilities for the current state-action pair
        transitions = self.game_engine.get_transition_probabilities(state, action)
        next_possible_states = list(transitions.keys())

        # Precompute rewards for each possible next state
        rewards = {next_state: self.game_engine.calculate_reward(state, action, next_state) for next_state in next_possible_states}

        for count, next_state in enumerate(next_possible_states):

            next_state_index = self.state_index[next_state]
            next_state_transition_probability = transitions[next_state] # Probability of transitioning to next_state
            next_state_reward = rewards[next_state]

            future_value = self.gamma * self.values[next_state_index] # Discounted value of next state
            total_value += next_state_transition_probability * (next_state_reward + future_value)

        return total_value


engine = YahtzeeEngine()

# Generate full action space
initial_state = YahtzeeState((0,0,0,0,0), (None,)*13, 3)
actions = engine.get_possible_actions(initial_state.is_final, initial_state.remaining_rolls, initial_state.score_card)
state = engine.apply_action(initial_state, actions[0])
action_space = engine.get_possible_actions(state.is_final, state.remaining_rolls, state.score_card)

# Generate full state space
dice_rolls = list(itertools.combinations_with_replacement(range(1,7), 5))
dice_rolls = [tuple(sorted(d)) for d in dice_rolls]
score_cards = list(itertools.product([None, 0], repeat=13)) # All possible scorecard for 13 turns

remaining_rolls = [3, 2, 1, 0]

state_space = [YahtzeeState(dice, score_card, rolls) for dice in dice_rolls for score_card in score_cards for rolls in remaining_rolls]

print(len(state_space))

value_iteration = ValueIteration(state_space, action_space, engine, 0.99, 1, 13)
value_iteration.run_iteration()
# value_iteration.save_vtable("vtable_t1.csv")
value_iteration = ValueIteration(state_space, action_space, engine, 0.99, 1, 13)
value_iteration.run_iteration()
# value_iteration.save_vtable("vtable_t1.csv")
