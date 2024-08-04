import numpy as np
import random
from engine.yahtzee_engine import YahtzeeState, YahtzeeAction, YahtzeeEngine
import time
import copy
import csv


class MCTSNode:
    def __init__(self, state: YahtzeeState, game_engine: YahtzeeEngine, parent=None, action: YahtzeeAction=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.sum_normalized_score = 0
        self.visits = 0
        self.game_engine = game_engine
        self.untried_actions = list(self.game_engine.get_possible_actions(state.is_final, state.remaining_rolls, tuple(state.score_card)))

    def expand(self):
        if not self.untried_actions:
            raise Exception("No actions left to expand")
        action = self.untried_actions.pop()
        next_state = self.game_engine.apply_action(self.state, action)
        new_node = MCTSNode(next_state, self.game_engine, parent=self, action=action)
        self.children.append(new_node)
        return new_node
    
    def update(self, result: float):
        self.visits += 1
        self.sum_normalized_score += result
    
    def uct_select_child(self):
        if self.state.is_final:
            raise Exception("Attempted to select a child from a terminal node")
        elif not self.children:
            raise Exception("No children available and no actions to expand")
        return max(self.children, key=lambda child: child.get_uct_value())

    def get_uct_value(self):
        if self.visits == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        return self.sum_normalized_score / self.visits + 1.414 * np.sqrt(2 * np.log(self.parent.visits) / self.visits)

    def __repr__(self):
        action_desc = f"Action: {self.action}, " if self.action else ""
        return (f"<MCTSNode {action_desc} sum_normalized_score: {self.sum_normalized_score}, Visits: {self.visits}, "
                f"Untried Actions: {len(self.untried_actions)}, Children: {len(self.children)}>")


class MCTSTree:
    def __init__(self, root: MCTSNode, game_engine: YahtzeeEngine, simulation_depth: int = 100, num_simulations: int = 1000):
        self.root = root
        self.game_engine = game_engine
        self.simulation_depth = simulation_depth
        self.num_simulations = num_simulations
    
    def simulate(self, node: MCTSNode):
        current_depth = 0
        current_node = node
        while not current_node.state.is_final and current_depth < self.simulation_depth:
            if current_node.untried_actions:
                current_node = current_node.expand()
                break
            else:
                current_node = current_node.uct_select_child()
            current_depth += 1
        return current_node
    
    def rollout(self, state: YahtzeeState):
        # Simulate a random play-out from the state
        current_state = state
        while not current_state.is_final:
            possible_actions = self.game_engine.get_possible_actions(current_state.is_final, current_state.remaining_rolls, tuple(current_state.score_card))
            action = random.choice(possible_actions)
            current_state = self.game_engine.apply_action(current_state, action)
        return self.game_engine.get_total_score(tuple(current_state.score_card)) / 400  # Normalize score
    
    def decide_move(self):
        current_score = self.game_engine.get_total_score(tuple(self.root.state.score_card)) / 400
        for i in range(self.num_simulations):
            node = self.simulate(self.root)  # Start simulation from the current root
            result = self.rollout(node.state)  # Perform rollout from the returned node's state
            value = result # Take the difference between rollout score and current score
            while node is not None:
                node.update(value)  # Update node with the result from the rollout
                node = node.parent  # Move to the parent node until the root is reached

        # Select the action leading to the child with the highest average score
        best_action = max(self.root.children, key=lambda child: child.sum_normalized_score / child.visits if child.visits > 0 else -float('inf')).action
        return best_action



## Test simple MCTS approach

for param_count, param in enumerate([500,1000,2500,5000]):

    game_engine = YahtzeeEngine()
    initial_state = YahtzeeState((0,0,0,0,0), (None,)*13, 3)
    actions = game_engine.get_possible_actions(initial_state.is_final, initial_state.remaining_rolls, initial_state.score_card)

    score_list = []
    decsion_time_list = []

    for count, game in enumerate(range(200)):

        decision_time = 0

        state = game_engine.apply_action(initial_state, actions[0])
        play_count = 0

        while True:

            # print('-------------')
            # print(state)

            root_node = MCTSNode(state=state, game_engine=game_engine)
            mcts_tree = MCTSTree(root=root_node, game_engine=game_engine, num_simulations=param)

            actions = game_engine.get_possible_actions(initial_state.is_final, initial_state.remaining_rolls, initial_state.score_card)

            if len(actions) == 1:
                best_action = actions[0]
            else:
                start = time.time()
                best_action = mcts_tree.decide_move()
                end = time.time()
                diff = end - start
                decision_time += diff

            # print("Chosen action")
            # print(best_action)

            new_state = game_engine.apply_action(state, best_action)
            #reward = game_engine.calculate_reward(state, best_action, new_state)

            state = new_state
            play_count += 1

            if state.is_final:
                score_list.append(state.total_score)
                decsion_time_list.append(decision_time)
                print(f'Game finished, score : {state.total_score}')
                break

    print(f"Evaluation for param={param} completed")
    print(f"Avg score : {np.mean(score_list)}")
    print(f"Total decision time : {np.mean(decsion_time_list)}")

    filename = f'output_mcts_seq_{param}_sim_20games_{param_count + 1}.csv'

    # Open the file in write mode
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Optionally write headers
        writer.writerow(['score', 'time'])
        
        # Write the data
        for item1, item2 in zip(score_list, decsion_time_list):
            writer.writerow([item1, item2])

    print(f"Data written to {filename} successfully.")
