import numpy as np
import random
from engine.yahtzee_engine import YahtzeeState, YahtzeeAction, YahtzeeEngine
import time
import copy
import csv
import multiprocessing

class PoolManager:
    _pool = None
    _num_processes = 2  # Default number of processes

    @classmethod
    def get_pool(cls):
        if cls._pool is None or cls._pool._state != multiprocessing.pool.RUN:
            cls._pool = multiprocessing.Pool(cls._num_processes)
        return cls._pool

    @classmethod
    def set_num_processes(cls, num_processes):
        cls._num_processes = num_processes
        cls.shutdown_pool()  # Reset pool if process number changes

    @classmethod
    def shutdown_pool(cls):
        if cls._pool is not None:
            cls._pool.close()
            cls._pool.join()
            cls._pool = None


class ParallelMCTSNode:
    def __init__(self, state: YahtzeeState, game_engine: YahtzeeEngine, parent=None, action: YahtzeeAction=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.game_engine = game_engine
        self.untried_actions = list(self.game_engine.get_possible_actions(state.is_final, state.remaining_rolls, tuple(state.score_card)))

    def expand(self):
        if not self.untried_actions:
            raise Exception("No actions left to expand")
        action = self.untried_actions.pop()
        next_state = self.game_engine.apply_action(self.state, action)
        new_node = ParallelMCTSNode(next_state, self.game_engine, parent=self, action=action)
        self.children.append(new_node)
        return new_node
    
    def update(self, result: float):
        self.visits += 1
        self.wins += result
    
    def uct_select_child(self):
        if self.state.is_final:
            raise Exception("Attempted to select a child from a terminal node")
        elif not self.children:
            raise Exception("No children available and no actions to expand")
        C = 1.414  # Exploration parameter
        return max(self.children, key=lambda child: child.get_uct_value())

    def get_uct_value(self):
        if self.visits == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        return self.wins / self.visits + 1.414 * np.sqrt(2 * np.log(self.parent.visits) / self.visits)

    def deep_copy(self):
        # Create a deep copy of the node
        copied_node = ParallelMCTSNode(copy.deepcopy(self.state), self.game_engine, parent=None, action=self.action)
        copied_node.children = [child.deep_copy() for child in self.children]
        copied_node.wins = self.wins
        copied_node.visits = self.visits
        copied_node.untried_actions = list(self.untried_actions)  # Ensure this is a new list
        return copied_node

    def __repr__(self):
        action_desc = f"Action: {self.action}, " if self.action else ""
        return (f"<ParallelMCTSNode {action_desc}Wins: {self.wins}, Visits: {self.visits}, "
                f"Untried Actions: {len(self.untried_actions)}, Children: {len(self.children)}>")


class ParallelMCTSTree:
    def __init__(self, pool_manager: PoolManager , root: ParallelMCTSNode, game_engine: YahtzeeEngine, simulation_depth: int = 1000, num_simulations: int = 1000, num_processes: int = 2):
        self.root = root
        self.game_engine = game_engine
        self.simulation_depth = simulation_depth
        self.num_simulations = num_simulations
        self.num_processes = num_processes

    def simulate(self, node: ParallelMCTSNode):
        current_depth = 0
        current_node = node
        while not current_node.state.is_final and current_depth < self.simulation_depth:
            if current_node.untried_actions:
                current_node = current_node.expand()
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
        return self.game_engine.get_total_score(tuple(current_state.score_card)) / 400


    def run(self, root_copy):
        current_score = self.game_engine.get_total_score(tuple(root_copy.state.score_card)) / 400
        for i in range(self.num_simulations):
            node = self.simulate(root_copy)  # Start simulation from the current root
            result = self.rollout(node.state)  # Perform rollout from the returned node's state
            value = result - current_score # Take the difference between rollout score and current score
            while node is not None:
                node.update(value)  # Update node with the result from the rollout
                node = node.parent  # Move to the parent node until the root is reached

        return root_copy, value

    def aggregate_trees(self, results):
        self.root = results[0][0]
        if len(results) > 1:
            action_to_original_child = {child.action: child for child in self.root.children}

            for result in results[1:]:
                self.root.visits += result[0].visits
                self.root.wins += result[0].wins

                for new_child in result[0].children:
                    if new_child.action in action_to_original_child:
                        action_to_original_child[new_child.action].visits += new_child.visits
                        action_to_original_child[new_child.action].wins += new_child.wins
                    else:
                        print("Aggregation error !")
                        print(new_child.action)
                        print(action_to_original_child)
                        print('*---------------*')

    def decide_move(self):
        pool = PoolManager.get_pool()
        tasks = [(self.root.deep_copy()) for _ in range(self.num_processes)]
        results = pool.map(self.run, tasks)

        self.aggregate_trees(results)

        best_action = max(self.root.children, key=lambda child: child.wins / child.visits if child.visits > 0 else -float('inf')).action
        return best_action

## Test simple MCTS approach

def main():
    pool_manager = PoolManager()
    for param_count, param in enumerate([500]):
        game_engine = YahtzeeEngine()
        initial_state = YahtzeeState((0,0,0,0,0), (None,)*13, 3)
        actions = game_engine.get_possible_actions(initial_state.is_final, initial_state.remaining_rolls, initial_state.score_card)

        score_list = []
        decsion_time_list = []

        for count, game in enumerate(range(1)):

            state = game_engine.apply_action(initial_state, actions[0])
            play_count = 0

            decision_time = 0

            while True:

                # print('-------------')
                # print(state)

                root_node = ParallelMCTSNode(state=state, game_engine=game_engine)
                mcts_tree = ParallelMCTSTree(pool_manager=pool_manager, root=root_node, game_engine=game_engine, num_simulations=param)

                start = time.time()
                best_action = mcts_tree.decide_move()
                end = time.time()
                diff = end - start

                # print("Chosen action")
                # print(best_action)

                decision_time += diff

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

        filename = f'output_mcts_multi_{param}_sim_2_worker_20_games.csv'

        # Open the file in write mode
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
        
            # Optionally write headers
            writer.writerow(['score', 'time'])
        
            # Write the data
            for item1, item2 in zip(score_list, decsion_time_list):
                writer.writerow([item1, item2])

        print(f"Data written to {filename} successfully.")

        pool_manager.shutdown_pool()

if __name__ == '__main__':
    main()

