"""Monte Carlo Tree Search agent for Yahtzee (decision / chance node version).

Dice re-rolls are modelled as explicit chance nodes, so the value of a roll
action is the expectation over outcomes rather than a single frozen sample.

Example:
    python mcts_v2.py --simulations 500 1000 2500 5000 \\
                      --c-param 0.47 1.0 1.41 \\
                      --games 200 --output-dir results --seed 0
"""

import argparse
import csv
import os
import random
import time

import numpy as np

from engine.yahtzee_engine import YahtzeeState, YahtzeeAction, YahtzeeEngine


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

class MCTSNode:
    """Fields and statistics common to decision and chance nodes."""

    def __init__(self, game_engine, parent=None, incoming_action=None):
        self.game_engine = game_engine
        self.parent = parent
        self.incoming_action = incoming_action
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    def get_value_estimate(self):
        if self.visits == 0:
            return float("inf")          # unvisited -> explored first
        return self.value_sum / self.visits

    def update(self, result):
        self.visits += 1
        self.value_sum += result

    def __repr__(self):
        action = f"action={self.incoming_action}, " if self.incoming_action else ""
        return (f"<{type(self).__name__} {action}"
                f"value={self.get_value_estimate():.2f}, visits={self.visits}, "
                f"children={len(self.children)}>")


class DecisionNode(MCTSNode):
    """A state where the agent chooses an action. Children are keyed by action."""

    def __init__(self, state, game_engine, incoming_action=None, parent=None, c_param=1.414):
        super().__init__(game_engine, parent, incoming_action)
        self.state = state
        self.c_param = c_param
        self.untried_actions = list(game_engine.get_possible_actions(
            state.is_final, state.remaining_rolls, tuple(state.score_card)))
        random.shuffle(self.untried_actions)     # avoid biased LIFO expansion order

    def is_terminal(self):
        return self.state.is_final

    def is_fully_expanded(self):
        return not self.untried_actions

    def expand(self):
        if self.is_fully_expanded():
            raise RuntimeError("No actions left to expand")

        action = self.untried_actions.pop()

        if action.action_type == "roll":
            chance_node = ChanceNode(self, action)
            self.children[action] = chance_node
            leaf, _ = chance_node.select_outcome()
            return leaf

        if action.action_type == "score":
            next_state = self.game_engine.apply_action(self.state, action)
            child = DecisionNode(next_state, self.game_engine,
                                 incoming_action=action, parent=self, c_param=self.c_param)
            self.children[action] = child
            return child

        raise ValueError(f"Unknown action type: {action.action_type}")

    def uct_select_child(self):
        if self.state.is_final:
            raise RuntimeError("Cannot select a child from a terminal node")
        if not self.children:
            raise RuntimeError("No children to select from")

        # Normalise child values into [0, 1] using the range seen at this node,
        # so c_param is invariant to the absolute reward scale.
        values = [c.get_value_estimate() for c in self.children.values() if c.visits > 0]
        v_min = min(values) if values else 0.0
        v_max = max(values) if values else 1.0
        v_range = (v_max - v_min) or 1.0

        def uct_score(child):
            if child.visits == 0:
                return float("inf")
            exploit = (child.get_value_estimate() - v_min) / v_range
            explore = self.c_param * np.sqrt(np.log(self.visits) / child.visits)
            return exploit + explore

        return max(self.children.values(), key=uct_score)

    def best_action(self):
        """Action of the most valuable visited child (final move choice)."""
        visited = [(a, c) for a, c in self.children.items() if c.visits > 0]
        if not visited:
            raise RuntimeError("No visited children to choose from")
        return max(visited, key=lambda ac: ac[1].get_value_estimate())[0]


class ChanceNode(MCTSNode):
    """A dice re-roll. No UCT: outcomes are sampled. Children keyed by state."""

    def __init__(self, parent_decision, incoming_action):
        super().__init__(parent_decision.game_engine,
                         parent=parent_decision, incoming_action=incoming_action)
        self.state = parent_decision.state       # pre-roll state we keep rolling from
        self.c_param = parent_decision.c_param

    def select_outcome(self):
        """Sample one outcome; return (decision_child, is_new)."""
        next_state = self.game_engine.apply_action(self.state, self.incoming_action)
        child = self.children.get(next_state)
        if child is None:
            child = DecisionNode(next_state, self.game_engine,
                                 parent=self, c_param=self.c_param)
            self.children[next_state] = child
            return child, True
        return child, False


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

class MCTSTree:
    def __init__(self, root, game_engine, simulation_depth=100, num_simulations=1000):
        self.root = root
        self.game_engine = game_engine
        self.simulation_depth = simulation_depth
        self.num_simulations = num_simulations

    def select(self, node):
        """Tree policy: descend to a fresh leaf, alternating decision (UCT) and
        chance (sampling) nodes."""
        current = node
        depth = 0
        while depth < self.simulation_depth:
            if isinstance(current, ChanceNode):
                current, is_new = current.select_outcome()
                if is_new:
                    break
            else:  # DecisionNode
                if current.state.is_final:
                    break
                if current.untried_actions:
                    current = current.expand()
                    break
                current = current.uct_select_child()
            depth += 1
        return current

    def rollout(self, state):
        """Random play-out to a terminal state; return the final total score."""
        current = state
        while not current.is_final:
            actions = self.game_engine.get_possible_actions(
                current.is_final, current.remaining_rolls, tuple(current.score_card))
            current = self.game_engine.apply_action(current, random.choice(actions))
        return self.game_engine.get_total_score(tuple(current.score_card))

    def backpropagate(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def decide_move(self):
        for _ in range(self.num_simulations):
            leaf = self.select(self.root)
            result = self.rollout(leaf.state)
            self.backpropagate(leaf, result)
        return self.root.best_action()


# --------------------------------------------------------------------------- #
# Experiment runner
# --------------------------------------------------------------------------- #

def play_game(engine, num_simulations, c_param, simulation_depth):
    """Play one solitaire game; return (final_score, total_decision_time)."""
    initial_state = YahtzeeState((0, 0, 0, 0, 0), (None,) * 13, 3)
    forced_first_roll = engine.get_possible_actions(
        initial_state.is_final, initial_state.remaining_rolls, initial_state.score_card)[0]
    state = engine.apply_action(initial_state, forced_first_roll)

    decision_time = 0.0
    while not state.is_final:
        actions = engine.get_possible_actions(
            state.is_final, state.remaining_rolls, tuple(state.score_card))
        if len(actions) == 1:
            action = actions[0]                       # forced move: skip the search
        else:
            root = DecisionNode(state, engine, c_param=c_param)
            tree = MCTSTree(root, engine,
                            simulation_depth=simulation_depth, num_simulations=num_simulations)
            start = time.perf_counter()
            action = tree.decide_move()
            decision_time += time.perf_counter() - start
        state = engine.apply_action(state, action)

    return state.total_score, decision_time


def write_csv(path, scores, times):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "decision_time"])
        writer.writerows(zip(scores, times))


def run_experiment(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    engine = YahtzeeEngine()
    os.makedirs(args.output_dir, exist_ok=True)

    for c_param in args.c_param:
        for num_simulations in args.simulations:
            scores, times = [], []
            for g in range(args.games):
                score, decision_time = play_game(
                    engine, num_simulations, c_param, args.simulation_depth)
                scores.append(score)
                times.append(decision_time)
                if args.verbose:
                    print(f"[C={c_param} sims={num_simulations}] "
                          f"game {g + 1}/{args.games}: score={score}")

            sd = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
            print(f"== C={c_param}, sims={num_simulations}: "
                  f"mean {np.mean(scores):.2f} (sd {sd:.2f}), "
                  f"mean decision time {np.mean(times):.3f}s ==")

            fname = os.path.join(
                args.output_dir,
                f"mcts_c{c_param}_sim{num_simulations}_games{args.games}.csv")
            write_csv(fname, scores, times)
            print(f"   wrote {fname}")


def parse_args():
    p = argparse.ArgumentParser(description="MCTS Yahtzee experiment runner.")
    p.add_argument("--simulations", type=int, nargs="+", default=[500, 1000, 2500, 5000],
                   help="Simulation budget(s) per move to sweep.")
    p.add_argument("--c-param", type=float, nargs="+", default=[1.414],
                   help="UCT exploration constant(s) to sweep.")
    p.add_argument("--games", type=int, default=200,
                   help="Games per (C, simulations) cell.")
    p.add_argument("--simulation-depth", type=int, default=100,
                   help="Max tree-policy depth (>= 52 never binds for Yahtzee).")
    p.add_argument("--output-dir", default=".", help="Directory for CSV output.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    p.add_argument("--verbose", action="store_true", help="Print per-game scores.")
    return p.parse_args()


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
