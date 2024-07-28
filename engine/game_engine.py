from abc import ABC, abstractmethod


class State(ABC):
	pass


class Action(ABC):
	pass


class GameEngine(ABC):

	@abstractmethod
	def apply_action(self, state: State, action: Action) -> State:
		"""
		Apply an action to a state (transition to a new state)
		"""
		pass

	@abstractmethod
	def calculate_reward(self, old_state: State, action: Action, new_state: State):
		"""
		Calculate the reward of an action by taking the difference between previous state and new state
		"""
		pass

	@abstractmethod
	def get_possible_actions(self, state: State):
		"""
		Abstract method that return all possible actions from a given state
		"""
		pass

	def get_transition_probabilities(self, state: State, action: Action):
		"""
		Optional method to get the transition probability for state-action pair
		"""
		raise NotImplementedError("Subclasses must implement this method.")

