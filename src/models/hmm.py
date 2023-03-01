import numpy as np
class HiddenMarkovModel:

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_probabilities: np.ndarray, transition_probabilities: np.ndarray, emission_probabilities: np.ndarray):
        """_summary_

        Args:
            observation_states (np.ndarray): _description_
            hidden_states (np.ndarray): _description_
            prior_probabilities (np.ndarray): _description_
            transition_probabilities (np.ndarray): _description_
            emission_probabilities (np.ndarray): _description_
        """             
        self.observation_states = observation_states
        self.observation_states_dict = {observation_state: observation_state_index \
                                  for observation_state_index, observation_state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {hidden_state_index: hidden_state \
                                   for hidden_state_index, hidden_state in enumerate(list(self.hidden_states))}
        

        self.prior_probabilities= prior_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities

        self.num_states = self.hidden_states.shape[0]
        self.num_observations = self.observation_states.shape[0]

        self.initial_state_distribution = np.log(self.prior_probabilities)
        self.transition_probabilities = np.log(self.transition_probabilities)
        self.observation_probabilities = np.log(self.emission_probabilities)

    def decode(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            decode_observation_states (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """        
        viterbi_algorithm = ViterbiAlgorithm(self)
        best_hidden_state_sequence = viterbi_algorithm.best_hidden_state_sequence(decode_observation_states)

        return best_hidden_state_sequence