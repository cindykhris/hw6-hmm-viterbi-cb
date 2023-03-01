import copy
import numpy as np
class ViterbiAlgorithm:
    """_summary_
    """    

    def __init__(self, hmm_object):
        """_summary_
        Args:
            hmm_object (_type_): _description_
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """_summary_
        Args:
            decode_observation_states (np.ndarray): _description_
        Returns:
            np.ndarray: _description_
        """

        # Initialize the best path score and best path
        best_path_score = np.zeros((self.hmm_object.num_states, decode_observation_states.shape[0]))
        best_path = np.zeros((self.hmm_object.num_states, decode_observation_states.shape[0]), dtype=int)

        # Initialize the best path score for the first observation
        best_path_score[:, 0] = self.hmm_object.initial_state_distribution + self.hmm_object.observation_probabilities[:, decode_observation_states[0]]

        # Iterate over the observations
        for t in range(1, decode_observation_states.shape[0]):

            # Iterate over the states
            for j in range(self.hmm_object.num_states):

                # Calculate the best path score
                best_path_score[j, t] = np.max(best_path_score[:, t-1] + self.hmm_object.transition_probabilities[:, j]) + self.hmm_object.observation_probabilities[j, decode_observation_states[t]]

                # Calculate the best path
                best_path[j, t] = np.argmax(best_path_score[:, t-1] + self.hmm_object.transition_probabilities[:, j])

        # Initialize the best path
        best_path_sequence = np.zeros(decode_observation_states.shape[0], dtype=int)

        # Get the best path
        best_path_sequence[-1] = np.argmax(best_path_score[:, -1])
        for t in range(decode_observation_states.shape[0]-2, -1, -1):
            best_path_sequence[t] = best_path[best_path_sequence[t+1], [t+1]]

        return best_path_sequence

