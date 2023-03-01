"""
UCSF BMI203: Biocomputing Algorithms
Author: Cindy Pino-Barrios
Date: February 28, 2023
Program: models
Description: Test the use case of the Hidden Markov Model and Viterbi Algorithm
"""
import pytest
import numpy as np
from models.hmm import HiddenMarkovModel
from models.decoders import ViterbiAlgorithm


def test_use_case_lecture():
    """_summary_
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('../data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_one():
    """_summary_
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_two():
    """_summary_ would a humminbird come to the surface to feed? 
    hypothesis: if the humminbird is in the water, it will not come to the surface to feed
    """
    prior_probabilities = np.array([0.5, 0.5])
    transition_probabilities = np.array([[0.7, 0.3],
                                            [0.3, 0.7]])
    emission_probabilities = np.array([[0.9, 0.1],
                                        [0.2, 0.8]])
    observation_states = ['no', 'yes']
    hidden_states = ['water', 'air']
    decode_observation_states = ['humminbir is in the water', 'humminbird is in the air']

    hmm = HiddenMarkovModel(observation_states = observation_states,
                            hidden_states = hidden_states,
                            prior_probabilities = prior_probabilities,
                            transition_probabilities = transition_probabilities,
                            emission_probabilities= emission_probabilities)
    
    viterbi = ViterbiAlgorithm(hmm)
    decoded_hidden_states = viterbi.best_hidden_state_sequence(decode_observation_states)
    assert np.alltrue(decoded_hidden_states == ['water', 'air'])

     


def test_user_case_three():
    """_summary_ does the grizzly bear leave hybernation in April or May?
    hypothesis: the grizzly bear leaves hybernation in May
    """
    prior_probabilities = np.array([0.5, 0.5])
    transition_probabilities = np.array([[0.7, 0.3],
                                            [0.3, 0.7]])
    emission_probabilities = np.array([[0.9, 0.1],
                                        [0.2, 0.8]])
    observation_states = ['hungry', 'not hungry']
    hidden_states = ['hybernation', 'awake']
    decode_observation_states = ['grizzly bear is hungry', 'grizzly bear is not hungry']
    correct_hidden_states = ['hybernation', 'hybernation','awake','awake']

    hmm = HiddenMarkovModel(observation_states = observation_states,
                            hidden_states = hidden_states,
                            prior_probabilities = prior_probabilities,
                            transition_probabilities = transition_probabilities,
                            emission_probabilities= emission_probabilities)
    
    viterbi = ViterbiAlgorithm(hmm)
    decoded_hidden_states = viterbi.best_hidden_state_sequence(decode_observation_states)
    assert np.alltrue(decoded_hidden_states == correct_hidden_states)

    