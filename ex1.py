import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import networkx as nx


tests = np.array(['dificult', 'medium', 'easy'])
n_tests = len(tests)

start_probability = np.array([(1/3), (1/3), (1/3)])

transition_probability = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
])

emission_probability = np.array([
    [0.1, 0.2, 0.4, 0.3],
    [0.15, 0.25, 0.5, 0.1],
    [0.2, 0.3, 0.4, 0.1]
])

model = hmm.CategoricalHMM(n_components=n_tests)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

obs_map = {'FB': 0, 'B': 1, 'S': 2, 'NS': 3}
note = ['FB', 'FB', 'S', 'B', 'B', 'S', 'B', 'B', 'NS', 'B', 'B']
observation_sequence = np.array([obs_map[grade] for grade in note]).reshape(-1, 1)

#B
log_prob = model.score(observation_sequence)
print(f"probabilitatea secventei observate e {np.exp(log_prob)}")

#C


logprob_viterbi, hidden_states = model.decode(observation_sequence, algorithm="viterbi")
decoded_states = [str(tests[s])  for  s in hidden_states]
print(f"cea mai frecventa e {decoded_states}")
print(f"Probabilitatea acestei secvențe: {np.exp(logprob_viterbi)}")
