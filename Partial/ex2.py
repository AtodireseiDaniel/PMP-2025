import numpy as np
import matplotlib as plt
from hmmlearn import hmm

from lab5.ex1 import target_sequence

tests=['Walking','Running','Resting']

n_tests=len(tests)


start_probability=[0.4,0.3,0.3]

transition_probability=[[0.6,0.3,0.1],
                        [0.2,0.7,0.1],
                        [0.3,0.2,0.5]]

emission_probability=[[0.1,0.7,0.2],
                      [0.05,0.25,0.7],
                      [0.8,0.15,0.05]]


model = hmm.CategoricalHMM(n_components=n_tests)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

observations=[1,2,0]
probability=model.predict(observations)

print(f"b:{probability}")


viterbih = model.predict(observations)
state_labels = ['Walking', 'Running', 'Resting']
most_likely_states = [state_labels[state] for state in viterbih]
print(f"c : {most_likely_states}")


generated_sequences = model.sample(10000)[0]
generated = generated_sequences[:, 0]

num_sequences = 10000
generated_sequences = model.sample(num_sequences)[0]
generated_observations = generated_sequences[:, 0]

target_sequence = [1, 2, 0]
count = sum(np.all(generated_observations[i:i+3] == target_sequence) for i in range(num_sequences - 3))
prob= count / (num_sequences - 3)

print(f"d:{prob}")






