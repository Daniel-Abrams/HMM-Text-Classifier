import numpy as np
from hmmlearn import hmm
from pathlib import Path
import json
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def getTrainingSet():
    with open(PROJECT_ROOT / "data" / "training.json", 'r') as file:
        training_set = json.load(file)
        emissions = []
        state_tranistion_sequences = []
        for doc in training_set:
            emissions.append(doc['emission_list'])
            state_tranistion_sequences.append(np.array(doc['state_list']))
    
    return emissions, state_tranistion_sequences

# Calculate Transition Probabilities according to first order Markovian assumption,
# i.e. P(x_n | x_1,x_2,...,x_n-1) = P(x_n | x_n-1)
def calculateTransitionProbabilities(sequences):
    states = {"human": 0, "ai": 1}
    counts = np.zeros((2, 2))
    
    for sequence in sequences:
        for current, next_state in zip(sequence, sequence[1:]):
            counts[states[current], states[next_state]] += 1

    # Normalize each row to get transition probabilities
    transition_matrix = counts / counts.sum(axis=1, keepdims=True)
    return transition_matrix

def convertToNestedIntegerArray(emissions_sequences):
    dictionary = list(set([word for sequence in emissions_sequences for word in sequence]))
    map = {}
    map['<UNK>'] = 0
    for i in range(1,len(dictionary) + 1):
        map[dictionary[i-1]] = i
        
    res = []
    for sequence in emissions_sequences:
        for word in sequence:
            res.append(map[word])
            
    
    res = np.array(res).reshape(-1,1)
    return res, dictionary

def trainHmmModel(transition_matrix, emissions_sequences):

    lengths = [len(sequence) for sequence in emissions_sequences]
    emissions_sequence_ints, dictionary = convertToNestedIntegerArray(emissions_sequences=emissions_sequences)
    print(emissions_sequence_ints)
    
    model = hmm.CategoricalHMM(
    n_components=2,
    n_features=len(dictionary),
    init_params="e",
    params="e",
    )

    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = transition_matrix
    model.fit(emissions_sequence_ints, lengths=lengths)


emissions, sequences = getTrainingSet()
transition_matrix = calculateTransitionProbabilities(sequences)
trainHmmModel(transition_matrix=transition_matrix, emissions_sequences=emissions)

