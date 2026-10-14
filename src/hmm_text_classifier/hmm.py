import numpy as np
from hmmlearn import hmm
from pathlib import Path
import json
import random
from collections import Counter, defaultdict


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

def getDistribution(words):
    d = defaultdict(int)
    for word in words:
        d[word] +=  1
    
    return d

def convertToNestedIntegerArray(emissions_sequences):
    word_list = [word for sequence in emissions_sequences for word in sequence]
    frequencies = getDistribution(word_list)
    dictionary = list(set(word_list))
    
    # Used to help with OOV words
    map = {}
    map['<UNK>'] = 0
    ctr = 1
    for i in range(1,len(dictionary) + 1):
        if frequencies[dictionary[i-1]] < 2 and random.choice([0,1]) == 1:
            map[dictionary[i-1]] = 0
        else:
            map[dictionary[i-1]] = ctr
            ctr += 1
        
    res = []
    for sequence in emissions_sequences:
        for word in sequence:
            res.append(map[word])
            
    
    res = np.array(res).reshape(-1,1)
    return res, dictionary

def trainHmmModel(transition_matrix, emissions_sequences):

    lengths = [len(sequence) for sequence in emissions_sequences]
    emissions_sequence_ints, dictionary = convertToNestedIntegerArray(emissions_sequences=emissions_sequences)
    
    model = hmm.CategoricalHMM(
    n_components=2,
    n_features=len(dictionary) + 1,
    init_params="e",
    params="e",
    )

    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = transition_matrix
    model.fit(emissions_sequence_ints, lengths=lengths)


emissions, sequences = getTrainingSet()
transition_matrix = calculateTransitionProbabilities(sequences)
trainHmmModel(transition_matrix=transition_matrix, emissions_sequences=emissions)

