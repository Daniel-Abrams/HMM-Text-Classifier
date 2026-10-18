import pickle
import json
import numpy as np

from hmmlearn import hmm
from pathlib import Path

from bpetokenizer import BPETokenizer

RED = '\033[31m'
GREEN = '\033[32m'

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def getTrainSet():
    with open(PROJECT_ROOT / "data" / "train.json", 'r') as file:
        training_set = json.load(file)
        emissions = []
        state_tranistion_sequences = []
        for doc in training_set:
            emissions.append(doc['emission_list'])
            state_tranistion_sequences.append(np.array(doc['state_list']))
    
    return emissions, state_tranistion_sequences

def getTestSet():
    with open(PROJECT_ROOT / "data" / "test.json", 'r') as file:
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

def trainHmmModel(transition_matrix, emissions_sequences):

    lengths = [len(sequence) for sequence in emissions_sequences]
    emissions_sequence_ints = np.concatenate([np.reshape(np.array(seq), (-1,1)) for seq in emissions_sequences])
    
    model = hmm.CategoricalHMM(
    n_components=2,
    init_params="e",
    params="e",
    )

    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = transition_matrix
    model.fit(emissions_sequence_ints, lengths=lengths)
    return model

def saveModel(model, filename="hmm_model.pkl"):
    with open(filename, "wb") as file: 
        pickle.dump(model, file)

def loadModel(filename="hmm_model.pkl"):
    with open("hmm_model.pkl", "rb") as file: 
        model = pickle.load(file)
    return model

def predictHiddenSequences(model: hmm.CategoricalHMM, emission_sequences):
    emission_sequence_ints = np.concatenate([np.reshape(np.array(seq), (-1,1)) for seq in emission_sequences])
    state_sequences = model.predict(emission_sequence_ints)
    return state_sequences
    
def accuracy(pred,true):
    return np.mean(pred == true)

def vizualizePrediction(emissions, predictions):
    tokenizer = BPETokenizer()
    tokenizer.load("bpetokenizer.json", mode="json") 
    res = f""
    for i in range(len(emissions)):
        decoded = tokenizer.decode([emissions[i]])
        if predictions[i] == 1:
            res += f"{GREEN}{decoded}"
        else:
            res += f"{RED}{decoded}"
    return res


def Train():
    emissions,sequences = getTrainSet()
    mat = calculateTransitionProbabilities(sequences=sequences)
    model = trainHmmModel(emissions_sequences=emissions, transition_matrix=mat)
    saveModel(model=model)

def Test():
    emissions, sequences = getTestSet()
    model = loadModel()
    predictions = predictHiddenSequences(model,emission_sequences=emissions)
    encoded_sequences = np.array([int(x == "ai") for s in sequences for x in s])
    print(f"accuracy {accuracy(predictions,encoded_sequences)}")
    print(vizualizePrediction(np.concatenate([np.array(seq) for seq in emissions]), predictions))
    
Test()