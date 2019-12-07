import numpy as np
from enum import Enum
NUM_OF_PARAMETERS = 10
class nameToVal(Enum):
    pass
# for MLP
""" I've come to a realization that we may not actually have enough data to use an MLP.
    So I'm developing the single layer first.
"""
# We need a numerical representation of these string values.
def prefixToVal(self, name):
    if(name == 'Mr' or name == 'Ms'):
        weight = 1
    elif(name == 'Mrs'):
        weight = 2
    elif(name == 'Upper'):
        weight = 3
    elif(name == 'Royalty'):
        weight = 4
    else:
        weight = 0
        print("prefixToVal: NO PREFIX")
    return weight

def sexToVal(self, sex):
    if(sex == 'male'):
        weight = 1
    elif(sex == 'female'):
        weight = 2
    else:
        weight = 0
        print("sexToVal: NO GENDER")
   
    return weight

def originToVal(self, origin):
    if(origin == 'C'):
        weight = 1
    elif(origin == 'S'):
        weight = 2
    elif(origin == 'Q'):
        weight = 3
    else:
        weight = 0
        print("originToVal: NO ORIGIN CITY")

class multiLayer(object):
    inputNodes = [] # 1-D array of inputs, supplied from a function in main
    in_to_hid_weight = [] # 1-D array of the weights from the input nodes --> hidden nodes
    hiddenNodes = [] # k-D array of hidden nodes, lets start with 1 layer for now
    hid_to_out_weight = [] # k-D array of weights from the hidden nodes --> output nodes
    survived = 0 # 1 == yes, 0 == no
    hiddenLayers = 1

    def __init__(self, numOfHiddenLayers):
        self.hiddenLayers = numOfHiddenLayers

    def sigmoid(self, value):
        return (1/(1+np.exp(-1*val)))

    # for single layer perceptron
class singleLayer(object):
    inputNodes = [] # 1-D arr of inputs, supplied from main
    weights = [] # 1-D arr of weights
    survivedLabel = 0 # 1 == yes, 0 == no
    prediction = 0 # 1 == survived, 0 == dead forever

    def signFunction(self, value):
        if value > 0.5:
            return 1
        else:
            return 0

    def sigmoid(self, value):
        return (1/(1+np.exp(-1*val)))

    def readPassenger(survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked):
        # thats 10 inputs plus a label. Big arg list, sorry.
        self.inputNodes = [] ## ALWAYS reset our list
        self.survivedLabel = survived # assign label

        self.inputNodes.append(pclass)
        self.inputNodes.append(prefixToVal(name))
        self.inputNodes.append(sexToVal(sex))
        self.inputNodes.append(age)  
        self.inputNodes.append(parch) # num of siblings aboard Titanic
        self.inputNodes.append(0) # <-- We need to find a way to represent the tickets numerically
        self.inputNodes.append(fare)
        self.inputNodes.append(originToVal(embarked))
        
        #validate input
        for i in range(len(self.inputNodes)):
            print(self.inputNodes[i], "\n")

        





