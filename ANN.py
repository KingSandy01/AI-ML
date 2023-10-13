# We import library numpy

import numpy as np

"""
Prolem statement : 
    Given patterns of digits in binary form
    as training data, design an ANN that 
    can  recognize the label of a given pattern.

"""

# Write the patterns from 0 to one represented by binary digits

zero = [
  1, 1, 1, 1,
  1, 0, 0, 1,
  1, 0, 0, 1,
  1, 0, 0, 1,
  1, 1, 1, 1
]

one = [
  0, 0, 1, 0,
  0, 0, 1, 0,
  0, 0, 1, 0,
  0, 0, 1, 0,
  0, 0, 1, 0
]

two = [
  1, 1, 1, 1,
  0, 0, 0, 1,
  1, 1, 1, 1,
  1, 0, 0, 0,
  1, 1, 1, 1
]

three = [
  1, 1, 1, 1,
  0, 0, 0, 1,
  1, 1, 1, 1,
  0, 0, 0, 1,
  1, 1, 1, 1
]

four = [
  1, 0, 0, 1,
  1, 0, 0, 1,
  1, 1, 1, 1,
  0, 0, 0, 1,
  0, 0, 0, 1
]

five = [
  1, 1, 1, 1,
  1, 0, 0, 0,
  1, 1, 1, 1,
  0, 0, 0, 1,
  1, 1, 1, 1
]

six = [
  1, 1, 1, 1,
  1, 0, 0, 0,
  1, 1, 1, 1,
  1, 0, 0, 1,
  1, 1, 1, 1
]

seven = [
  1, 1, 1, 1,
  0, 0, 0, 1,
  0, 0, 0, 1,
  0, 0, 0, 1,
  0, 0, 0, 1
]

eight = [
  1, 1, 1, 1,
  1, 0, 0, 1,
  1, 1, 1, 1,
  1, 0, 0, 1,
  1, 1, 1, 1
]

nine = [
  1, 1, 1, 1,
  1, 0, 0, 1,
  1, 1, 1, 1,
  0, 0, 0, 1,
  1, 1, 1, 1
]

################################
#Number to be predicted = 8

predict = [
  1, 1, 1, 1,
  1, 0, 0, 1,
  1, 1, 1, 1,
  1, 0, 0, 1,
  1, 1, 1, 1
]

###############################

Xinp = np.array((zero, one, two, three, four, five, six, seven, eight, nine,))
yout = np.array(([0],[1],[2],[3],[4],[5],[6],[7],[8],[9]))
XPredicted = np.array((predict))

yout = yout/10

class Artificial_Neural_Network():

    def __init__(self):
        self.inputLayerSize = 20
        self.outputLayerSize = 1
        self.hiddenLayerSize = 20

        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
    

    def        