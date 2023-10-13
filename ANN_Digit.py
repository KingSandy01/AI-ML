# -*- coding: utf-8 -*-
"""
Prolem statement : 
    Given patterns of digits in binary form
    as training data, design an ANN that 
    can  recognize the label of a given pattern.

"""
# We import library numpy

import numpy as np

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


# For repeatibility
np.random.seed(0)

#input features or patterns 
Xinp = np.array((zero, one, two, three, four, five, six, seven,eight,nine)) 

#output labels
yout = np.array(([0], [1], [2], [3],[4],[5], [6],[7], [8], [9])) 

#(our input data for which the label is to be predicted)
XPredicted = np.array((predict)) 


# scale units
yout = yout/10 # max test score is 100


# Define class Artificial_Neural_Network

class Artificial_Neural_Network():
    
    
#Randomly initialize weights by using the function np.random.randn 
    
  def __init__(self):
      
    #Define the size of the layers
    self.inputLayerSize = 20
    self.outputLayerSize = 1
    self.hiddenLayerSize = 20

    # Initialize the weight matrix from input to hidden layer with random numbers
    self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize) 
    
    # Initialize the weight matrix from hidden to output layer
    self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize) 
    
# Define sigmoid activation function    
  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

# Define  the derivative of the sigmoid activation function  
  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)


#forward propagation through our network
    
  def forward_pass(self, Xinp):
     
    # dot product of Xinp (input) and first set of input weights  W1
    self.L1 = np.dot(Xinp, self.W1) 
    
    # Apply the activation function
    self.L2 = self.sigmoid(self.L1) 
    
    # dot product of hidden layer (z2) and second set of weights
    self.L3 = np.dot(self.L2, self.W2) 
    
    # Applying final activation function at the output layer
    o = self.sigmoid(self.L3) 
    
    
    #return the output
    return o

  

# backward propgate through the network  
  def backward_pass(self, Xinp, y, o):
    
      
    # calculate the error from the output  
    self.output_error = yout - o 
    
    # applying derivative of sigmoid to error
    self.output_e_delta = self.output_error*self.sigmoidPrime(o) 
    
    # multiplying delta error with the weights in the hidden layer
    #i.e. L2 error: how much our hidden layer weights contributed to output error
    self.L2_error = self.output_e_delta.dot(self.W2.T) 
    
    # applying derivative of sigmoid to L2 error
    self.L2_e_delta = self.L2_error*self.sigmoidPrime(self.L2) 
    
    # adjusting first set (input --> hidden) weights
    self.W1 = self.W1 + Xinp.T.dot(self.L2_e_delta) 
    
    # adjusting second set (hidden --> output) weights
    self.W2 = self.W2 + self.L2.T.dot(self.output_e_delta) 
    
    
# Train the network with forward and backward pass
  def train_ANN(self, Xinp, yout):
    o = self.forward_pass(Xinp)
    self.backward_pass(Xinp, yout, o)
    
    
# Make predictions 
    
  def make_prediction(self):
    print ("Predicted data based on trained weights: ")
    print ("Input: \n" + str(XPredicted.reshape(5,4)))
    print ("Actual Output: \n" + str((self.forward_pass(XPredicted))*10))
    print ("Rounded Output: \n" + str(np.round((self.forward_pass(XPredicted))*10)))


# Make an object of the class
ANN = Artificial_Neural_Network()

# Iterate 5000 times to train the network with the training data
# Print the loss

for i in range(5000): # trains the NN 5000 times
  ANN.train_ANN(Xinp, yout)
  print ("Loss: \n" + str(np.mean(np.square(yout - ANN.forward_pass(Xinp)))))


# Make predictions  
ANN.make_prediction()


