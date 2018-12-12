import numpy as np 
      
# Each row is a training example, each column is a feature  [X1, X2, X3]
X = np.array(([.2,.32,.44], [.1,.23,.36], [.1,.47,.84], [.1, .34, .58]), dtype = float)
y = np.array(([.12], [.13], [.37], [.24]), dtype = float)
xPredicted = np.array(([.1,.42,.74]), dtype = float)

# X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input data for prediction)
# X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
# y = np.array(([92], [86], [89]), dtype=float)
# xPredicted = np.array(([4,8]), dtype=float)

# # scale units
# X = X/np.amax(X, axis=0) # maximum of X array
# xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
# y = y/100 # max test score is 100

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 3
    self.outputSize = 1
    self.hiddenSize = 4

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self):
    print("Predicted data based on trained weights: ");
    print("Input (scaled): \n" + str(xPredicted));
    print("Output: \n" + str(self.forward(xPredicted)));

NN = Neural_Network()
for i in range(1000000): # trains the NN 1,000 times
  if i % 100000 == 0:
    print("# " + str(i) + "\n")
    print("Input (scaled): \n" + str(X))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" + str(NN.forward(X)))
    print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
    print("\n")
  NN.train(X, y)

NN.saveWeights()
NN.predict()

# scale units
# x = x/np.amax(x, axis=0) # maximum of X array
# xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
# y = y/100 # max test score is 100

# # Define useful functions

# # Activation function
# def sigmoid(t):
# 	return 1 / (1 + np.exp(-t))

# # Derivative of sigmoid
# def sigmoid_derivative(p):
# 	return p * (1 - p)

# # Class definition
# class NeuralNetwork:
# 	def __init__(self, x, y):
# 		self.nodes = 3
# 		self.input = x
# 		self.weights1 = np.random.rand(self.input.shape[1], self.nodes) # considering we have 4 nodes in the hidden layer
# 		self.weights2 = np.random.rand(self.nodes, 1)
# 		self.y = y
# 		self.output = np.zeros(y.shape)

# 	def feedforward(self, x):
# 		self.layer1 = sigmoid(np.dot(x, self.weights1))
# 		self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
# 		return self.layer2

# 	def backprop(self):
# 		d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output)*sigmoid_derivative(self.output))
# 		d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
		
# 		self.weights1 += d_weights1
# 		self.weights2 += d_weights2

# 	def train(self, x, y):
# 		self.output = self.feedforward(x)
# 		self.backprop()

# 	def saveWeights(self):
# 		np.savetxt("w1.txt", self.weights1, fmt="%s")
# 		np.savetxt("w2.txt", self.weights2, fmt="%s")

# 	def predict(self):
# 		print("Predicted data based on trained weights: ")
# 		print("Input: \n" + str(xPredicted))
# 		print("Output: \n" + str(self.feedforward(xPredicted)))

# neuralNet = NeuralNetwork(x, y)
# for i in range(1000000): # trains the NN 1,000 times
# 	if i % 100000 == 0: 
# 		print("for iteration # " + str(i) + "\n")
# 		print("Input : \n" + str(x))
# 		print("Actual Output: \n" + str(y))
# 		print("Predicted Output: \n" + str(neuralNet.feedforward(x)))
# 		print("Loss: \n" + str(np.mean(np.square(y - neuralNet.feedforward(x))))) # mean sum squared loss
# 		print("\n")

# 	neuralNet.train(x, y)

# neuralNet.saveWeights()
# neuralNet.predict()