import numpy as np 
      
# Each row is a training example, each column is a feature  [X1, X2, X3]
x = np.array(([2,4,6], [1,2,3], [2,7,12], [1,4,7]), dtype = float)
y = np.array(([2], [1], [5], [3]), dtype = float)

# Define useful functions    

# Activation function
def sigmoid(t):
	return 1 / (1 + np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
	return p * (1 - p)

# Class definition
class NeuralNetwork:
	def __init__(self, x, y):
		self.input = x
		self.weights1 = np.random.rand(self.input.shape[1], 4) # considering we have 4 nodes in the hidden layer
		self.weights2 = np.random.rand(4, 1)
		self.y = y
		self.output = np.zeros(y.shape)

	def feedforward(self):
		self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
		return self.layer2

	def backprop(self):
		d_weights2 = np.dot(self.layer1.T, 4 * (self.y - self.output)*sigmoid_derivative(self.output))
		d_weights1 = np.dot(self.input.T, np.dot(4 * (self.y - self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))

		self.weights1 += d_weights1
		self.weights2 += d_weights2

	def train(self, X, y):
		self.output = self.feedforward()
		self.backprop()

neuralNet = NeuralNetwork(x, y)
for i in range(1000000): # trains the NN 1,000 times
	if i % 100000 == 0: 
		print("for iteration # " + str(i) + "\n")
		print("Input : \n" + str(x))
		print("Actual Output: \n" + str(y))
		print("Predicted Output: \n" + str(neuralNet.feedforward()))
		print("Loss: \n" + str(np.mean(np.square(y - neuralNet.feedforward())))) # mean sum squared loss
		print("\n")

	neuralNet.train(x, y)