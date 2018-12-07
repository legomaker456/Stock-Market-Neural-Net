import numpy as np 
      
# Each row is a training example, each column is a feature  [X1, X2, X3]
x = np.array(([.2,.4,.6], [.1,.2,.3], [.1,.5,.9], [.1,.4,.7]), dtype = float)
y = np.array(([.2], [.1], [.4], [.3]), dtype = float)
xPredicted = np.array(([.1,.7,1.3]), dtype = float)

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
		self.weights3 = np.random.rand(1, 4)
		self.y = y
		self.output = np.zeros(y.shape)

	def feedforward(self, x):
		self.layer1 = sigmoid(np.dot(x, self.weights1))
		self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
		self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
		return self.layer3

	def backprop(self):
		d_weights3 = np.dot(self.layer2.T, 2 * (self.y - self.output)*sigmoid_derivative(self.output))
		d_weights2 = np.dot(self.layer1.T, np.dot(2 * (self.y - self.output)*sigmoid_derivative(self.output), self.weights3.T)*sigmoid_derivative(self.layer2))
		d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output)*sigmoid_derivative(self.output), self.weights2)*sigmoid_derivative(self.layer1))
		
		self.weights1 += d_weights1
		self.weights2 += d_weights2
		self.weights3 += d_weights3

	def train(self, x, y):
		self.output = self.feedforward(x)
		self.backprop()

	def saveWeights(self):
		np.savetxt("w1.txt", self.weights1, fmt="%s")
		np.savetxt("w2.txt", self.weights2, fmt="%s")

	def predict(self):
		print("Predicted data based on trained weights: ")
		print("Input: \n" + str(xPredicted))
		print("Output: \n" + str(self.feedforward(xPredicted)))

neuralNet = NeuralNetwork(x, y)
for i in range(10000): # trains the NN 1,000 times
	if i % 1000 == 0: 
		print("for iteration # " + str(i) + "\n")
		print("Input : \n" + str(x))
		print("Actual Output: \n" + str(y))
		print("Predicted Output: \n" + str(neuralNet.feedforward(x)))
		print("Loss: \n" + str(np.mean(np.square(y - neuralNet.feedforward(x))))) # mean sum squared loss
		print("\n")

	neuralNet.train(x, y)

neuralNet.saveWeights()
neuralNet.predict()