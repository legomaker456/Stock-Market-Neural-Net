import numpy as np
from datetime import datetime
import os
      
# Each row is a training example, each column is a feature  [X1, X2, X3]
X = np.array(([.2,1.19,2.18], [.1,0.64,1.18], [.2,0.37,0.54], [.1,0.3,0.5], [.2,0.44,0.68], [.1,0.2,0.3], [.2,0.95,1.7], [.1,1.01,1.92], [.2,0.31,0.42]), dtype = float)
y = np.array(([0.99], [0.54], [0.17], [0.2], [0.24], [0.1], [0.75], [0.91], [0.11]), dtype = float)
xPredicted = np.array(([.23,.35,.47]), dtype = float)

class Neural_Network(object):
	def __init__(self):
		# parameters
		self.inputSize = 3
		self.outputSize = 1
		self.hiddenSize = 100

		# weights
		if(os.stat("w1.txt").st_size == 0 or os.stat("w2.txt") == 0): 
			self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
			self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
		self.W1 = np.loadtxt("w1.txt")
		self.W1.resize(self.inputSize, self.hiddenSize)
		self.W2 = np.loadtxt("w2.txt")
		self.W2.resize(self.hiddenSize, self.outputSize)

	def forward(self, X):
		self.z = np.dot(X, self.W1)
		self.z2 = self.sigmoid(self.z)
		self.z3 = np.dot(self.z2, self.W2)
		o = self.sigmoid(self.z3)
		return o

	def sigmoid(self, s):
		return 1/(1+np.exp(-s))

	def sigmoidPrime(self, s):
		return s * (1 - s)

	def backward(self, X, y, o):
		self.o_error = y - o
		self.o_delta = self.o_error*self.sigmoidPrime(o)

		self.z2_error = self.o_delta.dot(self.W2.T)
		self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

		self.W1 += X.T.dot(self.z2_delta)
		self.W2 += self.z2.T.dot(self.o_delta)

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

def convert_to_epoch(date):
	print(datetime.strptime(date, "%Y/%m/%d").timestamp())

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
convert_to_epoch("2018/11/13")