import numpy as np
from datetime import datetime
import os
import csv
from sklearn.preprocessing import MinMaxScaler

# Conversion to epoch
def convert_to_epoch(date):
	return datetime.strptime(date, "%Y/%m/%d").timestamp()

# Data handling
with open("appl.csv", 'r') as file:
	reader = csv.reader(file)

	data = list(reader)
	data = [r for r in data if r != data[0] and r != data[1]]
	data = list(reversed(data))
	epoch_list = list()
	price_list = list()
	for i in range(0, len(data)):
		data[i][0] = convert_to_epoch(data[i][0])
		epoch_list.append([data[i][0]])
		price_list.append([data[i][1]])

# Each row is a training example, each column is a feature  [X1, X2, X3]
x = np.array((epoch_list), dtype = float)
y = np.array((price_list), dtype = float)

# Scale inputs
input_scaler = MinMaxScaler()
x = input_scaler.fit_transform(x)

# Scale outputs
output_scaler = MinMaxScaler()
y = output_scaler.fit_transform(y)

xPredicted = np.array((convert_to_epoch("2018/12/28")), dtype = float)
xPredicted *= float(input_scaler.scale_)

class Neural_Network(object):
	def __init__(self):
		# parameters
		self.input_layer_size = 1
		self.output_layer_size = 1
		self.hidden_layer_1_size = 1024
		self.hidden_layer_2_size = 512
		self.hidden_layer_3_size = 256
		self.hidden_layer_4_size = 128
		self.loss = None
		self.learning_rate = None

		# weights
		np.random.seed(0)
		self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_1_size)
		self.W2 = np.random.randn(self.hidden_layer_1_size, self.hidden_layer_2_size)
		self.W3 = np.random.randn(self.hidden_layer_2_size, self.hidden_layer_3_size)
		self.W4 = np.random.randn(self.hidden_layer_3_size, self.hidden_layer_4_size)
		self.W5 = np.random.randn(self.hidden_layer_4_size, self.output_layer_size)
		self.b1 = 0
		self.b2 = 0
		self.b3 = 0
		self.b4 = 0
		self.b5 = 0

	def forward(self, x):
		self.z1 = np.dot(x, self.W1) + self.b1
		self.a1 = self.sigmoid(self.z1)
		self.z2 = np.dot(self.a1, self.W2) + self.b2
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W3) + self.b3
		self.a3 = self.sigmoid(self.z3)
		self.z4 = np.dot(self.a3, self.W4) + self.b4
		self.a4 = self.sigmoid(self.z4)
		self.z5 = np.dot(self.a4, self.W5) + self.b5
		a5 = self.sigmoid(self.z5)
		return a5

	def sigmoid(self, s):
		return 1 / (1 + np.exp(-s))

	def sigmoidPrime(self, s):
		return s * (1 - s)

	def relu(self, s):
		s[s < 0] = 0
		return s

	def reluPrime(self, s):
		s[s < 0] = 0
		s[s > 0] = 1
		return s

	def backward(self, x, y, a5):
		m = y.shape[0]

		self.dz5 = y - a5
		self.dW5 = (1 / m) * (self.a4.T).dot(self.dz5)
		self.db5 = (1 / m) * np.sum(self.dz5, axis=0)

		self.dz4 = np.multiply(self.dz5.dot(self.W5.T), self.sigmoidPrime(self.a4))
		self.dW4 = (1 / m) * (self.a3.T).dot(self.dz4)
		self.db4 = (1 / m) * np.sum(self.dz4, axis=0)
		
		self.dz3 = np.multiply(self.dz4.dot(self.W4.T), self.sigmoidPrime(self.a3))
		self.dW3 = (1 / m) * (self.a2.T).dot(self.dz3)
		self.db3 = (1 / m) * np.sum(self.dz3, axis=0)
		
		self.dz2 = np.multiply(self.dz3.dot(self.W3.T), self.sigmoidPrime(self.a2))
		self.dW2 = (1 / m) * (self.a1.T).dot(self.dz2)
		self.db2 = (1 / m) * np.sum(self.dz2, axis=0)
		
		self.dz1 = np.multiply(self.dz2.dot(self.W2.T), self.sigmoidPrime(self.a1))
		self.dW1 = (1 / m) * (x.T).dot(self.dz1)
		self.db1 = (1 / m) * np.sum(self.dz1, axis=0)

		self.W1 += self.learning_rate * self.dW1 
		self.b1 += self.learning_rate * self.db1
		self.W2 += self.learning_rate * self.dW2
		self.b2 += self.learning_rate * self.db2
		self.W3 += self.learning_rate * self.dW3
		self.b3 += self.learning_rate * self.db3
		self.W4 += self.learning_rate * self.dW4
		self.b4 += self.learning_rate * self.db4
		self.W5 += self.learning_rate * self.dW5
		self.b5 += self.learning_rate * self.db5

	def train(self, x, y):
		o = self.forward(x)
		self.backward(x, y, o)

	def saveWeights(self):
		np.savetxt("w1.txt", self.W1, fmt="%s")
		np.savetxt("w2.txt", self.W2, fmt="%s")

	def predict(self):
		print("Predicted data based on trained weights: ")
		print("Input (scaled): \n" + str(xPredicted / input_scaler.scale_))
		print("Output: \n" + str(self.forward(xPredicted) / output_scaler.scale_))

	def fit(self, x, y, epochs, batch_size, learning_rate=1e-3):
		self.learning_rate = learning_rate

		for i in range(epochs):
			seed = np.arange(x.shape[0])
			np.random.shuffle(seed)
			x_ = x[seed]
			y_ = y[seed]
			for j in range(x.shape[0]):
				k = j * batch_size
				l = (j + 1) * batch_size
				NN.train(x_[k:l], y_[k:l])

			if (i + 1) % 10 == 0:
				print("Loss: \n" + str(np.mean(np.square(y - NN.forward(x))))) # mean sum squared loss

		
NN = Neural_Network()
NN.learning_rate = 1
	# if i % 100 == 0:
	# 	print("# " + str(i) + "\n")
	# 	print("Input (scaled): \n" + str(x_))
	# 	print("Actual Output: \n" + str(y_))
	# 	print("Predicted Output: \n" + str(NN.forward(x_)))
	# 	print("\n")
# NN.fit(x, y, 100, 15, 1e-3)
for i in range(1000):
	if i % 100 == 0:
		print("Loss: \n" + str(np.mean(np.square(y - NN.forward(x))))) # mean sum squared loss

	NN.train(x, y)

NN.saveWeights()
NN.predict()