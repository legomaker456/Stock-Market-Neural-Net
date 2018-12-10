import numpy as np

def sigmoid(t):
	return 1 / (1 + np.exp(-t))

def sigmoid_derivative(p):
	return p * (1 - p)

def tanh(t):
	return np.tanh(t)

def tanh_derivative(p):
	return 1.0 - np.tanh(p) ** 2