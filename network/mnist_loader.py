import pickle
import gzip
import numpy as np

def load_data():
	"""
	Loads the data from the mnist.pkl.gz file
	1) Creates training data as a list of tuples of training values and labels. Before adding the training values
	as a tuple element it is reshaped in (784,1) numpy array. Also, before adding the training labels as a tuple
	element it is converted into one-hot vector by using hot_vector() function.
	2) Creates the testing and validation data as a list of tuples just like the training data. Just that the 
	labels are kept as single values instead of being a hot vector. 
	"""
	f = gzip.open('data/mnist.pkl.gz', 'rb')
	train, valid, test  = pickle.load(f, encoding='latin1')
	f.close()
	training = [[np.reshape(value[0], (784, 1)), hot_vector(value[1])] for value in zip(train[0], train[1])]
	validation = [[np.reshape(value[0], (784, 1)), value[1]] for value in zip(valid[0], valid[1])]
	testing = [[np.reshape(value[0], (784, 1)), value[1]] for value in zip(test[0], test[1])]
	return (training, validation, testing)

def hot_vector(j):
	"""
	This function converts the label into a one-hot vector with shape being (10, 1) with jth index 1.
	"""
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e