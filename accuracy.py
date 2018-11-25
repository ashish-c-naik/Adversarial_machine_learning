import numpy as np


def accuracy(net, test_data):
	count = 0
	for x in range(len(test_data)):
		if np.argmax(test_data[x][1]) == np.argmax(np.round(net.feedforward(test_data[x][0]), 2)):
			count += 1
	return count/float(len(test_data))
