import random 
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        sizes is a list that contains the number of neurons in the respective layers of the network.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Return the output of the network if 'a' is input.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying mini-batch 
        stochastic gradient descent with backpropagation
        """
        _b = [np.zeros(b.shape) for b in self.biases]
        _w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            _b = [nb+dnb for nb, dnb in zip(_b, delta_b)]
            _w = [nw+dnw for nw, dnw in zip(_w, delta_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, _w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, _b)]

    def backprop(self, x, y):
        """
        Backpropagation procedure.
        """

        _b = [np.zeros(b.shape) for b in self.biases]
        _w = [np.zeros(w.shape) for w in self.weights]
        
        # Feed the inputs feed
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Backpropagation
        delta = self.error(activations[-1], y) * sigmoid_prime(zs[-1])
        _b[-1] = delta
        _w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            _b[-l] = delta
            _w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (_b, _w)

    def evaluate(self, test_data):
        """
        This function evaluates the network weights by comparing the outputs on the 
        test data.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def error(self, output_activations, y):
        return output_activations - y

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
