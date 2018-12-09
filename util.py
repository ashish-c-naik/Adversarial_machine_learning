import network.mnist_loader as mnist
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

with open('FNN_model.pkl', 'rb') as f:
    net = pkl.load(f, encoding="latin1")

FNN_adversarial = my_file = Path("FNN_adversarial.pkl")
if FNN_adversarial.is_file():
    with open('FNN_adversarial.pkl', 'rb') as f:
        net2 = pkl.load(f, encoding="latin1")

training_data, validation_data, test_data = mnist.load_data()

def gen_hot_vec(test_data):
    hotvec_test_data = []
    for x in test_data:
        hot_vector = np.zeros((10,1))
        hot_vector[x[1]] = 1
        hot_vector = np.expand_dims(hot_vector, axis=1)
        hotvec_test_data.append([x[0], hot_vector])
    return hotvec_test_data

def predict_binary_thresholding(net, x):
    y = (x > .5).astype(float)
    return predict(net, y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
                                                                                                                                                                                
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def predict(net, a):
    """Return the output of the network if `a` is the input."""
    FNN = net.feedforward(a)
    FNN = np.round(FNN, 2)
    return FNN


def input_derivative(net, x, y):
    #Calculate and return derivatives w.r.t the inputs
    _b = [np.zeros(b.shape) for b in net.biases]
    _w = [np.zeros(w.shape) for w in net.weights]
    #feedforward
    activation = x
    activations = [x] 
    zs = []
    for b, w in zip(net.biases, net.weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)    
    # backward pass
    delta = net.error(activations[-1], y) * sigmoid_prime(zs[-1])
    _b[-1] = delta
    _w[-1] = np.dot(delta, activations[-2].transpose())
    for l in range(2, net.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(net.weights[-l+1].transpose(), delta) * sp
        _b[-l] = delta
        _w[-l] = np.dot(delta, activations[-l-1].transpose())   
    return net.weights[0].T.dot(delta) 

def _adversarial(net, p, target, steps, eta, lam=0.05):
    """
    net : network; neural network instance to use
    p : integer; the label advarsarial example should be predicted with 
    target : vector; the image the adversarial example should look like
    steps : integer; number of steps for gradient descent
    eta : float; step size for gradient descent
    lam : float; regularization parameter 
    """
    goal = np.zeros((10, 1))
    goal[p] = 1
    x = np.random.normal(.5, .3, (784, 1)) #Create a random image to initialize gradient descent with
    for i in range(steps): #Gradient descent on the input
        d = input_derivative(net,x,goal) #Calculate the derivative
        x -= eta * (d + lam * (x - target)) #GD update on x, with an added penalty to the cost function  
    return x

def _generate(p, a):
    """
    p: integer; the label advarsarial example should be predicted with 
    a: integer; the label advarsarial example actually has 
    """
    i = np.random.randint(0,8000)
    while test_data[i][1] != a:
        i += 1
    image = _adversarial(net, p, test_data[i][0], 100, 1)
    return image

def generate(count):
    adversarial_dataset = []
    for i in range(count):
        for j in range(10):
            for k in range(10):
                if j!=k:
                    a = _generate(j,k)
                    hot_vector = np.zeros((10,1))
                    hot_vector[k] = 1.
                    adversarial_dataset.append((a,hot_vector)) 
    return adversarial_dataset

def generate_using_training_set(count):
    adversarial_dataset = []
    for i in range(count):
        for j in range(10):
            for k in range(10):
                if j!=k:
                    idx = np.random.randint(0,8000)
                    while np.argmax(training_data[idx][1]) != k:
                        idx += 1
                    a = _adversarial(net, j, training_data[idx][0], 100, 1)
                    hot_vector = np.zeros((10,1))
                    hot_vector[k] = 1.
                    adversarial_dataset.append((a,hot_vector)) 
    return adversarial_dataset

def accuracy(net, test_data, binary=0):
    count = 0

    for x in range(len(test_data)):
        if np.argmax(test_data[x][1]) == np.argmax((predict(net, test_data[x][0]), predict_binary_thresholding(net, test_data[x][0]))[binary==1]):
            count += 1
    return (count/float(len(test_data)))*100