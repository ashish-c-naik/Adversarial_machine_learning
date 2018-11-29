import network.network as Network
import network.mnist_loader as mnist_loader
import pickle
import matplotlib.pyplot as plt
import numpy as np

"""
50000 - Adversarial training samples
10000 - Adversarial test samples

"""

with open('trained_using_SGD.pkl', 'rb') as f:
	net = pickle.load(f, encoding="latin1")

# with open('trained_adversarial.pkl', 'rb') as f:
#     net2 = pickle.load(f, encoding="latin1")

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)

def gen_hot_vec(test_data):
    hotvec_test_data = []
    for x in test_data:
        hot_vector = np.zeros((10,1))
        hot_vector[x[1]] = 1
        hot_vector = np.expand_dims(hot_vector, axis=1)
        hotvec_test_data.append([x[0], hot_vector])
    return hotvec_test_data

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
                                                                                                                                                                                
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def predict(net, a):
	"""Return the output of the network if ``a`` is input."""
    #a = binary_thresholding(a)
    #plt.imshow(a.reshape(28, 28), cmap= "Greys")
    #plt.show()
	FNN = net.feedforward(a)
	FNN = np.round(FNN, 2)
	return FNN

def binary_thresholding(x):
    # Binarize image
    y = (x > .5).astype(float)
    return y

def input_derivative(net, x, y):
    #Calculate derivatives wrt the inputs
    nabla_b = [np.zeros(b.shape) for b in net.biases]
    nabla_w = [np.zeros(w.shape) for w in net.weights]
    #feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(net.biases, net.weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)    
    # backward pass
    delta = net.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    for l in range(2, net.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(net.weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())   
    # Return derivatives WRT to input
    return net.weights[0].T.dot(delta) #Why this?

def sneaky_adversarial(net, n, x_target, steps, eta, lam=0.3):
    """
    net : network, object neural network instance to use
    n : integer, our goal label (just an int, the function transforms it into a one-hot vector)
    x_target : numpy vector, our goal image for the adversarial example
    steps : integer, number of steps for gradient descent
    eta : float, step size for gradient descent
    lam : float lambda, our regularization parameter. Default is .05
    """
    #Set the goal output
    goal = np.zeros((10, 1))
    goal[n] = 1
    #Create a random image to initialize gradient descent with
    x = np.random.normal(.5, .3, (784, 1))
    #Gradient descent on the input
    for i in range(steps):
        #Calculate the derivative
        d = input_derivative(net,x,goal)
        #The GD update on x, with an added penalty to the cost function
        x -= eta * (d + lam * (x - x_target))
    return x

#Wrapper function
def sneaky_generate(n, m):
    """
    n: int 0-9, the target number to match
    m: index of example image to use (from the test set)
    """
    #Find random instance of m in test set
    idx = np.random.randint(0,8000)
    while test_data[idx][1] != m:
        idx += 1
    #Hardcode the parameters for the wrapper function
    a = sneaky_adversarial(net, n, test_data[idx][0], 100, 1)
    return a

def generate(count):
    adversarial_dataset = []
    for i in range(count):
        for j in range(10):
            for k in range(10):
                if j!=k:
                    a = sneaky_generate(j,k)
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
                    a = sneaky_adversarial(net, j, training_data[idx][0], 100, 1)
                    hot_vector = np.zeros((10,1))
                    hot_vector[k] = 1.
                    adversarial_dataset.append((a,hot_vector)) 
    return adversarial_dataset

def accuracy(net, test_data):
    count = 0
    for x in range(len(test_data)):
        if np.argmax(test_data[x][1]) == np.argmax(np.round(net.feedforward(test_data[x][0]), 2)):
            count += 1
    return (count/float(len(test_data)))*100

# Global Code
# a = sneaky_generate(0,3) 
# a = test_data[2][0]
# plt.imshow(a.reshape(28,28), cmap='Greys')
# plt.show() 
# p = predict(net,a)
# p = predict(net2,a)
# print('Network output: \n'+ str(p))
# print('Network prediction: '+ str(np.argmax(p)))

# adversarial_test_set = generate(112)
# new_test_set = adversarial_test_set + gen_hot_vec(test_data)
# print('Accuracy of attack on untrained FNN: ' + str(100 - accuracy(net, adversarial_test_set)))
# print('Accuracy of FNN on hybrid(adversarial+non-adversarial) test set without adversarial training: ' + str(accuracy(net, new_test_set)) +'%')
# print('Accuracy of FNN on hybrid(adversarial+non-adversarial) test set with adversarial training: ' + str(accuracy(net2, new_test_se  t)) + '%')

# adversarial_samples_test_set = generate(112)
# fname = 'adversarial_samples_test_set.pkl'
# pickle.dump(adversarial_samples_test_set, open(fname, 'wb'))