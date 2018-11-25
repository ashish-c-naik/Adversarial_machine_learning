import network.network as Network
import network.mnist_loader as mnist_loader
import pickle
import matplotlib.pyplot as plt
import numpy as np
import accuracy as ac

with open('trained_using_SGD.pkl', 'rb') as f:
	net = pickle.load(f, encoding="latin1")

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)

hybrid_test_data = []
for x in test_data:
    hot_vector = np.zeros((10,1))
    hot_vector[x[1]] = 1
    hybrid_test_data.append([x[0], hot_vector])

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
                                                                                                                                                                                
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def predict(a):
	"""Return the output of the network if ``a`` is input."""
	FNN = net.feedforward(a)
	FNN = np.round(FNN, 2)
	return FNN

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

def sneaky_adversarial(net, n, x_target, steps, eta, lam=0.05):
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
    for i in range(1):
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
    a = sneaky_adversarial(net, n, test_data[idx][0], 500, 1)
    return a

#Global Code
# a = sneaky_generate(0,4) 
# a = test_data[2][0]
#plt.imshow(a.reshape(28,28), cmap='Greys')
# plt.show()   
#print('Network output: \n'+ str(predict(a)))
#print('Network prediction: '+ str(np.argmax(predict(a)))) 

print('Accuracy: ' + str(ac.accuracy(net, hybrid_test_data)))