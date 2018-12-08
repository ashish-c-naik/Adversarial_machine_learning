import network.network as network
import network.mnist_loader as mnist
import pickle

# Load data
training_data, validation_data, test_data = mnist.load_data()


# Train the Network
net = network.Network([784,30,10])
net.SGD(training_data, 100, 5, 0.1, validation_data)
filename = 'FNN_model.pkl'
pickle.dump(net, open(filename, 'wb'))