import network.network as Network
import network.mnist_loader as mnist_loader
import pickle

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)



net = Network.Network([784,30,10])
net.SGD(training_data, 50, 5, 0.1, test_data)
filename = 'trained_using_SGD.pkl'
pickle.dump(net, open(filename, 'wb'))
