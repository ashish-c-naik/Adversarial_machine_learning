import network.network as Network
import network.mnist_loader as mnist_loader
import pickle
import loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)
with open('adversarial_samples_training_set.pkl', 'rb') as f:
	adversarial_dataset = pickle.load(f, encoding="latin1")

# net = Network.Network([784,30,10])
# net.SGD(training_data, 50, 5, 0.1, test_data)
# filename = 'trained_using_SGD.pkl'
# pickle.dump(net, open(filename, 'wb'))

net2 = Network.Network([784,30,10])
net2.SGD(adversarial_dataset + training_data, 30, 10, 3.0, test_data=test_data)
filename = 'trained_adversarial.pkl'
pickle.dump(net2, open(filename, 'wb'))

