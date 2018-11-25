import network.network as Network
import network.mnist_loader as mnist_loader
import pickle
import adversarial_dataset

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)
adversarial_dataset = adversarial_dataset.generate(556)
net2 = Network.Network([784,30,10])
net2.SGD(adversarial_dataset + training_data, 30, 10, 3.0, test_data=test_data)
filename = 'trained_adversarial.pkl'
pickle.dump(net2, open(filename, 'wb'))

