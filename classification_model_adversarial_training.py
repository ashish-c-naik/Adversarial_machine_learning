import network.network as network
import network.mnist_loader as mnist
import pickle

# Open the mnist dataset
training_data, validation_data, test_data = mnist.load_data()

# Open the adversarial dataset that we created using the "adversarial_dataset" file
with open('adversarial_samples_training_set.pkl', 'rb') as f:
	adversarial_dataset = pickle.load(f, encoding="latin1")

# Train FNN using the adversarial samples along with the normal training dataset.
net2 = network.Network([784,30,10])
net2.SGD(adversarial_dataset + training_data, 100, 5, 0.1)
filename = 'trained_adversarial.pkl'
pickle.dump(net2, open(filename, 'wb'))