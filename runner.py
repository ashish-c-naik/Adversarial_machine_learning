# Before running this file run:
# 
# 1) classification_model.py
# 2) adversarial_dataset.py
# 3) classification_mode_adversarial_training.py
# 
# in sequence

import network.mnist_loader as mnist
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import util as u

confusion_matrix = [[0 for x in range(10)] for y in range(10)]
training_data, validation_data, test_data = mnist.load_data()


with open('FNN_model.pkl', 'rb') as f:
    net = pkl.load(f, encoding="latin1")

with open('FNN_adversarial.pkl', 'rb') as f:
    net2 = pkl.load(f, encoding="latin1")

with open('adversarial_samples_test_set.pkl', 'rb') as f:
    adversarial_test_set = pkl.load(f, encoding="latin1")
new_test_set = adversarial_test_set + gen_hot_vec(test_data)


# for x in confusion_matrix:
    # print(x)  
print('Accuracy of untrained FNN on non-adversarial test set: ' + str(u.accuracy(net, u.gen_hot_vec(test_data))) +'%')
print(" ")
print('Accuracy of attack on untrained FNN: ' + str(100 - u.accuracy(net, adversarial_test_set)) +'%')
print(" ")
print('Accuracy of untrained FNN on hybrid(adversarial+non-adversarial) test set: ' + str(u.accuracy(net, new_test_set)) +'%')
print(" ")
print('Accuracy of attack on trained FNN: ' + str(100 - u.accuracy(net2, adversarial_test_set)) +'%')
print(" ")
print('Accuracy of trained FNN on hybrid(adversarial+non-adversarial) test set: ' + str(u.accuracy(net2, new_test_set)) + '%')
print(" ")
print('Accuracy of untrained FNN on hybrid(adversarial+non-adversarial) test set with binary thresholding: ' + str(u.accuracy(net, new_test_set,1)) + "%")
print(" ")
print('Accuracy of trained FNN on hybrid(adversarial+non-adversarial) test set with binary thresholding: ' + str(u.accuracy(net2, new_test_set,1)) + "%")
