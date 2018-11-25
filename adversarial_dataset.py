import matplotlib.pyplot as plt
import numpy as np
import loader

def generate(count):
	adversarial_dataset = []
	for i in range(count):
		if len(adversarial_dataset) % 1000 == 0:
			print("Generated ", len(adversarial_dataset), " points")
		for j in range(10):
			for k in range(10):
				if j!=k:
					a = loader.sneaky_generate(j,k)
					hot_vector = np.zeros((10,1))
					hot_vector[k] = 1.
					adversarial_dataset.append((a,hot_vector)) 
	return adversarial_dataset
