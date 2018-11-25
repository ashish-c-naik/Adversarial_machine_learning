import matplotlib.pyplot as plt
import numpy as np
import loader

def generate(count):
	aaugmented = []
    
    for i in range(n):
        # Progress "bar"
        if i % 500 == 0:
            print("Generated digits: " + str(i))
            
        # Randomly choose a digit that the example will look like
        rnd_actual_digit = np.random.randint(10)
        
        # Find random instance of rnd_actual_digit in the training set
        rnd_actual_idx = np.random.randint(len(data))
        while np.argmax(data[rnd_actual_idx][1]) != rnd_actual_digit:
            rnd_actual_idx = np.random.randint(len(data))
        x_target = data[rnd_actual_idx][0]
        
        # Choose value for adversarial attack
        rnd_fake_digit = np.random.randint(10)
        
        # Generate adversarial example
        x_adversarial = sneaky_adversarial(net, rnd_fake_digit, x_target, steps, 1)
        
        # Add new data
        y_actual = data[rnd_actual_idx][1]
        
        augmented.append((x_adversarial, y_actual))
        
    return augmented