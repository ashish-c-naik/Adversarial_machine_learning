import pickle
import loader

# 50000 - Adversarial training samples
# 10000 - Adversarial test samples

adversarial_samples_training_set = loader.generate_using_training_set(556)
fname = 'adversarial_samples_training_set.pkl'
pickle.dump(adversarial_samples_training_set, open(fname, 'wb'))

adversarial_samples_test_set = loader.generate(112)
fname = 'adversarial_samples_test_set.pkl'
pickle.dump(adversarial_samples_test_set, open(fname, 'wb'))