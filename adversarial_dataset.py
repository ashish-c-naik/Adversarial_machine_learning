import pickle
import loader

adversarial_samples_training_set = loader.generate_using_training_set(556)
fname = 'adversarial_samples_training_set.pkl'
pickle.dump(adversarial_samples_training_set, open(fname, 'wb'))