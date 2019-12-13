import numpy as np
import sys
import matplotlib.pyplot as plt
iteration = int(sys.argv[1])
population_OUT = int(sys.argv[2])
accuracy = np.zeros(iteration)
training_iteration = np.arange(1, iteration + 1, 1)
for i in xrange(iteration):
    accuracy_10 = np.load('iteration_' + str(i) + '/accuracy_test_' + str(population_OUT) + '_' + str(i) + '.npy')
    accuracy[i] = np.mean(accuracy_10)
np.save('./accuracy_test.npy', accuracy)
plt.figure(1)
plt.plot(training_iteration, accuracy, '.-')
plt.ylim(ymax=100)
plt.savefig('./accuracy_test.png')
plt.ioff()
plt.show()
