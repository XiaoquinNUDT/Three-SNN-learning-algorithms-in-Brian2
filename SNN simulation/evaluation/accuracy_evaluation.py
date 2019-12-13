"""
This code is for neuron labeling and performance evaluation by processing the records of SpikeMonitor. 
The spike records are loaded from the '../result' folder copied from the spike record in the code generation mdoe.
"""
"""
on 16th November
by xiaoquinNUDT
version 0.0
"""
"""
test: no
"""
"""
optimazation record:
revised on 30th November
"""
##-----------------------------------------------------------------------------------------
import numpy as np
# np.fromfile() np.where() np.argsort() np.zeros() np.sum()
import matplotlib.pyplot as plt
# plt.figure() plt.title() plt.plot()
import sys
# sys.argv[]
import os
# os.path.isfile()
import cPickle as pickle
# pickle.load() pickle.dump()
from struct import unpack
# pack and unpack
import gc 
# gc.collect() 
import time
# time.time()
import math
# math.exp()
np.set_printoptions(threshold = np.inf)
##-----------------------------------------------------------------------------------------
def Load_dataset_label(path_dataset, name_dataset_label, using_test_dataset):
    """read input class labels (0-9), dump  into 
       '.pickle' format for next loading, and return it as a list of tuples.
       name_dataset_label: the defined name for the pickle file of label in this code
       path_dataset: the relative path where the dataset and the reprocessed dataset are saved
    """
    if name_dataset_label == 'mnist_label_test' or name_dataset_label == 'mnist_label_train':
        dataset_path_name = path_dataset + name_dataset_label 
        if os.path.isfile('%s.pickle' % dataset_path_name): 
            dataset_label = pickle.load(open('%s.pickle' % dataset_path_name))
        else:
            if using_test_dataset:
                label = open(path_dataset+'t10k-labels.idx1-ubyte', 'rb')
            else:
                label = open(path_dataset+'train-labels.idx1-ubyte', 'rb')
            # get metadata for labels 
            label.read(4) # skip the magic number
            num_label = unpack('>I', label.read(4))[0]
            dataset_label = np.zeros((num_label, 1), dtype = np.uint8)
            for i in xrange(num_label):
                dataset_label[i] = np.uint8(unpack('>B', label.read(1))[0])
            pickle.dump(dataset_label, open('%s.pickle' % dataset_path_name, 'wb'))
    else:        
        raise Exception('Failed to load the required dataset, please check the name_dataset and other printed information!')
    return dataset_label.flatten()
def Update_assignment_score(spike_counter, input_label_interval, population_OUT, class_number):
    """ calculate the probabilistic of one neuron spiking for specific class.
        assignment_score: the probabilistic of every neuron for every class
        spike_counter_interval: the number of spikes of every output neuron during update_interval
        dataset_label: labels of examples from dataset
        output_neuron_num: the number of output neurons
        class_number: the number of class
    """
    input_example_label = np.asarray(input_label_interval)
    num_spike_interval = np.zeros((population_OUT, class_number), dtype = np.int)
    assignment_score = np.zeros((population_OUT, class_number), dtype = np.float)
    for i in xrange(class_number):
        class_index = np.where(input_example_label == i)[0]
        if len(class_index) == 0:
            raise Exception("There is no input of label %s" % j)
    for k in xrange(population_OUT):
        for m in xrange(class_number):
            class_index = np.where(input_example_label == m)[0]
            num_spike_interval[k, m] = np.sum(spike_counter[class_index, k])
        spike_sum = float(np.sum(num_spike_interval[k,:])) + 0.01
        assignment_score[k] = num_spike_interval[k]/spike_sum
        #assignment_score[k] = np.exp(num_spike_interval[k]/spike_sum*10)
    return assignment_score 
def Reference_label_of_input(spike_counter_one, assignment_score, population_OUT, class_number):
    """ calculate the label ranking from network spike records.
    """
    assignment_total_score = np.zeros(class_number, dtype = np.float)
    for i in xrange(population_OUT):
        assignment_total_score += spike_counter_one[i] * assignment_score[i]
    #print(assignment_total_score)
    label = np.where(assignment_total_score == np.max(assignment_total_score))[0][0]
    return label 
def Classification_accuracy(performance, current_example_number, dataset_label, output_label, update_interval):
    """ calculate the performance according to the difference between input labels and evaluated labels.
        performance: index by inferrence accuracy for each update interval
        dataset_label: labels of input examples
        output_label: network output label 
    """
        
    current_evaluation = current_example_number/update_interval - 1
    start_example_number = current_example_number - update_interval
    end_example_number = current_example_number

    difference = output_label[start_example_number:end_example_number] - dataset_label[start_example_number: end_example_number]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct/float(update_interval)*100
    return performance

def Plot_accuracy(performance, time_step, fig_number):
    fig = plt.figure(fig_number, figsize=(5,5))
    fig_number += 1
    plt.plot(time_step, performance)
    plt.ylim(ymax=100)
    plt.title('Classification performance')

##-----------------------------------------------------------------------------------------
#test_mode = bool(int(raw_input("Test mode: ")))
#using_test_dataset = bool(int(raw_input("Using test dataset: ")))
#num_example = int(raw_input("The number of example: "))
#update_interval = int(raw_input("The number of update interval: "))
#population_OUT = int(raw_input("The number of output neuron: "))
population_OUT = int(sys.argv[1])
test_mode = bool(int(sys.argv[2]))
using_test_dataset = test_mode
num_example = int(sys.argv[3])
archive_folder = str(sys.argv[4])
number_iteration = int(sys.argv[5])
update_interval = int(sys.argv[6])
iteration_folder = archive_folder + '/iteration_' + str(number_iteration)
class_number = 10
path_dataset = '../dataset_mnist/'
path_spike_record = iteration_folder + '/result' # relative to the location of talent_run.sh
num_evaluation = int(num_example/update_interval)
time_step = range(0, num_evaluation) 
if using_test_dataset:
    name_dataset_label = 'mnist_label_test'
    num_per_dataset = 10000
else:
    name_dataset_label = 'mnist_label_train'
    num_per_dataset = 60000
dataset_label_source = Load_dataset_label(path_dataset, name_dataset_label, using_test_dataset) 
if test_mode:
    dataset_label = dataset_label_source
else:
    if ((number_iteration + 1) * num_example) % num_per_dataset == 0:
        dataset_label = dataset_label_source[(number_iteration * num_example) % num_per_dataset : num_per_dataset]
    else:
        dataset_label = dataset_label_source[(number_iteration * num_example) % num_per_dataset : ((number_iteration + 1) * num_example) % num_per_dataset]
performance = np.zeros(num_evaluation)
if test_mode:
    print('load saved assignment during training')
    assignment_score = np.load(iteration_folder + '/assignment'+ '_' + 'OUTe' + '_' + str(population_OUT) + '_' + str(number_iteration) + '.npy')
else:
    assignment_score = np.zeros((population_OUT, class_number), dtype = np.float)
if test_mode:
    spike_index_path_name = path_spike_record + '/dynamic_array_spikemonitor_i_t'
    spike_time_path_name = path_spike_record + '/dynamic_array_spikemonitor_t_t'
else:
    spike_index_path_name = path_spike_record + '/dynamic_array_spikemonitor_i_'
    spike_time_path_name = path_spike_record + '/dynamic_array_spikemonitor_t_'
spike_monitor_index = np.fromfile(spike_index_path_name, dtype = np.int32, count = -1, sep = "" )
spike_monitor_time = np.fromfile(spike_time_path_name, dtype = np.float64, count = -1, sep = "")
if len(spike_monitor_index) != len(spike_monitor_time):
    raise Exception('Spike records may not read properly!')
print('firing rate for per neuron in one 0.5s cycle:' + str(float(len(spike_monitor_index))/population_OUT/num_example))
spike_counter_interval = np.zeros((update_interval, population_OUT))
spike_counter_one = np.zeros(population_OUT)
output_label = np.zeros(num_example)
time_per_example = 0.5 
passed_time_interval = update_interval * time_per_example
passed_time_one = time_per_example
number_example = 0
for j, item in enumerate(spike_monitor_time):
    if item > passed_time_one:
        number_example = int(item/time_per_example) - 1
        spike_counter_interval[number_example % update_interval, :] = spike_counter_one 
        output_label[number_example] = Reference_label_of_input(spike_counter_one, assignment_score, population_OUT, class_number)
        spike_counter_one = np.zeros(population_OUT)
        passed_time_one += time_per_example
    spike_counter_one[spike_monitor_index[j]] += 1
    if item >= passed_time_interval and not test_mode:
        assignment_score = Update_assignment_score(spike_counter_interval, dataset_label[number_example - update_interval + 1 : number_example + 1], population_OUT, class_number)
    if item >= passed_time_interval:
        spike_counter_interval = np.zeros((update_interval, population_OUT))
        passed_time_interval += update_interval * time_per_example
        performance = Classification_accuracy(performance, number_example + 1, dataset_label, output_label, update_interval)
    if j == spike_monitor_time.size - 1:
        number_example = num_example - 1 
        spike_counter_interval[j % update_interval, :] = spike_counter_one 
        output_label[number_example] = Reference_label_of_input(spike_counter_one, assignment_score, population_OUT, class_number)
        assignment_score = Update_assignment_score(spike_counter_interval, dataset_label[number_example - update_interval + 1 : number_example + 1], population_OUT, class_number)
        performance = Classification_accuracy(performance, number_example + 1, dataset_label, output_label, update_interval)

if not test_mode:
    np.save(iteration_folder + '/assignment' + '_' + 'OUTe' + '_' + str(population_OUT) + '_' + str(number_iteration) + '.npy', assignment_score)
if test_mode:
    np.save(iteration_folder + '/accuracy' + '_' + 'test_' + str(population_OUT) + '_' + str(number_iteration) + '.npy', performance)
else:
    np.save(iteration_folder + '/accuracy' + '_' + str(population_OUT) + '_' + str(number_iteration) + '.npy', performance)

print 'Classification performance', performance 
print 'Best classification performance', np.max(performance)
if test_mode:
    print 'Average classification performance', np.sum(performance) / (num_evaluation)
else:
    print 'Average classification performance', np.sum(performance[1:]) / (num_evaluation - 1)
