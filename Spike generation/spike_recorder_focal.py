"""
load the dataset example and return the maximum image size, which is used to definite the spike generation network;
images with different size are focused onto the center of the spike generation network;
the generated poisson spikes are recorded and saved for further use.
"""
"""
on 12th November
by xiaoquinNUDT
version 0.0
"""
"""
test: no
"""
"""
optimization record:
"""
##-----------------------------------------------------------------------------------------
## module import
##-----------------------------------------------------------------------------------------
import brian2 as b2
from brian2 import *
import numpy as np
import cPickle as pickle 
import os
import sys 
from struct import unpack 
np.set_printoptions(threshold = np.inf)
##-----------------------------------------------------------------------------------------
## code generation device setup 
##-----------------------------------------------------------------------------------------
b2.defaultclock.dt = 0.2*b2.ms
b2.core.default_float_dtype = float64 ### reconsider
b2.core.default_integer_dtype = int16  ### retest
codegen.target = 'cython' # default 'auto', other setting include numpy, weave, cython
#clear_cache('cython') #clear the disk cache manually, or use the clear_cache function
codegen.cpp_compiler = 'gcc'
codegen.cpp_extra_compile_args_gcc = ['-ffast-math -march=native']
## Cython runtime codegen preferences
''' 
Location of the cache directory for Cython files. By default, 
will be stored in a brian_extensions subdirectory 
where Cython inline stores its temporary files (the result of get_cython_cache_dir()).
'''
codegen.runtime_cython_cache_dir = None
codegen.runtime_cython_delete_source_files = True
codegen.runtime_cython_multiprocess_safe = True
##-----------------------------------------------------------------------------------------
## self-definition method 
##-----------------------------------------------------------------------------------------
def get_dataset_example_mnist(path_dataset, name_dataset, using_test_dataset):
    """
    read input images (vector), dump into 
    '.pickle' format for next load, and return it as a numpy array.
    """
    flag_dataloaded = 0
    if name_dataset != 'mnist_test_example' and name_dataset != 'mnist_train_example':
        raise Exception('You have provide the wrong dataset name or path, please check carefully')
    else:
        dataset_path_name = path_dataset + name_dataset
        if os.path.isfile('%s.pickle' % dataset_path_name): 
            example = pickle.load(open('%s.pickle' % dataset_path_name))
            flag_dataloaded = 1
        else:
            flag_datasetsource = os.path.isfile(path_dataset+'train-images.idx3-ubyte') & \
                              os.path.isfile(path_dataset+'train-labels.idx1-ubyte') & \
                              os.path.isfile(path_dataset+'t10k-images.idx3-ubyte') & \
                              os.path.isfile(path_dataset+'t10k-labels.idx1-ubyte')
            if flag_datasetsource == False:
                raise Exception("You haven't downloaded the dataset into the %s!" % path_dataset)
            else:
                if using_test_dataset:
                    image = open(path_dataset+'t10k-images.idx3-ubyte', 'rb')
                else:
                    image = open(path_dataset+'train-images.idx3-ubyte', 'rb')
               # get metadata for images 
                image.read(4) # skip the magic number
                num_image = unpack('>I', image.read(4))[0]
                height_image = unpack('>I', image.read(4))[0]
                length_image = unpack('>I', image.read(4))[0]
                example = np.zeros((num_image, height_image, length_image), dtype = np.uint8)
                for i in xrange(num_image):
                    example[i] = [[unpack('>B', image.read(1))[0] for m in xrange(length_image)] for n in xrange(height_image)]
                pickle.dump(example, open('%s.pickle' % dataset_path_name, 'wb'))
                # the dataset has been readed and processed
                flag_dataloaded = 1
    if flag_dataloaded == 0:
        raise Exception('Failed to load the required dataset, please check the name_dataset and other printed information!')
    else:
        return example
## file system 
path_dataset = '../dataset_mnist/'
spike_record_path = './'
## input parameter
using_test_dataset = bool(int(sys.argv[1]))
print(using_test_dataset)
num_example = int(sys.argv[2])
print(num_example)
num_iteration = int(sys.argv[3])
print(num_iteration)

height_receptive_field = 28
length_receptive_field = 28

if using_test_dataset:
    num_per_dataset = 10000
    name_dataset = 'mnist_test_example'
    name_spike_record = 'mnist_spike_record_test'
else:
    num_per_dataset = 60000
    name_dataset = 'mnist_train_example'
    name_spike_record = 'mnist_spike_record_train'
## network setting parameters
input_intensity = 2.0
population_IN = height_receptive_field * length_receptive_field
working_time = 350 * b2.ms
resting_time = 150 * b2.ms
neuron_group_record = {}
spike_monitor_record = {}
name_neuron_group = 'Poisson_spike'
## create input poisson spike train  
neuron_group_record[name_neuron_group] = b2.PoissonGroup(population_IN, 0*Hz)
spike_monitor_record[name_neuron_group] = b2.SpikeMonitor(neuron_group_record[name_neuron_group])

network_record = b2.Network()
for obj_sim in [neuron_group_record, spike_monitor_record]:
    for key in obj_sim:
        network_record.add(obj_sim[key])
## dataset loading and record the input poisson spike
input_example = get_dataset_example_mnist(path_dataset, name_dataset, using_test_dataset) 
number_example = 0 
while number_example < num_example:
    input_image = input_example[(number_example + num_iteration * num_example) % num_per_dataset]
    height_example, length_example = input_image.shape
    length_margin = int((length_receptive_field - length_example)/2)
    height_margin = int((height_receptive_field - height_example)/2)
    input_rate = np.zeros((height_receptive_field, length_receptive_field), dtype = np.float32)
    for i in xrange(height_example):
        for j in xrange(length_example):
            input_rate[i + height_margin, j + length_margin] = input_image[i, j]
    neuron_group_record[name_neuron_group].rates = input_rate.flatten() / 8.0 * input_intensity * Hz
    network_record.run(working_time, report = 'text')
    neuron_group_record[name_neuron_group].rates = 0*Hz
    network_record.run(resting_time)
    number_example += 1
spike_index = np.asarray(spike_monitor_record[name_neuron_group].i, dtype = np.int16)
spike_time = np.asarray(spike_monitor_record[name_neuron_group].t, dtype = np.float64)
if using_test_dataset:
    spike_record_path_name = spike_record_path + name_spike_record + '_' + str(num_example)
else:    
    spike_record_path_name = spike_record_path + name_spike_record + '_' + str(num_example) + '_' + str(num_iteration)
file_spike_record = open('%s.pickle' % spike_record_path_name, 'wb')
pickle.dump(spike_index, file_spike_record)
pickle.dump(spike_time, file_spike_record)
file_spike_record.close()
