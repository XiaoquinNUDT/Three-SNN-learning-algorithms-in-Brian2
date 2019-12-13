"""
This code is a reproduction of Dimien's design[1] of SNN implemented on Brian2.
[1] Querlioz D, Bichler O, Dollfus P, et al. Immunity to Device Variations in a Spiking Neural Network With Memristive Nanodevices[J]. IEEE Transactions on Nanotechnology, 2013, 12(3): 288-295.
There are only one difference between the repreduced design and the original one: the shape and duration of the spike.
"""
"""
on 7th April 2019
by xiaoquinNUDT
email: qulianhua14@nudt.edu.cn
"""
"""
version 0.0
"""
"""
unit test: 
"""
"""
optimization record:
"""
##-----------------------------------------------------------------------------------------
## module import
##-----------------------------------------------------------------------------------------
import brian2 as b2
from brian2 import *
# b2.NeuronGroup() b2.Synapses() b2.SpikeMonitor()
import numpy as np
# np.load() np.array() np.zeros() np.random()
import cPickle as pickle 
# pickle.load()
import os
# os.path.isfile()
import sys
# sys.argv[1]
##-----------------------------------------------------------------------------------------
## setting for print and plotting 
##-----------------------------------------------------------------------------------------
#np.set_printoptions(threshold=np.inf)
##-----------------------------------------------------------------------------------------
## standalone code generation mode setup
##-----------------------------------------------------------------------------------------
'''
for multiple run, At the beginning of the script, i.e. after the import statements, add:
set_device('cpp_standalone', build_on_run=False)
After the last run() call, call device.build() explicitly:
device.build(directory='output', compile=True, run=True, debug=False)
'''
set_device('cpp_standalone', directory = 'output', build_on_run = True, clean = True) 
## setup for C++ compilation preferences
codegen.cpp_compiler = 'gcc' #Compiler to use (uses default if empty);Should be gcc or msvc.
prefs.devices.cpp_standalone.extra_make_args_unix = ['-j']
#prefs.devices.cpp_standalone.openmp_threads = 1 
##-----------------------------------------------------------------------------------------
## basic simulation setup
##-----------------------------------------------------------------------------------------
b2.defaultclock.dt = 0.1 * b2.ms
b2.core.default_float_dtype = float32 ### can be chaged to float32 in brian 2.2
b2.core.default_integer_dtype = int32
##-----------------------------------------------------------------------------------------
## self-defined functions: mainly for the dataset input processing and output visualization
##-----------------------------------------------------------------------------------------
def Load_spike_record(path_spike_record, name_spike_record):
    """ 
       load the generated spike train
    """
    spike_record_path_name = path_spike_record + name_spike_record 
    print(spike_record_path_name)
    if os.path.isfile('%s.pickle' % spike_record_path_name): 
        file_spike_record = open('%s.pickle' % spike_record_path_name)
        spike_record_index = pickle.load(file_spike_record)
        spike_record_time = pickle.load(file_spike_record)
        file_spike_record.close()
    else:
        raise Exception("Spike record hasn't been created or saved in the right path")
    return spike_record_index, spike_record_time 
def Save_weight(weight, weight_save_name):
    print('Save the learned synapse weight')
    np.save(weight_save_name, weight) 
def Save_thresh(neuron_thresh, thresh_save_name):
    print('Save the thresh of neurons for homeostasis')
    np.save(thresh_save_name, neuron_thresh) 
##-----------------------------------------------------------------------------------------
## main 
##-----------------------------------------------------------------------------------------
## simulation control parameters
population_OUT = int(sys.argv[1]) 
test_mode = bool(int(sys.argv[2]))
if test_mode:
    STDP_on = False
    using_test_dataset = True
else:
    STDP_on = True
    using_test_dataset = False 
num_example = int(sys.argv[3]) # number of example for one iteration 
archive_folder = str(sys.argv[4])
number_iteration = int(sys.argv[5])
iteration_folder = archive_folder + '/iteration_' + str(number_iteration)
iteration_folder_last = archive_folder + '/iteration_' + str(number_iteration - 1)
np.random.seed(0)
##-----------------------------------------------------------------------------------------
## file system
##-----------------------------------------------------------------------------------------
path_input_spike_record = '../input_spike_train/'
if using_test_dataset:
    name_input_spike_record = 'mnist_spike_record_test' + '_' + str(num_example)
else:
    num_iteration = 60000 / num_example
    name_input_spike_record = 'mnist_spike_record_train' + '_' + str(num_example) + '_' + str(number_iteration % num_iteration)
##-----------------------------------------------------------------------------------------
## neuron and synapse dynamics definition
##-----------------------------------------------------------------------------------------
working_time = 350 * b2.ms
resting_time = 150 * b2.ms
## standard neuron parameters
v_rest_excite = 0 * b2.volt
v_rest_inhibite = 0 * b2.volt
v_reset_excite = 0 * b2.volt
v_reset_inhibite = 0 * b2.volt
v_thresh_excite = 0.5 * b2.volt
t_refrac_excite = 10.0 * b2.ms
t_refrac_inhibite = 10.0 * b2.ms
time_inhibite = 10.0 * b2.ms
gama = 1
## spike conditon
v_thresh_excite_str = '(v > (thresh)) and (timer > t_refrac_excite)'
v_thresh_inhibite_str = '(ge > 0) and (timer > time_inhibite)'
## excitatory neuron membrane dynamics 
neuron_eqs_excite = '''
        dv/dt = int(gi <= 0) * (-v + gama * ge * volt) / (100 * ms)  : volt (unless refractory)
        dge/dt = -ge / (1 * ms)             : 1
        dgi/dt = -0.1 / (1 * ms)            : 1
        '''
neuron_eqs_inhibite = '''
        ge                                   : 1 
        '''
## thresh damping dynamic 
tc_threshold = 4e7 * b2.ms # slow decay
neuron_eqs_excite += '\n  dthresh/dt = int(timer_c > 2*second) * ((spike_number / (timer_c + 10 *ms) - 0.2 * Hz) * second * volt) / (tc_threshold)  : volt'
neuron_eqs_excite += '\n  dtimer_c/dt = 1  : second'   
neuron_eqs_excite += '\n  spike_number  : 1'   
## time
neuron_eqs_excite += '\n  dtimer/dt = 1  : second'   
neuron_eqs_inhibite += '\n  dtimer/dt = 1  : second'   
## reset dynamics for excitatory and inhibitory neuron
reset_excite = 'v = v_reset_excite; timer = 0 * ms; ge = 0; gi = 0; spike_number += 1'
reset_inhibite = 'timer = 0 * ms; ge = 0' 
## synapse conductance model
model_synapse_base = 'w : 1'
spike_pre_excite = 'ge_post += w' 
spike_post_inhibite = 'gi_pre = 1; v_pre = v_reset_excite' 
## STDP hyperparameter
nu_learning_p = 1e-2 
nu_learning_d = 5e-3 
w_max = 1 
w_mid = 0.5
w_min = 0.0001 
exp_p = exp_d = 3 
pre_target = 0.1
nu_learning = 4e-2 
## STDP mechanism 
eqs_stdp = '''
                dpre/dt = -pre / (25*ms)         : 1 (clock-driven) 
            '''   
eqs_stdp_pre = 'pre = 1.' 
#eqs_stdp_post = 'w += nu_learning * sign(pre - pre_target) * w * (w_max -w); pre = 0' 
eqs_stdp_post = 'w += sign(pre - pre_target) * (int(pre > pre_target) * nu_learning_p * exp(-exp_p * (w - w_min) / (w_max - w_min)) +  int(pre < pre_target) * nu_learning_d * exp(-exp_d * (w_max - w) / (w_max - w_min))); pre = 0' 

##-----------------------------------------------------------------------------------------
## network structure definition
##-----------------------------------------------------------------------------------------
height_max_image = 28
length_max_image = 28
height_IN = height_max_image
length_IN = length_max_image
population_IN = height_IN * length_IN
##-----------------------------------------------------------------------------------------
## definite and create the neuron and synapse group 
##-----------------------------------------------------------------------------------------
neuron_group = {}
synapse = {}
spike_monitor = {} 
## create input poisson neuron group
name_neuron_group = 'IN'
input_spike_record_index, input_spike_record_time = Load_spike_record(path_input_spike_record, name_input_spike_record)
neuron_group[name_neuron_group] = SpikeGeneratorGroup(population_IN, input_spike_record_index, input_spike_record_time * b2.second)
#spike_monitor['IN'] = SpikeMonitor(neuron_group['IN'])
## create the output neuron group 
name_neuron_group = 'OUT'+ 'e' 
neuron_group[name_neuron_group] = NeuronGroup(population_OUT, neuron_eqs_excite, threshold = v_thresh_excite_str, refractory = t_refrac_excite, reset = reset_excite)
spike_monitor[name_neuron_group] = SpikeMonitor(neuron_group[name_neuron_group])
neuron_group[name_neuron_group].v = v_rest_excite
neuron_group[name_neuron_group].timer = 0 * b2.ms
neuron_group[name_neuron_group].timer_c = 0 * b2.ms
neuron_group[name_neuron_group].spike_number = 0
neuron_group[name_neuron_group].ge = 0 
neuron_group[name_neuron_group].gi = 0 
if STDP_on:
    if number_iteration == 0:
        neuron_group[name_neuron_group].thresh = v_thresh_excite
    else:
        neuron_group[name_neuron_group].thresh = np.load(iteration_folder_last + '/thresh_' + name_neuron_group + '_' + str(population_OUT) + '_' + str(number_iteration - 1) + '.npy') * b2.volt 
else:
    print('load the saved thresh')
    neuron_group[name_neuron_group].thresh = np.load(iteration_folder + '/thresh_' + name_neuron_group + '_' + str(population_OUT) + '_' + str(number_iteration) + '.npy') * b2.volt 
## create the inhibite neuron group 
name_neuron_group = 'OUT' + 'i' 
neuron_group[name_neuron_group] = NeuronGroup(1, neuron_eqs_inhibite, threshold = v_thresh_inhibite_str, refractory = t_refrac_inhibite, reset = reset_inhibite)
#spike_monitor[name_neuron_group] = SpikeMonitor(neuron_group[name_neuron_group])
neuron_group[name_neuron_group].timer = 0 * b2.ms
neuron_group[name_neuron_group].ge = 0 
##-----------------------------------------------------------------------------------------
## create the synapse from 'IN' to 'OUT'
##-----------------------------------------------------------------------------------------
connection_name = 'IN_OUTe'
if STDP_on:
    model_synapse_ee = model_synapse_base + eqs_stdp
    on_pre_ee = spike_pre_excite + ';' + eqs_stdp_pre
    on_post_ee = eqs_stdp_post
else:
    model_synapse_ee = model_synapse_base
    on_pre_ee = spike_pre_excite 
    on_post_ee = ''
synapse[connection_name] = b2.Synapses(neuron_group['IN'], neuron_group['OUTe'], model = model_synapse_ee, on_pre = on_pre_ee, on_post = on_post_ee)
synapse[connection_name].connect()
if STDP_on: 
    if number_iteration == 0:
        synapse[connection_name].w = '(0.2 * randn() + 0.8) * w_mid'
    else:
        synapse[connection_name].w = np.load(iteration_folder_last + '/weight_' + connection_name + '_' + str(population_OUT) + '_' + str(number_iteration - 1) + '.npy') 
else:
    print('load the saved weight')
    synapse[connection_name].w = np.load(iteration_folder + '/weight_' + connection_name + '_' + str(population_OUT) + '_' + str(number_iteration) + '.npy') 
##-----------------------------------------------------------------------------------------
## create the synapse from 'OUTe' to 'OUTi'
##-----------------------------------------------------------------------------------------
connection_name = 'OUTe_OUTi'
model_synapse_ei = model_synapse_base
on_pre_ei = spike_pre_excite 
on_post_ei = spike_post_inhibite 
synapse[connection_name] = b2.Synapses(neuron_group['OUTe'], neuron_group['OUTi'], model = model_synapse_ei, on_pre = on_pre_ei, on_post = on_post_ei)
synapse[connection_name].connect()
synapse[connection_name].w = 1.0
##-----------------------------------------------------------------------------------------
## setup the network and run the simulations 
##-----------------------------------------------------------------------------------------
net = Network()
for obj_sim in [neuron_group, spike_monitor, synapse]:
    for key in obj_sim:
        net.add(obj_sim[key])
net.run(num_example * (working_time + resting_time), report = 'text', report_period = 1200 * second)
##-----------------------------------------------------------------------------------------
## save the learned weight and thresh 
##-----------------------------------------------------------------------------------------
if STDP_on:
    Save_weight(synapse['IN_OUTe'].w, iteration_folder + '/weight_IN_OUTe' + '_' + str(population_OUT) + '_' + str(number_iteration))
    Save_thresh(neuron_group['OUTe'].thresh, iteration_folder + '/thresh_OUTe' + '_' + str(population_OUT) + '_' + str(number_iteration))
