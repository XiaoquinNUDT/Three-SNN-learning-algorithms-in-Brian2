"""
This code is for synapse weight visualization.
The weight are loaded from the './output/result' folder generated in the code generation mdoe.
"""
"""
on 26th October
by xiaoquinNUDT
version 0.1
"""
"""
test: no
"""
"""
optimization record:
"""
##-----------------------------------------------------------------------------------------
import numpy as np 
# np.array, np.load, np.fromfile, np.copy, .reshape(), np.zeros()
import matplotlib.pyplot as plt 
# plt.figure, plt.imshow(), plt.colorbar(), plt.ion(), plt.ioff(), plt.savefig(), plt.get_cmap()
import sys
# sys.argv[]
##-----------------------------------------------------------------------------------------
def Get_2d_weight_source_first(weight, population_IN, population_OUT):
    """
    calculate the length and width of the 2D weight array for visualization.
    """
    weight_copy = np.copy(weight)
    sqrt_population_IN = int(np.sqrt(population_IN))
    sqrt_population_OUT = np.sqrt(population_OUT)
    if sqrt_population_OUT == int(sqrt_population_OUT):
        length_population_OUT = width_population_OUT = sqrt_population_OUT
    else:
        difference = population_OUT
        for i in xrange(population_OUT):
            if population_OUT % (i+1) == 0:
                factor_population_OUT_0 = i + 1
                factor_population_OUT_1 = population_OUT / (i + 1)
                if difference > abs(factor_population_OUT_0 - factor_population_OUT_1):
                    difference = abs(factor_population_OUT_0 - factor_population_OUT_1)
                    if factor_population_OUT_0 > factor_population_OUT_1:
                       length_population_OUT = factor_population_OUT_0
                       width_population_OUT = factor_population_OUT_1
                    else:
                       length_population_OUT = factor_population_OUT_1
                       width_population_OUT = factor_population_OUT_0
    aspect_ratio = length_population_OUT/width_population_OUT
    array_2d_length = sqrt_population_IN * int(length_population_OUT)
    array_2d_width = sqrt_population_IN * int(width_population_OUT)
    weight_2d_array = np.zeros((array_2d_width, array_2d_length))
    for i, item in enumerate(weight_copy):
        x = int((i % population_OUT) % length_population_OUT) * sqrt_population_IN + (i // population_OUT) % sqrt_population_IN
        y = int((i % population_OUT) // length_population_OUT) * sqrt_population_IN + (i // population_OUT) // sqrt_population_IN
        weight_2d_array[y,x] = item
    return weight_2d_array, aspect_ratio 
                        
def Plot_2d_weight(weight_2d_matrix, fig_number, savefig_name, aspect_ratio):
    fig =plt.figure(fig_number, figsize=(18,18/(aspect_ratio * 1.2)))
    im = plt.imshow(weight_2d_matrix, interpolation='nearest', vmin=0, vmax=wmax, cmap=plt.get_cmap('CMRmap_r')) 
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)
    plt.rcParams['font.size'] = 34
    plt.colorbar(im)
    #plt.title('Synapse weight matrix of %s' % connection_name)
    plt.savefig(savefig_name)
    #fig.canvas.draw() # used to update a figure that has been altered, but not automatically re-drawn
##-----------------------------------------------------------------------------------------
'''
main function
1. get the information of the saved weight: network structure; source and target layer name
2. transform the one-dimension wight array into 2-D visulization matrix 
3. plot the 2-D weight matrix
'''
name_IN = str(sys.argv[1])
name_OUT = str(sys.argv[2])
population_IN = int(sys.argv[3])
population_OUT = int(sys.argv[4])
wmax = 1.0 
archive_folder = str(sys.argv[5])
number_iteration = int(sys.argv[6])
iteration_folder = archive_folder + '/iteration_' + str(number_iteration)
#name_IN = raw_input("The name of source layer: ") 
#name_OUT = raw_input("The name of target layer: ")
#population_IN = int(raw_input("The number of neuron in source layer: "))
#population_OUT = int(raw_input("The number of neuron in target layer: "))
connection_name = name_IN + '_' + name_OUT + 'e'
'''
read weight from results saved at the end of the simulation 
'''
fig_number = 1
weight_path_name = iteration_folder + '/' + 'weight_' + connection_name + '_' + str(population_OUT) + '_' + str(number_iteration) + '.npy'
savefig_name = iteration_folder + '/' + 'weight_image_' + connection_name + '_' + str(population_OUT) + '_' + str(number_iteration) + '.eps'
weight_array = np.load(weight_path_name)
weight_2d_matrix, aspect_ratio = Get_2d_weight_source_first(weight_array, population_IN, population_OUT)
Plot_2d_weight(weight_2d_matrix, fig_number, savefig_name, aspect_ratio)
fig_number += 1
