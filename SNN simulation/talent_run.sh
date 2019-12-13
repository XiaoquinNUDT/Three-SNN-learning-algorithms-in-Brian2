#!/bin/bash
echo '#: simulation_name'
read simulation_name
archive_path='./simulation_archive'
archive_folder="${archive_path}/${simulation_name}"
if [ ! -d "${archive_folder}" ] ; then
    mkdir ${archive_folder}
fi
echo "#: the simulation archive subfolder has been created!"
echo '#: python_script_name'
read python_script_name
cp $python_script_name $archive_folder
echo "#: the python script file has been copied to the created folder!"
echo '#: population_OUT'
read population_OUT
echo '#: num_example_training'
read num_example_training
echo '#: num_example_testing'
read num_example_testing
echo '#: num_per_evaluation'
read num_per_evaluation
echo '#: num_iteration'
read num_iteration
echo '#: start_iteration'
read start_iteration
log_file="${archive_folder}/log.txt"
if [ ! -f "$log_file" ]; then
touch $log_file
fi
echo "#: the simulation logfile has been created!"
echo "simulation_name python_script_name population_OUT" >> $log_file 
echo ''$simulation_name' '$python_script_name' '$population_OUT'' >> $log_file 
for ((i=$start_iteration;i<$num_iteration;i++)) 
do
    iteration_folder="$archive_folder/iteration_$i"
    result_folder="$iteration_folder/result"
    if [ ! -d "${iteration_folder}" ] ; then
        mkdir ${iteration_folder}
        mkdir ${result_folder}
    else
        rm -r ${iteration_folder}
        mkdir ${iteration_folder}
        mkdir ${result_folder}
    fi
    echo '#--------------------------------------------------------------------#'
    echo "###: run the ${i}th training iteration:"
    python "$archive_folder/$python_script_name" $population_OUT 0 $num_example_training $archive_folder $i
    ./file_rename.sh
    mv "./result/dynamic_array_spikemonitor_i_" $result_folder
    mv "./result/dynamic_array_spikemonitor_t_" $result_folder
    python "./evaluation/accuracy_evaluation.py" $population_OUT 0 $num_example_training $archive_folder $i $num_per_evaluation 
    python "./evaluation/weight_visualization.py" 'IN' 'OUT' 784 $population_OUT $archive_folder $i 
    python "./evaluation/spike_counter.py" $num_example_training $population_OUT $archive_folder $i
    
    echo "###: run the ${i}th testing iteration:"
    python "$archive_folder/$python_script_name" $population_OUT 1 $num_example_testing $archive_folder $i
    ./file_rename.sh
    mv "./result/dynamic_array_spikemonitor_i_" "$result_folder/dynamic_array_spikemonitor_i_t"
    mv "./result/dynamic_array_spikemonitor_t_" "$result_folder/dynamic_array_spikemonitor_t_t"
    python "./evaluation/accuracy_evaluation.py" $population_OUT 1 $num_example_testing $archive_folder $i $num_per_evaluation
done
echo ''$simulation_name' '$python_script_name' '$population_OUT' '$num_example_training' '$num_example_testing''
