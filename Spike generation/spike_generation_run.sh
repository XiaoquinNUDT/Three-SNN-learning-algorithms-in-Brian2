#!/bin/bash
echo '#: python_script_name'
read python_script_name
echo '#: using test dataset'
read using_test_dataset
echo '#: num_per_iteration'
read num_per_iteration
echo '#: num_iteration'
read num_iteration
echo '#: start_iteration'
read start_iteration

for ((i=$start_iteration;i<$num_iteration;i++)) 
do
    echo '#--------------------------------------------------------------------#'
    echo "###: run the ${i}th simulation iteration:"
    python "$python_script_name" $using_test_dataset $num_per_iteration $i 
done
