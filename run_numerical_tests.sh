#!/bin/bash

# Define eligible values for each parameter
seed=($(seq 1 2))

# limit for LP1, LP2 solving times
timelimit=(1.0 10.0)

# Test scenario
scenario=("ScenarioA" "ScenarioB" "ScenarioC" "ScenarioD")

# Loop over all possible combinations of parameter values
for t in "${timelimit[@]}"
do
   for s in "${scenario[@]}"
   do
      for d in "${seed[@]}"
      do
         # Generate the command for each combination of parameter values
         cmd="python3 NumericalTest.py"
         cmd+=" -s $s"
         cmd+=" -ss $d"
         cmd+=" -t $t"

         echo " "
         echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
         echo "cmd: $cmd"
         echo " "

         # Run the command "cmd"
         eval "$cmd"

      done
   done
done
