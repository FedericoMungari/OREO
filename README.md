# OREO
This repository contains the code for the O-RAN intelligence orchestrator described in the paper "OREO: O-RAN Intelligence Orchestration of xApp-based Network Services", presented at the IEEE International Conference on Computer Communications (INFOCOM) 2024, in Vancouver, BC, CA.

## Content
This repository contains the following files:
*   LICENSE
    
    This file contains the license covering the software and associated documentation files provided in this repository.

*   README.md
    
    This is the file that contains the instructions to use the provided software that you are currently reading.


*   run_numerical_tests.sh

    Bash script to automate tests (NumericalTest.py) for different orchestrator parameters, such as:
	- The testing scenario (look at the test_scenario.py function for more details);
	- The time limit (i.e., the solving time limit for each iteration of the Orchestration engine).
	
*   NumericalTest.py

    Main function.

*   test_scenario.py
	
	A Python script to generate random testing scenarios. Given some hyperparameters, such as the number of requested services, service configurations, RAN functions, etc., the script generates random settings. To orchestrate your RAN applications, you may fit their specifics into the test_scenario.py function.
	
*   Orchestrator folder
	
	The Orchestrator contains the OREO framework and additional functions:
	- LagrangianProblem.py: contains the (decomposed) Lagrangian xDeSh Problem;
	- lagrangian_multipliers.py: defines the Lagrangian penalty terms;
	- EnsuringFeasibility.py: contains heuristic solution ensuring feasible xDeSh solutions;
	- utils.py: additional auxiliary functions.
	
	Please refer to our "OREO: O-RAN Intelligence Orchestration of xApp-based Network Services" INFOCOM paper for further information.



## Instructions
To test our code, clone this repository using the following command:
```
git clone https://github.com/FedericoMungari/OREO.git
```
Then, run the Python script "NumericalTest.py" with the desired parameters, or run the "run_numerical_tests.sh" bash script.

## Cite us
If you have used our work in your research, please consider citing our paper (preprint version):

```
@INPROCEEDINGS{oreo_infocom2024,
  author={Mungari, Federico and Puligheddu, Corrado and Garcia-Saavedra, Andres and Chiasserini, Carla Fabiana},
  booktitle={IEEE INFOCOM 2024 - IEEE Conference on Computer Communications}, 
  title={OREO: O-RAN intElligence Orchestration of xApp-based network services}, 
  year={2024}}
```
