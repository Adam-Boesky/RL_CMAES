import cma
import pandas as pd
import numpy as np
import subprocess
import os

COMMAND_TO_RUN_SIM = f"./sim/main"
CURRENT_TRAJECTORY = "straight"

# Running a simple command (e.g., 'ls' to list files in the current directory)
result = subprocess.run(COMMAND_TO_RUN_SIM, capture_output=True, text=True)

# Objective function
def objective_function(x):

    # 1. Run a trajectory with the sampled MVN in the CMA-ES (i.e. the input array x) --> do with command line
    subprocess.run(COMMAND_TO_RUN_SIM, capture_output=True, text=True)

    # 2. Read in the actual and target trajectory values generated during the previous run
    actual_trajectory = pd.read_csv("./sim_results/test.csv")
    target_trajectory = pd.read_csv(f"./trajectories/{CURRENT_TRAJECTORY}.csv")

    # 3. Calculate and return the loss values
    # TODO

# Initial mean and sigma values
initial_mean = [1.0, 1.0, 1.0]
sigma = 1

# Initialize the CMA-ES optimizer
es = cma.CMAEvolutionStrategy(initial_mean, sigma)

# Perform the optimization
es.optimize(objective_function)

# Retrieve the result
result = es.result
print("Optimized solution:", result[0])
print("Best value found:", result[1])

