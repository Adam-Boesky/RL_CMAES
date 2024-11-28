import cma
import pandas as pd
import numpy as np
import subprocess
import os


CURRENT_TRAJECTORY = "straight"

# Objective function
def objective_function(x):
    '''
    ::params::
        x = [gamma_i, s_i, d_i]
    '''

    gamma = str(x[0])
    s = str(x[0])
    d = str(x[0])

    command_to_run_sim = "./sim/main"
    # 1. Run a trajectory with the sampled MVN in the CMA-ES
    subprocess.run([command_to_run_sim, gamma, s, d], capture_output=True, text=True)

    # 2. Read in the actual and target trajectory values generated during the previous run
    actual_trajectory = pd.read_csv("./sim_results/test.csv")
    target_trajectory = pd.read_csv(f"./trajectories/{CURRENT_TRAJECTORY}.csv")

    # 3. Calculate and return the loss values
    loc_actual = actual_trajectory[['x']]
    loc_target = target_trajectory[['x']]

    n = len(loc_target)
    loss = sum([abs(loc_actual.iloc[i] - loc_target.iloc[i]) for i in range(n)])

    return loss


if __name__ == "__main__":
    initial_mean = [0.5, 0.5, 0.5]
    sigma = 0.5
    
    es = cma.CMAEvolutionStrategy(initial_mean, sigma)
    
    es.optimize(objective_function)
    result = es.result
    print("Optimized solution:", result[0])
    print("Best value found:", result[1])
