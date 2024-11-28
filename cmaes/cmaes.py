import cma
import pandas as pd
import numpy as np
import subprocess
import os


def norm(v):
    ''' Computes the Euclidian norm of a vector. '''
    return np.sqrt(sum([x*x for x in v]))

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
    actual_trajectory = pd.read_csv("./sim_results/test.csv").iloc[:-1, :]
    target_trajectory = pd.read_csv(f"./trajectories/{CURRENT_TRAJECTORY}.csv")

    # 3. Calculate and return the loss values
    actual_positions = [np.array([row['x'], row['y']]) for _, row in actual_trajectory.iterrows()]
    target_positions = [np.array([row['x'], row['y']]) for _, row in target_trajectory.iterrows()]
    
    n = len(target_positions)
    loss = sum([norm(actual_positions[i] - target_positions[i]) for i in range(n)])
    return loss


if __name__ == "__main__":
    initial_mean = [1.0, 1.0, 1.0]
    sigma = 10
    
    es = cma.CMAEvolutionStrategy(initial_mean, sigma)
    
    es.optimize(objective_function)
    result = es.result
    print("Optimized solution:", result[0])
    print("Best value found:", result[1])
