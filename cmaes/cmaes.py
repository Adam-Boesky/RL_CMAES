import cma
import pandas as pd
import numpy as np
import subprocess
import os
import random


def norm(v):
    '''Computes the Euclidian norm of a vector.'''
    return np.sqrt(sum([x*x for x in v]))

# Objective function
def objective_function(x, traj_name):
    '''
    ::params::
        x = [gamma_i, s_i, d_i]
    '''

    gamma = str(x[0])
    alpha = str(x[1])
    r = str(x[2])
    s = str(x[3])
    d = str(x[4])

    command_to_run_sim = "./sim/main"

    # 1. Run a trajectory with the sampled MVN in the CMA-ES
    subprocess.run([command_to_run_sim, gamma, alpha, r, s, d, traj_name, "0"], capture_output=True, text=True)

    # 2. Read in the actual and target trajectory values generated during the previous run
    actual_trajectory = pd.read_csv("./sim_results/train.csv").iloc[:-1, :]
    target_trajectory = pd.read_csv(f"./trajectories/{traj_name}.csv")

    # 3. Calculate and return the loss values
    actual_positions = [np.array([row['x'], row['y']]) for _, row in actual_trajectory.iterrows()]
    target_positions = [np.array([row['x'], row['y']]) for _, row in target_trajectory.iterrows()]
    
    n = len(target_positions)
    loss = sum([norm(actual_positions[i] - target_positions[i]) for i in range(n)])
    return loss


if __name__ == "__main__":

    traj_names = ['sine']#, 'straight', 'arc', 'loop', 'poly', 'sawtooth']
    # traj_names = ['poly']

    for i, t_name in enumerate(traj_names):
    
        initial_mean = [15,15,5,5,5]
        sigma = 5

        es = cma.CMAEvolutionStrategy(initial_mean, sigma, {'bounds':[0, np.inf], 'seed':207})
        es.optimize(objective_function, args=[t_name])
        
        result = es.result
        print("Best solution:", result[0])
        print("Best value found:", result[1])

        param_names = ["gamma", "alpha", "r", "s", "d", "min_loss"]
        # best_soln = pd.DataFrame([np.append(result[0], [result[1]])], columns=param_names)
        fav_soln = pd.DataFrame([np.append(result[5], [np.nan])], columns=param_names).to_csv(f"./bounded_params/{t_name}_vals.csv", index=False)
        # df = pd.concat([best_soln, fav_soln], ignore_index=True)

        
        
