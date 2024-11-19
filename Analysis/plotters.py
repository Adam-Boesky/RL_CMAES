from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_agent_results(trajectory_fname: str, result_fname: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Function that plots the trajectory as a dashed black line and the agent's path as a colored line."""
    # Load results
    traj = pd.read_csv(trajectory_fname)
    agent = pd.read_csv(result_fname)

    # Create axis if not provided
    if ax is None:
        ax = plt.gca()

    # Plot
    ax.plot(traj['x'], traj['y'], color='k', ls='--', label='Target')
    scatter = ax.scatter(agent['x'], agent['y'], c=agent['time'], s=7.5, label='Agent')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time')

    # Formatting
    ax.legend()

    return ax
