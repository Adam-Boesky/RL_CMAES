from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

def plot_trajectory(trajectory_fname: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Function that plots the trajectory as a dashed black line."""
    # Load results
    traj = pd.read_csv(trajectory_fname)

    # Create axis if not provided
    if ax is None:
        ax = plt.gca()

    # Plot
    ax.plot(traj['x'], traj['y'], color='k', ls='--', label='Target')

    # Formatting
    ax.legend()

    return ax


def plot_result(result_fname: str, ax: Optional[plt.Axes] = None, coloring: Optional[str] = None) -> plt.Axes:
    """Function that plots the trajectory as a dashed black line."""
    # Load results
    agent = pd.read_csv(result_fname)

    # Create axis if not provided
    if ax is None:
        ax = plt.gca()

    # Decide how to color the agent's trajectory
    if coloring == 'speed':
        color = np.sqrt(agent['velocity_x']**2 + agent['velocity_y']**2)
        cbar_label = "Speed"
    elif coloring == 'acceleration':
        color = np.sqrt(agent['acceleration_x']**2 + agent['acceleration_y']**2)
        cbar_label = "Acceleration"
    elif coloring == 'firing':
        color = agent['firing']
        cbar_label = 'Thruster Firing'
    else:
        color = agent['time']
        cbar_label = "Time"

    # Plot
    scatter = ax.scatter(agent['x'], agent['y'], c=color, s=7.5, label='Agent', zorder=-10)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(cbar_label)

    # # Add arrows for each point (ChatGPT)
    # arrow_scale = 0.2
    # for _, row in agent.iterrows():
    #     x, y = row['x'], row['y']
    #     theta = row['theta']
    #     firing = row['firing']
    #     color = 'red' if firing == 1 else 'black'

    #     # Arrow parameters
    #     dx = arrow_scale * np.cos(theta)  # Length in x-direction
    #     dy = arrow_scale * np.sin(theta)  # Length in y-direction

    #     # Draw the arrow
    #     ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, fc=color, ec=color, zorder=5)



    # Formatting
    ax.legend()

    return ax


def plot_result_and_trajectory(trajectory_fname: str, result_fname: str, ax: Optional[plt.Axes] = None, coloring: Optional[str] = None) -> plt.Axes:
    """Function that plots the trajectory as a dashed black line and the agent's path as a colored line."""
    # Load results
    ax = plot_result(result_fname, ax, coloring)
    ax = plot_trajectory(trajectory_fname, ax)

    # Formatting
    ax.legend()

    return ax


def plot_distance_from_target(trajectory_fname: str, result_fname: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Function that plots the trajectory as a dashed black line and the agent's path as a colored line."""
    # Load results
    traj = pd.read_csv(trajectory_fname)
    agent = pd.read_csv(result_fname)

    # Create axis if not provided
    if ax is None:
        ax = plt.gca()

    # Get distance
    x_interp = np.interp(traj['time'], agent['time'], agent['x'])
    y_interp = np.interp(traj['time'], agent['time'], agent['y'])
    distance = np.sqrt((traj['x'] - x_interp)**2 + (traj['y'] - y_interp)**2)

    # Plot
    ax.plot(traj['time'], distance)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance [m]')

    return ax

def generate_animation(trajectory_fname: str, result_fname: str, output_fname: str):
    """Function that generates a GIF of the given trajectory and agent path."""
    traj = pd.read_csv(trajectory_fname)
    agent_path = pd.read_csv(result_fname)

    fig, ax = plt.subplots()

    ax.plot(traj['x'], traj['y'], color='k', ls='--', label='Target')

    ax.set_xlim(min(agent_path['x'].min(), traj['x'].min()) - 1, 
            max(agent_path['x'].max(), traj['x'].max()) + 1)
    ax.set_ylim(min(agent_path['y'].min(), traj['y'].min()) - 1, 
            max(agent_path['y'].max(), traj['y'].max()) + 1)

