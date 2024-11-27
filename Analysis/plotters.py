from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_result(result_fname: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Function that plots the trajectory as a dashed black line."""
    # Load results
    agent = pd.read_csv(result_fname)

    # Create axis if not provided
    if ax is None:
        ax = plt.gca()

    # Plot
    scatter = ax.scatter(agent['x'], agent['y'], c=agent['time'], s=7.5, label='Agent', zorder=-10)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time')

    # Formatting
    ax.legend()

    return ax


def plot_result_and_trajectory(trajectory_fname: str, result_fname: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Function that plots the trajectory as a dashed black line and the agent's path as a colored line."""
    # Load results
    ax = plot_result(result_fname, ax)
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
