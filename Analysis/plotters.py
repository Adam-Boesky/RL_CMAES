from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d


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
    # Read the trajectory and agent path data
    traj = pd.read_csv(trajectory_fname)
    agent_path = pd.read_csv(result_fname)

    # Downsample the path
    num_frames = 100
    agent_time = np.linspace(agent_path['time'].min(), agent_path['time'].max(), num_frames)
    agent_x = np.interp(
        x=agent_time,
        xp=agent_path['time'],
        fp=agent_path['x'],
    )
    agent_y = np.interp(
        x=agent_time,
        xp=agent_path['time'],
        fp=agent_path['y'],
    )

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Plot the target trajectory
    ax.plot(traj['x'], traj['y'], color='k', lw=0.5)

    # Set the plot limits
    ax.set_xlim(min(agent_x.min(), traj['x'].min()) - 1, 
                max(agent_x.max(), traj['x'].max()) + 1)
    ax.set_ylim(min(agent_y.min(), traj['y'].min()) - 1, 
                max(agent_y.max(), traj['y'].max()) + 1)

    # Add labels and legend
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Initialize the line object for the agent's path
    agent_line, = ax.plot([], [], color='b', label='Agent')

    # Initialize the point for the moving target
    target_point, = ax.plot([], [], 'ro', label='Target Position')

    # Create interpolation functions for the target trajectory
    traj_x_interp = interp1d(traj['time'], traj['x'], kind='linear', fill_value='extrapolate')
    traj_y_interp = interp1d(traj['time'], traj['y'], kind='linear', fill_value='extrapolate')

    # Initialization function for the animation
    def init():
        agent_line.set_data([], [])
        target_point.set_data([], [])
        return agent_line, target_point

    # Animation function that updates the agent's path
    def animate(i):
        # Update agent's path up to current frame
        x_agent = agent_x[:i+1]
        y_agent = agent_y[:i+1]
        agent_line.set_data(x_agent, y_agent)

        # Get current time from agent_path
        t = agent_time[i]

        # Get target position at current time using interpolation
        x_target = traj_x_interp(t)
        y_target = traj_y_interp(t)

        # Update target point position
        target_point.set_data(x_target, y_target)

        return agent_line, target_point

    # Create the animation
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=num_frames, interval=50, blit=True)

    # Save the animation as a GIF file
    anim.save(output_fname, writer='pillow')

    # Close the plot to prevent display
    plt.close(fig)
