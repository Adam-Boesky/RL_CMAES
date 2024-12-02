import os
import numpy as np
import pandas as pd

from typing import Tuple


class Trajectory():
    def __init__(self, trajectory_directory: str, t_final: Tuple[float], dt: float = 0.1):
        """Parent class for trajectories.
        
        Args:
            time_range: The final time value, in seconds.
            dt: The timestep to create the trajectory for.
        """
        self.time_vector = np.arange(0, t_final + dt, step=dt)
        self.trajectory_directory = trajectory_directory

    def func(self) -> Tuple[np.ndarray, np.ndarray]:
        """Function of self.time_vector that dictates how the trajectory looks. This function returns a tuple of numpy
        arrays of the form (x_values, y_values).
        
        NOTE: Must be implemented in subclasses.
        """
        raise NotImplementedError('func() must be implemented in subclass!')

    def write(self, fname: str):
        x, y = self.func()
        pd.DataFrame(
            data={
                'time': self.time_vector,
                'x': x,
                'y': y,
            }
        ).to_csv(os.path.join(self.trajectory_directory, fname), index=False)


class StraightTrajectory(Trajectory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func(self):
        return 0.3 * self.time_vector, -0.5 * self.time_vector


class ArcTrajectory(Trajectory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func(self):
        theta = 0.25 * np.pi * (self.time_vector / np.max(self.time_vector))

        return (100 * np.sin(theta), 100 * np.cos(theta) - 100)

class SineTrajectory(Trajectory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func(self):
        return [2 * self.time_vector, 10 * np.sin(self.time_vector * 2 * np.pi / 100)]

class LoopTrajectory(Trajectory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def func(self):
        return [30*np.cos(self.time_vector*2*np.pi/100) + self.time_vector*30*2*np.pi/400 - 30, 30*np.sin(self.time_vector*2*np.pi /100)]


def f(x):
    return x * (x - 100)**2 * (x - 40)**3 / 9e7

class PolyTrajectory(Trajectory): # ChatGPT
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func(self):
        x = self.time_vector
        y = f(self.time_vector)
        x_rot = (np.sqrt(2) / 2) * (x - y)  # Rotated x
        y_rot = (np.sqrt(2) / 2) * (x + y)  # Rotated y
        return [0.25 * x_rot, 0.25 * y_rot]

class SawtoothTrajectory(Trajectory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func(self):
        # Generate a sawtooth wave: y(t) ranges from -1 to 1 over the period
        return [2 * self.time_vector, 10 * (self.time_vector / 50 % 1) ]


def write_trajectories():
    trajectory_dirpath = os.environ['RL_CMAES_ROOT'] + '/trajectories'

    # Write a straight line trajectory
    straight_traj = StraightTrajectory(trajectory_dirpath, t_final = 100)
    straight_traj.write('straight.csv')

    # Write an arc trajectory
    arc_traj = ArcTrajectory(trajectory_dirpath, t_final = 100)
    arc_traj.write('arc.csv')

    # Write a sine trajctory
    sin_traj = SineTrajectory(trajectory_dirpath, t_final = 100)
    sin_traj.write('sine.csv')

    # Write a loop trajectory
    loop_traj = LoopTrajectory(trajectory_dirpath, t_final = 100)
    loop_traj.write('loop.csv')

    # Write a polynomial trajectory
    poly_traj = PolyTrajectory(trajectory_dirpath, t_final = 100)
    poly_traj.write('poly.csv')

    # Write a sawtooth trajctory
    saw_traj = SawtoothTrajectory(trajectory_dirpath, t_final = 100)
    saw_traj.write('sawtooth.csv')


if __name__=='__main__':
    write_trajectories()
