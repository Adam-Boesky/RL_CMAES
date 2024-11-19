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
        return self.time_vector, self.time_vector


class ArcTrajectory(Trajectory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func(self):
        theta = 0.25 * np.pi * (self.time_vector / np.max(self.time_vector))

        return (100 * np.cos(theta), 100 * np.sin(theta))


def write_trajectories():
    trajectory_dirpath = '/Users/adamboesky/Research/RL_CMAES/trajectories'

    # Write a straight line trajectory
    straight_traj = StraightTrajectory(trajectory_dirpath, t_final = 100)
    straight_traj.write('straight.csv')

    # Write an arc trajectory
    arc_traj = ArcTrajectory(trajectory_dirpath, t_final = 100)
    arc_traj.write('arc.csv')


if __name__=='__main__':
    write_trajectories()
