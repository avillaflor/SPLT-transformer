import numpy as np
from gym.spaces import Box


from src.envs.nocrash.environment.config.base_config import BaseConfig


class BaseActionConfig(BaseConfig):
    def __init__(self):
        # What action space to use
        self.action_type = None

        # Gym Action Space
        self.action_space = None

        # Whether or not to use the brake when driving
        # If False, vehicle will not use brakes to decelarate
        self.enable_brake = None
        self.discrete_actions = None
        # Number of frames to skip between policy actions
        self.frame_skip = None
        # If true, the PID controller will calculate new commands at each skipped time step
        # If false, the same command (calculated from PID at the first step) will be used
        self.use_pid_in_frame_skip = None

        # "Speed Limit" of the vehicle
        self.target_speed = None


class MergedSpeedTanhConfig(BaseActionConfig):
    def __init__(self):
        self.action_type = "merged_speed_tanh"
        self.action_space = Box(low=np.array([-0.5, -1.0]), high=np.array([0.5, 1.0]))
        self.enable_brake = True
        self.discrete_actions = False
        self.frame_skip = 1
        self.use_pid_in_frame_skip = True
        self.target_speed = 20
