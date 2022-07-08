""" Environment file wrapper for CARLA """
# Gym imports
import gym


# General imports
from datetime import datetime
import os
import traceback
import numpy as np
from copy import deepcopy


# Environment imports
from src.envs.nocrash.environment.multi_agent.carla_interface import CarlaInterface


class CarlaEnv(gym.Env):
    def __init__(self, config, num_agents, like_gym=False, behavior='normal', logger=None, log_dir=''):
        self.carla_interface = None

        self.config = config

        self.num_agents = num_agents
        self.like_gym = like_gym

        if self.like_gym:
            assert(self.num_agents == 1)

        self.carla_interface = CarlaInterface(config, num_agents, log_dir, behavior=behavior)

        ################################################
        # Logging
        ################################################
        self.log_dir = os.path.join(log_dir, "env")
        if(not os.path.isdir(self.log_dir)):
            os.makedirs(self.log_dir)

        self.logger = logger

        ################################################
        # Creating Action and State spaces
        ################################################
        self.action_space = self.config.action_config.action_space

        self.observation_space = self.config.obs_config.observation_space

    def step(self, actions):
        if self.like_gym:
            actions = actions[None]

        carla_obses, carla_rewards, carla_dones = self.carla_interface.step(actions)

        gym_obses = self.create_observations(carla_obses)

        rewards = np.array(carla_rewards)
        dones = np.array(carla_dones)
        if self.like_gym:
            return gym_obses[0], rewards[0], dones[0], deepcopy(carla_obses)[0]
        else:
            return gym_obses, rewards, dones, deepcopy(carla_obses)

    def create_observations(self, carla_obses):
        obses = []
        for carla_obs in carla_obses:
            if carla_obs is None:
                obs_output = np.zeros(self.observation_space.low.size)
            elif self.config.obs_config.input_type == "pid_obs_speed_ldist_light":
                heading_error = carla_obs['heading_error']
                speed = carla_obs['speed'] / self.config.obs_config.max_speed
                obstacle_dist = carla_obs['obstacle_dist']
                obstacle_speed = carla_obs['obstacle_speed']
                ldist = carla_obs['dist_to_trajectory']
                light = carla_obs['red_light_dist']

                # normalization

                if obstacle_dist == np.inf:
                    obstacle_dist = self.config.obs_config.default_obs_traffic_val
                else:
                    obstacle_dist = obstacle_dist / self.config.obs_config.vehicle_proximity_threshold

                if obstacle_speed == np.inf:
                    obstacle_speed = self.config.obs_config.default_obs_traffic_val
                else:
                    obstacle_speed = obstacle_speed / self.config.obs_config.max_speed

                if light == np.inf:
                    light = self.config.obs_config.default_obs_traffic_val
                else:
                    light = light / self.config.obs_config.traffic_light_proximity_threshold

                obs_output = np.concatenate((
                    np.array([heading_error]),
                    np.array([speed]),
                    np.array([obstacle_dist]),
                    np.array([obstacle_speed]),
                    np.array([ldist]),
                    np.array([light])))
            obses.append(obs_output)
        obses = np.stack(obses, axis=0)
        return obses

    def reset(self, random_spawn=True, index=None):
        ################################################
        # Episode information and initialization
        ################################################
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.num_steps = 0  # Episode level step count
        carla_obs = self.carla_interface.reset(random_spawn=random_spawn, index=index)
        if self.like_gym:
            return self.create_observations(carla_obs)[0]
        else:
            return self.create_observations(carla_obs)

    def close(self):

        try:
            if self.carla_interface is not None:
                self.carla_interface.close()

        except Exception as e:
            print("********** Exception in closing env **********")
            print(e)
            print(traceback.format_exc())

    def __del__(self):
        self.close()

    def get_autopilot_action(self):
        return self.carla_interface.get_autopilot_actions()
