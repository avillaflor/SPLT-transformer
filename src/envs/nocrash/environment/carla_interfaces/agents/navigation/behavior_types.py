# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains the different parameters sets for each behavior. """


import numpy as np


class Cautious(object):
    """Class for Cautious agent."""
    max_speed = 40
    speed_lim_dist = 6
    speed_decrease = 12
    #  safety_time = 3
    safety_time = 4
    braking_distance = 6


class Normal(object):
    """Class for Normal agent."""
    max_speed = 50
    speed_lim_dist = 3
    speed_decrease = 10
    safety_time = 3
    braking_distance = 5


class Aggressive(object):
    """Class for Aggressive agent."""
    max_speed = 70
    speed_lim_dist = 1
    speed_decrease = 8
    safety_time = 3
    braking_distance = 4


class Random(object):
    """Class for sampling params from distribution"""
    def __init__(self):
        self.min_behavior = Cautious()
        #  self.max_behavior = Aggressive()
        self.max_behavior = SuperAggressive()

        self.max_speed = np.random.uniform(self.min_behavior.max_speed, self.max_behavior.max_speed)
        self.speed_lim_dist = np.random.uniform(self.min_behavior.speed_lim_dist, self.max_behavior.speed_lim_dist)
        self.speed_decrease = np.random.uniform(self.min_behavior.speed_decrease, self.max_behavior.speed_decrease)
        self.safety_time = np.random.uniform(self.min_behavior.safety_time, self.max_behavior.safety_time)
        self.braking_distance = np.random.uniform(self.min_behavior.braking_distance, self.max_behavior.braking_distance)


class NewAggressive(object):
    """Class for Aggressive agent."""
    max_speed = 70
    speed_lim_dist = 2
    speed_decrease = 4
    safety_time = 1
    braking_distance = 1


class SuperAggressive(object):
    """Class for Aggressive agent."""
    max_speed = 80
    speed_lim_dist = -5
    speed_decrease = 3
    safety_time = 1
    braking_distance = 1


class NewCautious(object):
    """Class for Aggressive agent."""
    max_speed = 40
    speed_lim_dist = 20
    speed_decrease = 20
    safety_time = 5
    braking_distance = 10
