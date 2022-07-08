#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
import numpy as np
import math


from src.envs.nocrash.environment.carla_interfaces.agents.navigation.agent import Agent, AgentState
from src.envs.nocrash.environment.carla_interfaces.agents.tools.misc import get_speed
from src.envs.nocrash.environment.carla_interfaces.agents.navigation.behavior_types import Cautious, Aggressive, Normal, Random, NewAggressive, NewCautious, SuperAggressive


class CustomAgent(Agent):
    def __init__(
            self,
            vehicle,
            traffic_light_proximity_threshold=10.0,
            vehicle_proximity_threshold=45.,
            behavior='normal',
            follow_speed_limit=True):
        super().__init__(
            vehicle,
            traffic_light_proximity_threshold=traffic_light_proximity_threshold,
            vehicle_proximity_threshold=vehicle_proximity_threshold)

        self._traffic_light_proximity_threshold = traffic_light_proximity_threshold  # meters
        self._state = AgentState.NAVIGATING
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._follow_speed_limit = follow_speed_limit
        self._grp = None

        # TODO new variables
        self._min_speed = 5
        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'new_cautious':
            self._behavior = NewCautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

        elif behavior == 'random':
            self._behavior = Random()

        elif behavior == 'new_aggressive':
            self._behavior = NewAggressive()

        elif behavior == 'super_aggressive':
            self._behavior = SuperAggressive()

    def is_vehicle_hazard(self, vehicle_list, next_waypoints=None):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
         vehicles, which center is actually on a different lane but their
         extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = self._vehicle.get_location()
        #  ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        if next_waypoints is None:
            next_waypoints = [self._map.get_waypoint(ego_vehicle_location)]

        found_flag = False
        found_vehicle = None
        closest_dist = self._vehicle_proximity_threshold

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            loc = target_vehicle.get_location()
            dist = loc.distance(ego_vehicle_location)
            if dist < closest_dist:
                target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
                for waypoint in next_waypoints:
                    target_vehicle_junction = target_vehicle_waypoint.get_junction()
                    waypoint_junction = waypoint.get_junction()
                    if ((target_vehicle_waypoint.road_id == waypoint.road_id and
                            target_vehicle_waypoint.lane_id == waypoint.lane_id) or
                            ((target_vehicle_junction is not None) and
                                (waypoint_junction is not None) and
                                (target_vehicle_junction.id == waypoint_junction.id))):

                        if self._is_ahead(loc, waypoint):
                            closest_dist = dist
                            found_flag = True
                            found_vehicle = target_vehicle
                            break

        return (found_flag, found_vehicle)

    def _is_ahead(self, target_location, waypoint=None):
        if waypoint is None:
            transform = self._vehicle.get_transform()
        else:
            transform = waypoint.transform
        current_location = transform.location
        orientation = transform.rotation.yaw

        target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
        norm_target = np.linalg.norm(target_vector)

        forward_vector = np.array(
            [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
        d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

        return d_angle < 90.0

    def _emergency_behavior(self):
        return 0.

    def _default_behavior(self):
        if self._follow_speed_limit:
            speed = self._vehicle.get_speed_limit() - self._behavior.speed_lim_dist
        else:
            speed = self._behavior.max_speed
        return speed

    def _car_following_behavior(self, vehicle):
        distance = vehicle.get_location().distance(self._vehicle.get_location())
        distance = distance - max(
            vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
        vehicle_speed = get_speed(vehicle)
        ego_speed = get_speed(self._vehicle)
        speed_limit = self._vehicle.get_speed_limit()
        delta_v = max(1, (ego_speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if (ttc <= 0.) or (distance <= self._behavior.braking_distance):
            target_speed = 0.
        elif self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                #  max([vehicle_speed - self._behavior.speed_decrease, 0.]),
                max([ego_speed - self._behavior.speed_decrease, 0.]),
                self._behavior.max_speed,
                speed_limit - self._behavior.speed_lim_dist])

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                speed_limit - self._behavior.speed_lim_dist])

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                speed_limit - self._behavior.speed_lim_dist])

        return target_speed

    def get_desired_speed(self, next_waypoints):
        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")
        walker_list = actor_list.filter("*walker*")

        traffic_actor, dist, traffic_light_orientation = self.find_nearest_traffic_light(lights_list)
        if traffic_actor is not None:
            if traffic_actor.state == carla.TrafficLightState.Red:
                if dist < self._traffic_light_proximity_threshold:
                    self._state = AgentState.BLOCKED_RED_LIGHT
                    return self._emergency_behavior()

        # check possible obstacles
        vehicle_state, vehicle = self.is_vehicle_hazard(vehicle_list, next_waypoints)
        if vehicle_state:
            self._state = AgentState.BLOCKED_BY_VEHICLE
            return self._car_following_behavior(vehicle)

        walker_state, walker = self._is_walker_hazard(walker_list)
        if walker_state:
            self._state = AgentState.BLOCKED_BY_VEHICLE
            return self._emergency_behavior()

        return self._default_behavior()
