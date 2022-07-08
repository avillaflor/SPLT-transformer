import numpy as np
import math
import carla
import random


def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.
    :param target_transform: location of the target object
    :param current_transform: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True, 0, norm_target

    if norm_target > max_distance:
        return False, -1, norm_target

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0, d_angle, norm_target


def compute_done_condition(prev_episode_measurements, curr_episode_measurements, config):
    # Episode termination conditions
    success = curr_episode_measurements["distance_to_goal"] < config.scenario_config.dist_for_success

    # Check if static threshold reach, always False if static is disabled
    static = (curr_episode_measurements["static_steps"] > config.scenario_config.max_static_steps) and \
        not config.scenario_config.disable_static

    # Check if collision, always False if collision is disabled
    collision = curr_episode_measurements["is_collision"] and not config.scenario_config.disable_collision
    runover_light = curr_episode_measurements["runover_light"] and not config.scenario_config.disable_traffic_light
    maxStepsTaken = curr_episode_measurements["num_steps"] > config.scenario_config.max_steps
    offlane = (curr_episode_measurements['num_lane_intersections'] > 0) and not config.obs_config.disable_lane_invasion_sensor

    # Conditions to check there is obstacle or red light ahead for last 2 timesteps
    obstacle_ahead = curr_episode_measurements['obstacle_dist'] != -1 and prev_episode_measurements['obstacle_dist'] != -1
    red_light = curr_episode_measurements['red_light_dist'] != -1 and prev_episode_measurements['red_light_dist'] != -1

    if success:
        termination_state = 'success'
        termination_state_code = 0
    elif collision:
        if curr_episode_measurements['obs_collision']:
            termination_state = 'obs_collision'
            termination_state_code = 1
        elif not config.obs_config.disable_lane_invasion_sensor and curr_episode_measurements["out_of_road"]:
            termination_state = 'out_of_road'
            termination_state_code = 2
        elif not config.obs_config.disable_lane_invasion_sensor and curr_episode_measurements['lane_change']:
            termination_state = 'lane_invasion'
            termination_state_code = 3
        else:
            termination_state = 'unexpected_collision'
            termination_state_code = 4
    elif runover_light:
        termination_state = 'runover_light'
        termination_state_code = 5
    elif offlane:
        termination_state = 'offlane'
        termination_state_code = 6
    elif static:
        termination_state = 'static'
        termination_state_code = 7
    elif maxStepsTaken:
        if obstacle_ahead:
            termination_state = 'max_steps_obstacle'
            termination_state_code = 8
        elif red_light:
            termination_state = 'max_steps_light'
            termination_state_code = 9
        else:
            termination_state = 'max_steps'
            termination_state_code = 10
    else:
        termination_state = 'none'
        termination_state_code = 11

    curr_episode_measurements['termination_state'] = termination_state
    curr_episode_measurements['termination_state_code'] = termination_state_code

    done = success or collision or runover_light or offlane or static or maxStepsTaken
    return done


def compute_reward(prev, current, config, verbose=False):
    # Convenience variable
    reward_config = config.reward_config

    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    # Steer Reward
    steer = np.abs(current['control_steer'])
    steer_reward = -reward_config.steer_penalty_coeff * steer
    current["steer_reward"] = steer_reward

    # Speed Reward
    rel_speed = current["speed"] / config.obs_config.max_speed
    prev_rel_speed = prev["speed"] / config.obs_config.max_speed
    speed_reward = reward_config.speed_coeff * rel_speed
    current["speed_reward"] = speed_reward

    # Speed^2 Reward
    speed_squared_reward = reward_config.speed_squared_coeff * (rel_speed ** 2)
    current["speed_squared_reward"] = speed_squared_reward

    # Acceleration Reward
    acceleration_reward = reward_config.acceleration_coeff * (rel_speed - prev_rel_speed)
    current["acceleration_reward"] = acceleration_reward

    # Dist_to_trajectory reward
    if verbose:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    dist_to_trajectory_reward = -reward_config.dist_to_trajectory_coeff * np.abs(current['dist_to_trajectory'])
    current["dist_to_trajectory_reward"] = dist_to_trajectory_reward

    # Light Reward
    light_reward = 0
    current["runover_light"] = False
    # Only compute reward if traffic light enabled
    if (not config.scenario_config.disable_traffic_light):
        if (_check_if_signal_crossed(prev, current)  # Signal Crossed
                and (prev['nearest_traffic_actor_state'] == carla.TrafficLightState.Red)  # Light is red
                and (current["speed"] > config.scenario_config.zero_speed_threshold)  # Vehicle is moving forward
                and (prev['initial_dist_to_red_light'] > config.obs_config.min_dist_from_red_light)):  # We are within threshold distance of red light

            # Add reward if these conditions are true
            current["runover_light"] = True
            light_reward = -1 * (reward_config.const_light_penalty + reward_config.light_penalty_speed_coeff * rel_speed)
    current["light_reward"] = light_reward

    # Collision Reward
    is_collision = False
    lane_change = False
    obs_collision = (current["num_collisions"] - prev["num_collisions"]) > 0
    is_collision = obs_collision

    # count out_of_road also as a collision
    if not config.obs_config.disable_lane_invasion_sensor:
        is_collision = obs_collision or current["out_of_road"]

        # count any lane change also as a collision
        # if config.scenario_config.disable_lane_invasion_collision:
        lane_change = current['num_lane_intersections'] > 0
        is_collision = is_collision or lane_change

    current['obs_collision'] = obs_collision
    current['lane_change'] = lane_change
    current["is_collision"] = is_collision

    if(is_collision):
        # Using prev_speed in collision reward computation
        # due to non-determinism in speed at the time of collision
        collision_reward = -1 * (reward_config.const_collision_penalty + reward_config.collision_penalty_speed_coeff * prev_rel_speed)
        speed_reward = reward_config.speed_coeff * prev_rel_speed

    else:
        collision_reward = 0
    current["collision_reward"] = collision_reward

    # Success Reward
    success_reward = 0
    success = current["distance_to_goal"] < config.scenario_config.dist_for_success
    if success:
        success_reward += reward_config.success_reward
    current["success_reward"] = success_reward

    reward = dist_to_trajectory_reward + \
        speed_reward + \
        speed_squared_reward +\
        steer_reward + \
        collision_reward + \
        light_reward + \
        acceleration_reward + \
        success_reward +  \
        reward_config.constant_positive_reward

    # normalize reward
    reward = reward / reward_config.reward_normalize_factor

    current["step_reward"] = reward

    if verbose:
        print("dist_to_trajectory_reward, speed_reward, acceleration_reward, collision_reward, light_reward, steer_reward, success_reward, reward")
        print(dist_to_trajectory_reward, speed_reward, acceleration_reward, collision_reward, light_reward, steer_reward, success_reward, reward)

    return reward


def _check_if_signal_crossed(prev, current):

    # cross_from_one_light_to_no_light
    cross_to_no_light = current['red_light_dist'] == np.inf and prev['red_light_dist'] != np.inf

    cross_to_next_light = (current['nearest_traffic_actor_id'] != np.inf and prev['nearest_traffic_actor_id'] != np.inf
                           and current['nearest_traffic_actor_id'] != prev['nearest_traffic_actor_id'])

    return cross_to_no_light or cross_to_next_light


# Reordered so that duplicates are at the end
def get_no_crash_path(random_spawn=False, town="Town01", index=0):
    " Returns a list of [start_idx, target_idx]"

    paths_Town01 = [[79, 227], [105, 21], [129, 88], [19, 105], [231, 212],
                    [252, 192], [222, 120], [202, 226], [11, 17],
                    [3, 177], [191, 114], [235, 240], [4, 54], [17, 207],
                    [223, 212], [154, 66], [187, 123], [114, 6],
                    [40, 192], [176, 123], [121, 187], [238, 225], [219, 154],
                    [79, 247], [129, 56]]

    paths_Town02 = [[66, 19], [6, 71], [46, 32], [25, 59],
                    [32, 9], [43, 72], [54, 14], [26, 50], [38, 69],
                    [75, 24], [19, 82], [65, 6], [71, 29], [59, 16],
                    [83, 56], [69, 71], [82, 28], [8, 17],
                    [39, 18], [51, 8], [24, 36], [64, 73],
                    [66, 28], [6, 66], [19, 12]]

    if town == "Town01":
        paths = paths_Town01
    elif town == "Town02":
        paths = paths_Town02
    else:
        raise NotImplementedError

    if random_spawn:
        return random.choice(paths)
    else:
        return paths[index % len(paths)]
