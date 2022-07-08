#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import math
import numpy as np
import carla


def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return d_angle < 90.0

def is_within_distance_ahead_v2(target_transform, current_transform, max_distance=10.0, min_distance=6.0):
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

    if norm_target > max_distance:
        return False, norm_target, None

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    crossed_vector = np.cross(target_vector, forward_vector)

    return d_angle < 90.0, norm_target, crossed_vector


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2
    location_1, location_2:   carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]

SEMANTIC_COLOR_MAP = {
    0	: ["Unlabeled", ( 0, 0, 0)],
    1	: ["Building",	( 70, 70, 70)],
    2	: ["Fence",	(190, 153, 153)],
    3	: ["Other",	(250, 170, 160)],
    4	: ["Pedestrian",	(220, 20, 60)],
    5	: ["Pole",	(153, 153, 153)],
    6	: ["Road line",	(157, 234, 50)],
    7	: ["Road",	(128, 64, 128)],
    8	: ["Sidewalk",	(244, 35, 232)],
    9	: ["Vegetation",	(107, 142, 35)],
    10	: ["Car",	( 0, 0, 142)],
    11	: ["Wall",	(102, 102, 156)],
    12	: ["Traffic sign",	(220, 220, 0)]
}

SEMANTIC_COLOR_MAP_ARRAY = np.array([
    [0, 0, 0],
    [70, 70, 70],
    [190, 153, 153],
    [250, 170, 160],
    [220, 20, 60],
    [153, 153, 153],
    [157, 234, 50],
    [128, 64, 128],
    [244, 35, 232],
    [107, 142, 35],
    [0, 0, 142],
    [102, 102, 156],
    [220, 220, 0]
]) 

CLASS_REMAP = {
    0	: 0,
    1	: 0,
    2	: 0,
    3	: 0,
    4	: 1,
    5	: 0,
    6	: 2,
    7	: 3,
    8	: 0,
    9	: 0,
    10	: 4,
    11	: 0,
    12	: 0,
    13  : 0,
    14  : 0,
    15  : 0,
    16  : 0,
    17  : 0,
    18  : 0,
    19  : 0,
    20  : 0,
    21  : 0,
    22  : 0
}

CLASS_REMAP_ARRAY = np.array([
    0,
    0,
    0,
    0,
    1,
    0,
    2,
    3,
    0,
    0,
    4,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
])

BINARIZED_REMAP_ARRAY = np.array([
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0
])

REDUCED_SEMANTIC_COLOR_MAP = {
    0	: ["Everything Else", ( 0, 0, 0)],
    1	: ["Pedestrian",	(220, 20, 60)],
    2	: ["Road line",	(157, 234, 50)],
    3	: ["Road",	(128, 64, 128)],
    4	: ["Car",	( 0, 0, 142)]
}

REDUCED_SEMANTIC_COLOR_MAP_ARRAY = np.array([
    [0, 0, 0],
    [220, 20, 60],
    [157, 234, 50],
    [128, 64, 128],
    [0, 0, 142]
])

BINARIZED_SEMANTIC_COLOR_MAP_ARRAY = np.array([
    [0, 0, 0],
    [255, 255, 255]
])

def reduce_classes_old(semantic_image):
    h, w = np.shape(semantic_image)
    # assert(d == 1)
    semantic_reduced_image = np.zeros_like(semantic_image)

    for i in range(h):
        for j in range(w):
            orig_class = semantic_image[i, j]
            new_class = int(CLASS_REMAP[orig_class])
            semantic_reduced_image[i, j] = new_class
    return semantic_reduced_image

def reduce_classes(semantic_image, binarized_image=False):
    h, w = np.shape(semantic_image)
    # # assert(d == 1)
    # semantic_reduced_image = np.zeros_like(semantic_image)
    if binarized_image:
        f = lambda x : BINARIZED_REMAP_ARRAY[x]
    else:
        f = lambda x : CLASS_REMAP_ARRAY[x]
    # print(semantic_image.reshape(-1))
    semantic_reduced_image = f(semantic_image.reshape(-1))
    return semantic_reduced_image.reshape((h,w))


def convert_to_one_hot(labels, num_classes):
    labels = np.squeeze(labels)
    h, w = labels.shape
    flattened_labels = labels.reshape((h*w))
    one_hot = np.zeros((flattened_labels.shape[0], num_classes))
    one_hot[np.arange(flattened_labels.shape[0]), flattened_labels] = 1
    one_hot = one_hot.reshape((h, w, -1))
    return one_hot

def convert_from_one_hot(one_hot):
    return np.argmax(one_hot, axis=2)

def convert_to_rgb_old(semantic_image, reduced_classes=False):
    h, w = np.shape(semantic_image)
    semantic_rgb_image = np.zeros((h, w, 3))

    if reduced_classes:
        semantic_map = REDUCED_SEMANTIC_COLOR_MAP
    else:
        semantic_map = SEMANTIC_COLOR_MAP
    for i in range(h):
        for j in range(w):
            label = semantic_image[i, j]
            rgb_tuple = semantic_map[label][1]
            # print("rgb_tuple", rgb_tuple)
            semantic_rgb_image[i, j, 0] = rgb_tuple[0]
            semantic_rgb_image[i, j, 1] = rgb_tuple[1]
            semantic_rgb_image[i, j, 2] = rgb_tuple[2]

    return semantic_rgb_image

def convert_to_rgb(semantic_image, reduced_classes=False, binarized_image=False):
    h, w = np.shape(semantic_image)
    semantic_rgb_image = np.zeros((h, w, 3))

    if reduced_classes:
        if binarized_image:
            semantic_map = BINARIZED_SEMANTIC_COLOR_MAP_ARRAY
        else:
            semantic_map = REDUCED_SEMANTIC_COLOR_MAP_ARRAY
    else:
        semantic_map = SEMANTIC_COLOR_MAP_ARRAY

    f = lambda x : semantic_map[x]

    semantic_rgb_image = f(semantic_image.reshape(-1))
    return semantic_rgb_image.reshape((h,w,3))
