import numpy as np
import os
import sys
import pyproj

CARLA_9_11_PATH = os.environ.get("CARLA_9_11_PATH")
CARLA_9_11_PYTHONPATH = os.environ.get("CARLA_9_11_PYTHONPATH")
if CARLA_9_11_PATH == None:
    raise ValueError("Set $CARLA_9_11_PATH to directory that contains CarlaUE4.sh")
if CARLA_9_11_PYTHONPATH == None:
    raise ValueError("Set $CARLA_9_11_PYTHONPATH to directory that contains egg file")

try:
    sys.path.append(CARLA_9_11_PYTHONPATH)
except IndexError:
    print(".egg file not found! Kindly check for your Carla installation.")
    pass


import carla
from carla.libcarla import Location


def _create_bb_points(vehicle):
        """
        Returns lower plane of 3D bounding box for a vehicle.
        """
        cords = np.zeros((4, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        return cords

def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = get_matrix(bb_transform)
        vehicle_world_matrix = get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

def get_bounding_box(vehicle):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = _create_bb_points(vehicle)
        cords_x_y_z = _vehicle_to_world(bb_cords, vehicle)[:3, :]
        return np.array(cords_x_y_z)

def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

def get_wp_from_bb(bbox_cords, world_map):
    """
    Get list of wp from given bbox_cords array
    """

    dimension, num_pts = bbox_cords.shape
    assert(dimension == 3)

    bb_wps = []
    for i in range(num_pts):
        vertex = bbox_cords[:,i]
        vertex_location = Location(x=vertex[0], y=vertex[1], z=vertex[2])
        vertex_wp = world_map.get_waypoint(vertex_location)
        bb_wps.append(vertex_wp)

    return bb_wps

def get_road_lane_id_set_from_wp(wp_list):

    road_lane_id_set = set()
    for wp in wp_list:
        road_lane_id_set.add((wp.road_id, wp.lane_id))

    return road_lane_id_set

def get_vehicle_bb_wp(world_map, vehicle):
    '''
    Get waypoints corresponding to vehicle centre and vertices of bb of vehicle
    '''
    vehicle_wp = world_map.get_waypoint(vehicle.get_location())
    vehicle_bb = get_bounding_box(vehicle)

    vehicle_bb_wp = get_wp_from_bb(vehicle_bb, world_map)
    vehicle_bb_wp.append(vehicle_wp)

    return vehicle_bb_wp


def check_if_vehicle_in_same_lane(vehicle_actor, target_vehicle, next_waypoints, world_map):
    '''
    Checks if target_vehicle is in same lane/road as vehicle_actor or next_waypoints
    '''

    target_vehicle_bb_wp = get_vehicle_bb_wp(world_map, target_vehicle)
    target_vehicle_road_id_set = get_road_lane_id_set_from_wp(target_vehicle_bb_wp)

    vehicle_bb_wp = get_vehicle_bb_wp(world_map, vehicle_actor)
    vehicle_wp_list = vehicle_bb_wp + next_waypoints
    vehicle_road_id_set = get_road_lane_id_set_from_wp(vehicle_wp_list)

    intersection_set = target_vehicle_road_id_set & vehicle_road_id_set

    return (len(intersection_set) > 0)

def _latlon_to_ecef(lat,lon,alt):
    # Projections
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    # Transform from lat/lon to ecef
    x,y,z= pyproj.transform(p1=lla,
        p2 = ecef,
        x = lon,
        y = lat,
        z = alt,
        radians=False)

    return x, y, z

def _ecef_to_latlon(x,y,z):
    # Projections
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    # Transform from lat/lon to ecef
    lon, lat, alt= pyproj.transform(p1=ecef,
        p2 = lla,
        x = x,
        y = y,
        z = z,
        radians=False)

    return lat, lon, alt

def get_world_coords_from_latlong(latitude, longitude, altitude, world_map):
    origin_latlong = world_map.transform_to_geolocation(carla.Location())

    # Origin in ECEF coordinates
    O_ecef = _latlon_to_ecef(origin_latlong.latitude, origin_latlong.longitude, origin_latlong.altitude)

    # Convert GNSS data to ECEF coordinates
    P_ecef = _latlon_to_ecef(latitude, longitude, altitude)

    # Calculate difference between current location and origin
    delta = np.expand_dims(np.array(P_ecef) - np.array(O_ecef), axis = 1)

    # Create the rotation matrix to convert from ECEF to ENU Coords
    ecef_to_enu_rot = np.array(
        [[-np.sin(longitude), np.cos(longitude), 0],
         [-np.sin(latitude) * np.cos(longitude), -np.sin(latitude) * np.sin(longitude), np.cos(latitude)],
         [np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)]]
    )
    enu = ecef_to_enu_rot@delta

    # Create rotation matrix to convert from right hand ENU frame to left-hand CARLA frame
    enu_to_carla_rot = np.array(
        [[1, 0, 0],
         [0,-1, 0],
         [0, 0, 1]]
    )

    return enu_to_carla_rot@enu

def get_speed_from_velocity(velocity):
    speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
    return speed

def convert_route_from_GPS_world(route, world_map):

    # Example route input
    # route =[({'z': 0.0, 'lat': 48.99822669411668, 'lon': 8.002271601998707}, RoadOption.LEFT),
    #     ({'z': 0.0, 'lat': 48.99822669411668, 'lon': 8.002709765148996}, RoadOption.RIGHT),
    #     ({'z': 0.0, 'lat': 48.99822679980298, 'lon': 8.002735250105061}, RoadOption.STRAIGHT)
    #     ]

    mapped_route = []
    for idx, pt in enumerate(route):
        print(pt)
        altitude = pt[0]['z']
        latitude = pt[0]['lat']
        longitude = pt[0]['lon']
        world_coord = get_world_coords_from_latlong(latitude, longitude, altitude, world_map)
        x, y, z = world_coord[0][0], world_coord[1][0], world_coord[2][0]
        mapped_route.append(carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation()))
    return mapped_route

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
