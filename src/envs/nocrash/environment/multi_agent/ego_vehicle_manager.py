import numpy as np
import carla
import math


from src.envs.nocrash.environment import env_util as util
import src.envs.nocrash.environment.carla_interfaces.sensors as sensors
import src.envs.nocrash.environment.carla_interfaces.controller as controller
from src.envs.nocrash.environment.carla_interfaces import planner
from src.envs.nocrash.environment.carla_interfaces.agents.navigation.custom_agent import CustomAgent


class EgoVehicleManager():
    def __init__(self, config, client, behavior):
        '''
        Manages ego vehicle, other actors and sensors
        Assumes that sensormanager is always attached to ego vehicle

        Common/High level attributes are:
        1) Spawn points (Used for spwaning actors and also by planner)
        2) Blueprints
        '''
        self.config = config
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.ego_agent = None
        self.ego_vehicle = None
        self.ego_longitudinal_controller = None
        self.ego_lateral_controller = None
        self.global_planner = None
        self._behavior = behavior
        self._num_wp_lookahead = 20

        ################################################
        # Spawn points
        ################################################
        self.spawn_points = self.world.get_map().get_spawn_points()

        ################################################
        # Blueprints
        ################################################
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        if self.config.scenario_config.disable_two_wheeler:
            self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        self.actor_list = []

    def spawn(self, source_transform, destination_transform):
        # Parameters for ego vehicle
        self.ego_agent, self.ego_vehicle, self.ego_longitudinal_controller, self.ego_lateral_controller = self._spawn_ego_vehicle(
            source_transform,
            destination_transform)

        self.global_planner = planner.GlobalPlanner()
        self.destination_transform = destination_transform
        dense_waypoints = self.global_planner.trace_route(self.map, source_transform, self.destination_transform)
        self.global_planner.set_global_plan(dense_waypoints)

        self._reset_counters()

        self.sensor_manager = self.spawn_sensors()

    def _reset_counters(self):
        self.static_steps = 0
        self.num_steps = 0

    def _spawn_ego_vehicle(self, source_transform, destination_transform):
        '''
        Spawns and return ego vehicle/Agent
        '''
        # Spawn the actor
        # Create an Agent object with that actor
        # Return the agent instance
        vehicle_bp = self.blueprint_library.find(self.config.scenario_config.vehicle_type)

        # Spawning vehicle actor with retry logic as it fails to spawn sometimes
        NUM_RETRIES = 5

        for _ in range(NUM_RETRIES):
            vehicle_actor = self.world.try_spawn_actor(vehicle_bp, source_transform)
            if vehicle_actor is not None:
                break
            else:
                print("Unable to spawn vehicle actor at {0}, {1}.".format(source_transform.location.x, source_transform.location.y))
                print("Number of existing actors, {0}".format(len(self.actor_list)))
                self.destroy_actors()              # Do we need this as ego vehicle is the first one to be spawned?
                # time.sleep(120)

        if vehicle_actor is not None:
            self.actor_list.append(vehicle_actor)
        else:
            raise Exception("Failed in spawning vehicle actor.")

        # Agent uses proximity_threshold to detect traffic lights.
        # Hence we use traffic_light_proximity_threshold while creating an Agent.
        vehicle_agent = CustomAgent(
            vehicle_actor,
            traffic_light_proximity_threshold=self.config.obs_config.traffic_light_proximity_threshold,
            vehicle_proximity_threshold=self.config.obs_config.vehicle_proximity_threshold,
            behavior=self._behavior)
        dt = self.config.action_config.frame_skip / self.config.server_fps
        args_longitudinal_dict = {
            'K_P': 1.4,
            'K_D': 0.0,
            'K_I': 0.0,
            'dt': dt}
        args_lateral_dict = {
            'K_P': 1.25,
            'K_D': 0.0,
            'K_I': 0.0,
            'dt': dt}
        longitudinal_controller = controller.PIDLongitudinalController(
                K_P=args_longitudinal_dict['K_P'],
                K_D=args_longitudinal_dict['K_D'],
                K_I=args_longitudinal_dict['K_I'],
                dt=args_longitudinal_dict['dt'])
        lateral_controller = controller.PIDLateralController(
                vehicle_actor,
                K_P=args_lateral_dict['K_P'],
                K_D=args_lateral_dict['K_D'],
                K_I=args_lateral_dict['K_I'],
                dt=args_lateral_dict['dt'])

        return vehicle_agent, vehicle_actor, longitudinal_controller, lateral_controller

    @property
    def ego_vehicle_transform(self):
        return self.ego_vehicle.get_transform()

    @property
    def ego_vehicle_velocity(self):
        return self.ego_vehicle.get_velocity()

    def get_control(self, action):
        """ Get Control object for Carla from action
        Input:
            - action: tuple containing (steer, throttle, brake) in [-1, 1]
        Output:
            - control: Control object for Carla
        """

        episode_measurements = {}

        if self.config.action_config.action_type != "control":
            action = action.flatten()

        if self.config.action_config.action_type == "merged_speed_tanh":
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float((action[1] + 1) * 20.0)
            if target_speed < 5.0:
                throttle = 0.0
                brake = -1.0
            else:
                current_speed = util.get_speed_from_velocity(self.ego_vehicle_velocity) * 3.6
                gas = self.ego_longitudinal_controller.pid_control(target_speed, current_speed, enable_brake=self.config.action_config.enable_brake)
                if gas < 0:
                    throttle = 0.0
                    brake = abs(gas)
                else:
                    throttle = gas
                    brake = 0.0
        else:
            raise Exception("Invalid Action Type")

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        episode_measurements["target_speed"] = target_speed

        episode_measurements['control_steer'] = control.steer
        episode_measurements['control_throttle'] = control.throttle
        episode_measurements['control_brake'] = control.brake
        episode_measurements['control_reverse'] = control.reverse
        episode_measurements['control_hand_brake'] = control.hand_brake

        return control, episode_measurements

    def step(self, action):
        control, ep_measurements = self.get_control(action)
        self.ego_vehicle.apply_control(control)
        return ep_measurements

    def obs(self, world_frame):
        all_sensor_readings = self.sensor_manager.get_sensor_readings(world_frame)

        (next_orientation,
            dist_to_trajectory,
            distance_to_goal_trajec,
            self.next_waypoints,
            next_wp_angles,
            next_wp_vectors,
            all_waypoints) = self.global_planner.get_next_orientation_new(self.ego_vehicle_transform)

        ep_measurements = {
            'next_orientation': next_orientation,
            'heading_error': self._get_heading_error(),
            'distance_to_goal_trajec': distance_to_goal_trajec,
            'dist_to_trajectory': dist_to_trajectory,
            'distance_to_goal': self.ego_vehicle_transform.location.distance(self.destination_transform.location),
            'speed': util.get_speed_from_velocity(self.ego_vehicle_velocity),
            'is_junction': self.next_waypoints[0].is_junction,
        }
        vehicle_detector_measurements = self._vehicle_detector()
        traffic_light_measurements = self._traffic_light_detector()

        sensor_readings = {}
        for key in all_sensor_readings:
            if 'sensor.camera' in key:
                sensor_readings[key] = all_sensor_readings[key]['image']
            else:
                sensor_readings.update(all_sensor_readings[key])

        episode_measurements = {**ep_measurements, **sensor_readings, **vehicle_detector_measurements, **traffic_light_measurements}

        self._update_counters(episode_measurements)

        return episode_measurements

    def _get_heading_error(self):
        waypoint = self.next_waypoints[0]
        vehicle_transform = self.ego_vehicle.get_transform()
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(
            x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
            y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([
            waypoint.transform.location.x - v_begin.x,
            waypoint.transform.location.y - v_begin.y,
            0.0
        ])
        _dot = math.acos(np.clip(
            np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)),
            -1.0,
            1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0
        return _dot

    def _update_counters(self, episode_measurements):
        if episode_measurements["speed"] <= self.config.scenario_config.zero_speed_threshold:
            self.static_steps += 1
        else:
            self.static_steps = 0
        self.num_steps += 1

        episode_measurements['static_steps'] = self.static_steps
        episode_measurements['num_steps'] = self.num_steps

    def _vehicle_detector(self):
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        episode_measurements = {}

        vehicle_state, vehicle = self.ego_agent.is_vehicle_hazard(vehicle_list, self.next_waypoints[:self._num_wp_lookahead])
        if vehicle_state:
            distance = vehicle.get_location().distance(self.ego_vehicle.get_location())
            episode_measurements['obstacle_dist'] = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self.ego_vehicle.bounding_box.extent.y, self.ego_vehicle.bounding_box.extent.x)
            episode_measurements['obstacle_speed'] = util.get_speed_from_velocity(vehicle.get_velocity())
        else:
            episode_measurements['obstacle_dist'] = np.inf
            episode_measurements['obstacle_speed'] = np.inf
        return episode_measurements

    def _traffic_light_detector(self):
        traffic_actors = self.world.get_actors().filter("*traffic_light*")
        episode_measurements = {}
        traffic_actor, dist, traffic_light_orientation = self.ego_agent.find_nearest_traffic_light(traffic_actors)
        if traffic_actor is not None:
            episode_measurements['nearest_traffic_actor_id'] = traffic_actor.id
            episode_measurements['nearest_traffic_actor_state'] = traffic_actor.state
            if traffic_actor.state == carla.TrafficLightState.Red:
                episode_measurements['red_light_dist'] = dist
            else:
                episode_measurements['red_light_dist'] = np.inf
        else:
            episode_measurements['nearest_traffic_actor_id'] = np.inf
            episode_measurements['nearest_traffic_actor_state'] = np.inf
            episode_measurements['red_light_dist'] = np.inf
        return episode_measurements

    def check_for_vehicle_elimination(self):
        # https://github.com/carla-simulator/carla/issues/3860
        if not self.ego_vehicle.is_alive:
            self.ego_vehicle = None

    def get_autopilot_action(self):
        desired_speed = self.ego_agent.get_desired_speed(self.next_waypoints[:self._num_wp_lookahead])
        if len(self.next_waypoints) > 0:
            waypoint = self.next_waypoints[0]
            steer = self.ego_lateral_controller.pid_control(waypoint)
        else:
            steer = 0
        steer = np.clip(steer, -1, 1)
        target_speed = (desired_speed / 20.) - 1.
        return np.array([steer, target_speed])

    def spawn_sensors(self):
        if self.ego_vehicle is None:
            print("Not spawning sensors as the parent actor is not initialized properly")
            return None
        sensor_manager = sensors.SensorManager(self.config, self.ego_vehicle)
        sensor_manager.spawn()
        for k, v in sensor_manager.sensors.items():
            self.actor_list.append(v.sensor)
        return sensor_manager

    def destroy_actors(self):
        for _ in range(len(self.actor_list)):
            try:
                actor = self.actor_list.pop()
                actor.destroy()
            except Exception:
                pass
