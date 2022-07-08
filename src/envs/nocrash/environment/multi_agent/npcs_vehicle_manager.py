import numpy as np
import random
import time


class NPCsVehicleManager():
    def __init__(self, config, client):
        '''
        Common/High level attributes are:
        1) Spawn points (Used for spwaning actors and also by planner)
        2) Blueprints
        '''
        self.config = config
        self.world = client.get_world()
        self.map = self.world.get_map()

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

        # tm is valid for carla0.9.10. If using carla0.9.6, this has to be commented out
        # This is for autopilot purpose on npcs
        # push it to spawn_npc() function?
        #  tm_port = np.random.randint(10000, 60000)
        # pseudo random so we can set seed
        tm_port = np.random.randint(10000, 50000) + (int(time.time() * 1e9) % 10000)
        self.tm = client.get_trafficmanager(tm_port)
        self.tm.set_synchronous_mode(True)
        # TODO allow adjusting seed
        if self.config.testing:
            self.tm.set_random_device_seed(0)

        self._dt = self.config.action_config.frame_skip / self.config.server_fps

        self.actor_list = []

    def spawn(self, random_spawn):
        if self.config.scenario_config.sample_npc:
            number_of_vehicles = np.random.randint(
                low=self.config.scenario_config.num_npc_lower_threshold,
                high=self.config.scenario_config.num_npc_upper_threshold)
        else:
            number_of_vehicles = self.config.scenario_config.num_npc

        self.spawn_npc(number_of_vehicles, random_spawn)
        self.world.reset_all_traffic_lights()

    def step(self):
        self.check_for_vehicle_elimination()

    def check_for_vehicle_elimination(self):
        # https://github.com/carla-simulator/carla/issues/3860
        new_actor_list = []
        for i, actor in enumerate(self.actor_list):
            if actor.is_alive:
                new_actor_list.append(actor)
        self.actor_list = [actor for actor in self.actor_list if actor.is_alive]

    def spawn_npc(self, number_of_vehicles, random_spawn):
        npc_spawn_points = self.pick_npc_spawn_points(random_spawn)
        count = number_of_vehicles
        for spawn_point in npc_spawn_points:
            if self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point):
                count -= 1
            if count <= 0:
                break

    def pick_npc_spawn_points(self, random_spawn):
        if random_spawn:
            spawn_points = np.random.permutation(self.spawn_points)
        else:
            spawn_points = self.spawn_points

        if self.config.verbose:
            print('found %d spawn points.' % len(spawn_points))

        return spawn_points

    def try_spawn_random_vehicle_at(self, blueprints, transform):
        # To spawn same type of vehicle
        blueprint = blueprints[0]
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        tm_port = self.tm.get_port()
        if vehicle is not None:
            self.actor_list.append(vehicle)
            vehicle.set_autopilot(True, tm_port)
            self.tm.ignore_lights_percentage(vehicle, 0)
            self.tm.ignore_signs_percentage(vehicle, 0)
            self.tm.ignore_vehicles_percentage(vehicle, 0)
            self.tm.ignore_walkers_percentage(vehicle, 0)

            min_speed_diff = -20.
            max_speed_diff = 30.
            speed_diff = np.random.rand() * (max_speed_diff - min_speed_diff) + min_speed_diff
            self.tm.vehicle_percentage_speed_difference(vehicle, speed_diff)

            min_lead_diff = 1.
            max_lead_diff = 3.
            lead_diff = np.random.rand() * (max_lead_diff - min_lead_diff) + min_lead_diff
            self.tm.distance_to_leading_vehicle(vehicle, lead_diff)

            if self.config.verbose:
                print('spawned %r at %s' % (vehicle.type_id, transform.location.x))
            return True
        return False

    def destroy_actors(self):
        for _ in range(len(self.actor_list)):
            try:
                actor = self.actor_list.pop()
                actor.destroy()
            except Exception:
                pass
