import time
import carla
import numpy as np


from src.envs.nocrash.environment.carla_interfaces.server import CarlaServer
from src.envs.nocrash.environment.multi_agent.ego_vehicle_manager import EgoVehicleManager
from src.envs.nocrash.environment.multi_agent.npcs_vehicle_manager import NPCsVehicleManager
from src.envs.nocrash.environment.multi_agent import utils


class CarlaInterface():

    def __init__(self, config, num_agents, log_dir, behavior):
        self.config = config

        # Instantiate and start server
        self.server = CarlaServer(config)

        self.client = None

        self.num_agents = num_agents
        self.ego_agents = [None for _ in range(num_agents)]
        self.actives = np.zeros(self.num_agents, dtype=bool)
        self.npcs_vehicle_manager = None

        self.log_dir = log_dir

        self._behavior = behavior

        self.setup()

    def setup(self):
        # Start the carla server and get a client
        self.server.start()
        self.client = self._spawn_client()

        # Get the world
        self.world = self.client.load_world(self.config.scenario_config.city_name)

        # Update the settings from the config
        settings = self.world.get_settings()
        if(self.config.sync_mode):
            settings.synchronous_mode = True
        if self.config.server_fps is not None and self.config.server_fps != 0:
            settings.fixed_delta_seconds = 1.0 / float(self.config.server_fps)
            settings.substepping = True
            settings.max_substep_delta_time = 0.0125
            #  settings.max_substep_delta_time = 0.01
            settings.max_substeps = int(settings.fixed_delta_seconds / settings.max_substep_delta_time)

        # Enable rendering
        # TODO render according to config settings
        settings.no_rendering_mode = True

        self.world.apply_settings(settings)

        # Sleep to allow for settings to update
        time.sleep(5)

        # Retrieve map
        self.map = self.world.get_map()

        # Get blueprints
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Instantiate a vehicle manager to handle other actors
        for i in range(self.num_agents):
            self.ego_agents[i] = EgoVehicleManager(self.config, self.client, self._behavior)
        self.npcs_vehicle_manager = NPCsVehicleManager(self.config, self.client)

        # Get traffic lights
        self.traffic_actors = self.world.get_actors().filter("*traffic_light*")

        print("server_version", self.client.get_server_version())

    def _spawn_client(self, hostname='localhost', port_number=None):
        port_number = self.server.server_port
        client = carla.Client(hostname, port_number)
        client.set_timeout(self.config.client_timeout_seconds)

        return client

    def _set_scenario(self, town="Town01", index_offset=None):
        for index in range(self.num_agents):
            if (self.config.scenario_config.scenarios == "no_crash_empty" or
                    self.config.scenario_config.scenarios == "no_crash_regular" or
                    self.config.scenario_config.scenarios == "no_crash_dense"):
                if index_offset is not None:
                    i = (index + index_offset)
                else:
                    i = index
                source_idx, destination_idx = utils.get_no_crash_path(random_spawn=False, town=town, index=i)
                self.ego_agents[index].spawn(self.spawn_points[source_idx], self.spawn_points[destination_idx])
            else:
                raise ValueError("Invalid Scenario Type {}. Check scenario config!".format(self.config.scenario_config.scenarios))

    # Assuming this is a full reset
    def reset(self, random_spawn=True, index=None):
        # Delete old actors
        self.destroy_actors()

        # Set the new scenarios
        if self.config.scenario_config.use_scenarios and \
                (self.config.scenario_config.city_name == "Town01" or self.config.scenario_config.city_name == "Town02"):
            self._set_scenario(town=self.config.scenario_config.city_name, index_offset=index)
        else:
            raise Exception

        # Spawn new actors
        self.npcs_vehicle_manager.spawn(random_spawn=random_spawn)

        # Tick for 15 frames to handle car initialization in air
        for _ in range(15):
            world_frame = self.world.tick()

        default_control = {
            "target_speed": 0.0,
            "control_steer": 0.0,
            "control_throttle": 0.0,
            "control_brake": 0.0,
            "control_reverse": False,
            "control_hand_brake": False
        }

        obses = []
        for ego_agent in self.ego_agents:
            ego_obs = ego_agent.obs(world_frame)
            obs = {**ego_obs, **default_control}
            obses.append(obs)
        self.prev_obses = obses
        self.actives = np.ones(self.num_agents, dtype=bool)
        return obses

    def step(self, actions):
        controls = []
        for i in range(self.num_agents):
            if self.actives[i]:
                control = self.ego_agents[i].step(actions[i])
                controls.append(control)
            else:
                controls.append(None)

        world_frame = self.world.tick()

        for i in range(self.num_agents):
            if self.actives[i]:
                self.ego_agents[i].check_for_vehicle_elimination()
        self.npcs_vehicle_manager.step()

        obses = []
        dones = []
        rewards = []
        for i in range(self.num_agents):
            if self.actives[i]:
                ego_obs = self.ego_agents[i].obs(world_frame)
                obs = {**ego_obs, **controls[i]}
                reward = utils.compute_reward(
                    prev=self.prev_obses[i],
                    current=obs,
                    config=self.config,
                    verbose=self.config.verbose,
                )
                done = utils.compute_done_condition(
                    prev_episode_measurements=self.prev_obses[i],
                    curr_episode_measurements=obs,
                    config=self.config)
                self.actives[i] = not done
                if done:
                    self.ego_agents[i].destroy_actors()
                    print(i, obs['termination_state'])
            else:
                obs = None
                done = True
                reward = 0.

            obses.append(obs)
            dones.append(done)
            rewards.append(reward)

        return obses, rewards, dones

    def get_autopilot_actions(self):
        actions = []
        for i in range(self.num_agents):
            if self.actives[i]:
                actions.append(self.ego_agents[i].get_autopilot_action())
            else:
                actions.append(np.array([0., -1.]))
        actions = np.stack(actions, axis=0)
        return actions

    @property
    def actor_list(self):
        actor_list = []
        for ego_agent in self.ego_agents:
            actor_list += ego_agent.actor_list
        actor_list += self.npcs_vehicle_manager.actor_list
        return actor_list

    def destroy_actors(self):
        for ego_agent in self.ego_agents:
            ego_agent.destroy_actors()
        self.npcs_vehicle_manager.destroy_actors()

    def close(self):
        self.destroy_actors()

        if self.server is not None:
            self.server.close()
