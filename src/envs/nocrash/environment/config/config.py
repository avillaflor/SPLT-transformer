import os
import sys


# Import other necessary configs
from src.envs.nocrash.environment.config.base_config import BaseConfig
from src.envs.nocrash.environment.config import observation_configs, action_configs, scenario_configs, reward_configs
from src.envs.nocrash.environment.config.observation_configs import BaseObservationConfig
from src.envs.nocrash.environment.config.action_configs import BaseActionConfig
from src.envs.nocrash.environment.config.scenario_configs import BaseScenarioConfig
from src.envs.nocrash.environment.config.reward_configs import BaseRewardConfig


episode_measurements = {
    "episode_id": None,
    "num_steps": None,
    "location": None,
    "speed": None,
    "distance_to_goal": None,
    "num_collisions": 0,
    "num_laneintersections": 0,
    "static_steps": 0,
    "offlane_steps": 0,
    "control_steer": 0
}


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


VEHICLE_TYPES = ['vehicle.ford.mustang', 'vehicle.audi.a2', 'vehicle.audi.tt', 'vehicle.bmw.isetta', 'vehicle.carlamotors.carlacola',
                      'vehicle.citroen.c3', 'vehicle.bmw.grandtourer', 'vehicle.mercedes-benz.coupe',
                      'vehicle.toyota.prius', 'vehicle.dodge_charger.police', 'vehicle.nissan.patrol',
                      'vehicle.tesla.model3', 'vehicle.seat.leon', 'vehicle.lincoln.mkz2017',
                      'vehicle.volkswagen.t2', 'vehicle.nissan.micra', 'vehicle.chevrolet.impala', 'vehicle.mini.cooperst',
                      'vehicle.jeep.wrangler_rubicon'],


SPAWN_POINTS_FIXED_IDX = [ 54, 234, 108,  12, 175,  71, 116,  99, 196,  63, 205,  46,  96,
    246, 128, 106, 143,  39,  72, 176, 140, 138,  91,  88, 241,  29,
    28, 238, 119, 221, 163,  81,  47, 255, 235,  64, 216, 151, 145,
    77,  35,  56,  68,  49, 154, 149, 201,  27, 212, 195, 230, 157,
        3,   5,  20, 193,   6,  90,  18,  13, 139,  44, 122, 220, 125,
    115,  43,   4, 213,  30,  62, 242, 219, 171,  41, 203,  57, 248,
    204, 226, 245, 135, 164, 153,  14, 188,   7, 123, 117, 222, 183,
    152, 150, 185, 224,  19, 104, 111,  82,  79,   0,  33,  38, 146,
    10, 173, 239,  32, 228, 209, 243, 200, 215, 236,  34,  84,  51,
    73,  53, 170, 217, 237, 102, 156,  45, 253,  37, 210, 118,  86,
    74,  61, 165, 179, 202, 101,  36, 132, 168, 137, 126, 178,  24,
        1, 247, 107,  93, 148,  50,  98,  87, 133, 162,   2, 214, 124,
    112, 211,  75, 121, 191, 113, 141,  26, 231, 174,  76, 207, 109,
    244, 129, 103,  52,  42,  55, 180,  89, 181,  69,  48,  21,  16,
    198,  66,  70, 130, 114,  15, 134,  40, 227, 223,  67,  78, 159,
    252, 147,  17, 166,  11, 131, 161, 105, 167,  95, 172, 233, 251,
    194,  60,  80, 182,  97,  59, 197,  25, 186, 136, 160, 120, 158,
    189, 192, 190, 187, 142, 232,   9, 127, 206, 169,  23, 208,  94,
    218,  83, 155,  65, 254, 249,  92, 240,  85, 100,  58,  22,   8,
    225,  31, 229, 250, 110, 177, 199, 184, 144]


class BaseMainConfig(BaseConfig):
    """Base Class defining the parameters required in main config.

    DO NOT instantiate this directly. Instead, using DefaultMainConfig
    """
    def __init__(self):
        #TODO These are all parameters that need to be set at each run
        self.reward_config = None
        self.scenario_config = None
        self.obs_config = None
        self.action_config = None

        # Are we testing?
        self.testing = None


        #### Server Setup ####
        self.server_path = None
        self.server_binary = None
        self.server_fps = None
        self.server_port = None
        self.server_retries = None
        self.sync_mode = None
        self.client_timeout_seconds = None
        self.carla_gpu = None

        self.render_server = None
        # X Rendering Resolution
        self.render_res_x = None
        # Y Rendering Resolution
        self.render_res_y = None

        # Input X Res (Default set to Atari)
        self.x_res = None
        # Input Y Res (Default set to Atari)
        self.y_res = None

        #### Logging related parameters ####
        self.print_obs = None
        self.log_measurements_to_file = None

        # TODO remove this parameter
        self.log_freq = None
        self.verbose = None
        self.videos = None


        #### UNKNOWN CLASSIFICATION ####
        #TODO MOVE THESE TO THE CORRECT PLACES


        self.test_fixed_spawn_points = None
        self.train_fixed_spawn_points = None
        self.spawn_points_fixed_idx = None
        self.vehicle_types = None



    def populate_config(self, observation_config = 'LowDimObservationConfig', action_config = 'MergedSpeedScaledTanhConfig', reward_config = 'Simple2RewardConfig', scenario_config = 'NoCrashEmptyTown01Config', testing = False, carla_gpu = 0):
        """Fill in the config parameters that are not set by default

        For each type of config, the parameter can be either passed in as a string containing the class name or
        an instance of the config type. The config must be a subclass of the respective config base class.
        Ex: observation_config can be a string "LowDimObsConfig" or it can be a class (or subclass) of type BaseObsConfig
        """
        # Observation Config
        if(isinstance(observation_config, str)):
            # Get reference to object
            config_type = getattr(observation_configs, observation_config)

            # Instantiate Object
            self.obs_config = config_type()
        elif(isinstance(observation_config, BaseObservationConfig)):
            # Just save object, since it is already instantiated
            self.obs_config = observation_config
        else:
            # Invalid Argument
            raise Exception("Invalid argument for observation_config")

        # Action Config
        if(isinstance(action_config, str)):
            # Get reference to object
            config_type = getattr(action_configs, action_config)

            # Instantiate Object
            self.action_config = config_type()
        elif(isinstance(action_config, BaseActionConfig)):
            # Just save object, since it is already instantiated
            self.action_config = action_config
        else:
            # Invalid Argument
            raise Exception("Invalid argument for action_config")

        # Reward Config
        if(isinstance(reward_config, str)):
            # Get reference to object
            config_type = getattr(reward_configs, reward_config)

            # Instantiate Object
            self.reward_config = config_type()
        elif(isinstance(reward_config, BaseRewardConfig)):
            # Just save object, since it is already instantiated
            self.reward_config = reward_config
        else:
            # Invalid Argument
            raise Exception("Invalid argument for reward_config")

        # Scenario Config
        if(isinstance(scenario_config, str)):
            # Get reference to object
            config_type = getattr(scenario_configs, scenario_config)

            # Instantiate Object
            self.scenario_config = config_type()
        elif(isinstance(scenario_config, BaseScenarioConfig)):
            # Just save object, since it is already instantiated
            self.scenario_config = scenario_config
        else:
            # Invalid Argument
            raise Exception("Invalid argument for scenario_config")

        # Testing
        self.testing = testing

        # Carla GPU
        self.carla_gpu = carla_gpu


class DefaultMainConfig(BaseMainConfig):
    """Default Config for the server
    """
    def __init__(self):
        super().__init__()
        #### Server Setup ####
        self.server_path = CARLA_9_11_PATH
        self.server_binary = CARLA_9_11_PATH + '/CarlaUE4.sh'
        self.server_fps = 10
        self.server_port = -1
        self.server_retries = 5
        self.sync_mode = True
        self.client_timeout_seconds = 30


        self.render_server = True
        # X Rendering Resolution
        self.render_res_x = 800
        # Y Rendering Resolution
        self.render_res_y = 800

        # Input X Res (Default set to Atari)
        self.x_res = 80
        # Input Y Res (Default set to Atari)
        self.y_res = 160

        #### Logging related parameters ####
        self.print_obs = True
        self.log_measurements_to_file = False

        self.log_freq = 1
        self.verbose = False
        self.videos = False


        #### UNKNOWN CLASSIFICATION ####
        self.test_fixed_spawn_points = True
        self.train_fixed_spawn_points = False
        self.spawn_points_fixed_idx = SPAWN_POINTS_FIXED_IDX
        self.vehicle_types = VEHICLE_TYPES
