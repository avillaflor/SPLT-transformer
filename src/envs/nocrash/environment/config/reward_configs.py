from src.envs.nocrash.environment.config.base_config import BaseConfig


class BaseRewardConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Speed reward coefficient
        self.speed_coeff = None

        # Acceleration reward coefficient
        self.acceleration_coeff = None

        # Coefficient for dist_to_trajec reward
        # Pass a positive value for this argument
        self.dist_to_trajectory_coeff = None

        # Penalty for collision
        self.const_collision_penalty = None

        # Penalty for collision proportional to speed
        self.collision_penalty_speed_coeff = None

        # Penalty for red light violation
        self.const_light_penalty = None

        # Penalty for red light infraction proportional to speed
        self.light_penalty_speed_coeff = None

        # Penalty for steer reward
        self.steer_penalty_coeff =  None

        # Reward for success completion of trajectory
        self.success_reward = None

        # Constant reward given at every time step
        self.constant_positive_reward = None

        # Factor to normalize rewards (reward is divided by this value)
        self.reward_normalize_factor = None


class Simple2RewardConfig(BaseRewardConfig):
    def __init__(self):
        # Speed reward coefficient
        self.speed_coeff = 1

        # Speed^2 reward coefficient
        self.speed_squared_coeff = 0

        # Acceleration reward coefficient
        self.acceleration_coeff = 0

        # Coefficient for dist_to_trajec reward
        # Pass a positive value for this argument
        self.dist_to_trajectory_coeff = 1

        # Penalty for collision
        self.const_collision_penalty = 250

        # Penalty for collision proportional to speed
        self.collision_penalty_speed_coeff = 0

        # Penalty for red light violation
        self.const_light_penalty = 250

        # Penalty for red light infraction proportional to speed
        self.light_penalty_speed_coeff = 0

        # Penalty for steer reward
        self.steer_penalty_coeff = 0

        # Reward for success completion of trajectory
        self.success_reward = 0

        # Constant reward given at every time step
        self.constant_positive_reward = 0

        # Factor to normalize rewards (reward is divided by this value)
        self.reward_normalize_factor = 1


class SpeedRewardConfig(BaseRewardConfig):
    def __init__(self):
        # Speed reward coefficient
        self.speed_coeff = 0

        # Speed^2 reward coefficient
        self.speed_squared_coeff = 10.

        # Acceleration reward coefficient
        self.acceleration_coeff = 0

        # Coefficient for dist_to_trajec reward
        # Pass a positive value for this argument
        self.dist_to_trajectory_coeff = 1.0

        # Penalty for collision
        self.const_collision_penalty = 250

        # Penalty for collision proportional to speed
        self.collision_penalty_speed_coeff = 0

        # Penalty for red light violation
        self.const_light_penalty = 250

        # Penalty for red light infraction proportional to speed
        self.light_penalty_speed_coeff = 0

        # Penalty for steer reward
        self.steer_penalty_coeff = 0

        # Reward for success completion of trajectory
        self.success_reward = 0

        # Constant reward given at every time step
        self.constant_positive_reward = 0

        # Factor to normalize rewards (reward is divided by this value)
        self.reward_normalize_factor = 1
