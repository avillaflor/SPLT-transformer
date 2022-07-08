import gym
import numpy as np


class ToyCar(gym.Env):
    def __init__(self):
        self.max_vel = 10.

        self.ego_x = 0.
        self.ego_vel = self.max_vel
        self.other_x = 10.
        self.other_vel = self.max_vel
        self.t = 0

        self.dt = 0.25
        self.sign_x = 70.  # stop sign for other vehicle
        self.max_t = 10.
        self.crash_dist = 5.
        self.crash_penalty = self.max_vel * self.max_t

        self.min_a = -1.
        self.max_a = 1.

        self.other_stopping = True
        self.z_p = 1

        # IDM parameters
        self.idm_params = {
            'v_0': self.max_vel,
            'T': 1.5,
            'a': 1.0,
            'b': 1.0,
            'delta': 4.,
            's_0': 2.,
        }

        self.testing_index = 0

        self.observation_space = gym.spaces.Box(
            low=np.array([0., -np.inf, 0., -np.inf]),
            high=np.array([self.max_vel, self.crash_penalty, self.max_vel, np.inf]),
        )
        self.action_space = gym.spaces.Box(
            low=np.array([self.min_a]),
            high=np.array([self.max_a]),
        )

    @property
    def _other_stop_dist(self):
        if self.other_stopping:
            min_dist = 0.5 * self.other_vel * (self.other_vel - self.dt)
            safe_dist = self.dt * self.other_vel + min_dist
            return safe_dist
        else:
            return -np.inf

    @property
    def idm_T(self):
        return self.idm_params['T'] * self.z_p

    def step(self, a):
        if isinstance(a, np.ndarray):
            a = a[0]
        a = np.clip(a, a_min=self.min_a, a_max=self.max_a)
        self.ego_x = self.ego_x + self.dt * self.ego_vel
        self.ego_vel = np.clip(self.ego_vel + self.dt * a, a_min=0., a_max=self.max_vel)

        other_sign_dist = abs(self.sign_x - self.other_x)
        if other_sign_dist <= self._other_stop_dist:
            other_a = -1.0
        else:
            other_a = 1.0
        self.other_x = self.other_x + self.dt * self.other_vel
        self.other_vel = np.clip(self.other_vel + self.dt * other_a, a_min=0., a_max=self.max_vel)

        if self.other_vel == 0.:
            self.other_stopping = False

        self.t += self.dt

        obs = np.array([self.ego_x, self.ego_vel, self.other_x, self.other_vel])

        dist = self.other_x - self.ego_x
        crash = dist <= self.crash_dist

        end = self.t >= self.max_t

        r = self.dt * self.ego_vel
        if crash:
            r -= self.crash_penalty
            print('Crashed')

        done = crash or end
        info = {}
        if done:
            info['success'] = ~crash
        return obs, r, done, info

    def autopilot(self):
        # IDM
        safe_other_vel = self.other_vel
        delta_v = self.ego_vel - safe_other_vel
        delta_x = self.other_x - self.ego_x - self.crash_dist
        dv_term = (self.ego_vel * delta_v) / (2 * np.sqrt(self.idm_params['a'] * self.idm_params['b']))
        s_star = self.idm_params['s_0'] + self.ego_vel * self.idm_T + dv_term
        acc = self.idm_params['a'] * (1 - (self.ego_vel / self.idm_params['v_0']) ** self.idm_params['delta'] - (s_star / delta_x) ** 2)
        acc = np.clip(acc, a_min=-0.99, a_max=0.99)
        return acc

    def reset(self, testing=False):
        if testing:
            temp_state = np.random.get_state()
            np.random.seed(self.testing_index)

        start_vel = self.max_vel * 0.25 * (np.random.rand() + 3.0)
        self.ego_x = 0.
        self.ego_vel = self.max_vel
        self.other_x = 10. + np.random.rand() * 10.
        self.other_vel = start_vel
        self.t = 0

        # distribution used to collect dataset
        self.z_p = np.random.uniform(0., 7.)
        # best idm controller
        #  self.z_p = 0.4

        if testing:
            self.other_stopping = (self.testing_index % 2) == 0
            self.testing_index += 1
            np.random.set_state(temp_state)
        else:
            self.other_stopping = np.random.rand() < 0.5

        obs = np.array([self.ego_x, self.ego_vel, self.other_x, self.other_vel])
        return obs
