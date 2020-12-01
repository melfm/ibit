import copy
import math
import os
import random
from collections import deque
from collections import OrderedDict
import re

import numpy as np
import scipy.linalg as sp_la

import gym
import dmc2gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
from skimage.util.shape import view_as_windows
from torch import distributions as pyd
from gym.spaces import Dict, Box

try:
    import gym
    from gym.spaces import Dict, Box
    from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT
    from metaworld.envs.mujoco.env_dict import RAND_MODE_ARGS_KWARGS, RAND_MODE_CLS_DICT

    mtw_envs_hard = {
        **HARD_MODE_CLS_DICT["train"],
        **HARD_MODE_CLS_DICT["test"]
    }
    mtw_args_hard = {
        **HARD_MODE_ARGS_KWARGS["train"],
        **HARD_MODE_ARGS_KWARGS["test"]
    }

    mtw_envs_rand = {
        **RAND_MODE_CLS_DICT["train"],
        **RAND_MODE_CLS_DICT["test"]
    }
    mtw_args_rand = {
        **HARD_MODE_ARGS_KWARGS["train"],
        **HARD_MODE_ARGS_KWARGS["test"]
    }

except:
    mtw_envs_hard = None
    mtw_args_hard = None

    mtw_envs_rand = None
    mtw_args_rand = None

debug_mode = False

if debug_mode:
    import glfw
    from gym.envs.registration import registry, register, make, spec
    from pyglet.window import key

    action_to_take = np.zeros(6)


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def get_latest_file(path):
    files = os.listdir(path)
    max_ckpt = 0
    for ckpt in files:
        step = int(re.findall(r'\d+', ckpt)[0])
        if step > max_ckpt:
            max_ckpt = int(step)
    # print('Loading Latest ', max_ckpt)
    return max_ckpt


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def irm_penalty(logits, labels):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = F.mse_loss(logits * scale, labels)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, metaworld_task=True):
        gym.Wrapper.__init__(self, env)
        self.frame_size = 84
        self._k = k
        self._frames = deque([], maxlen=k)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=[3, self.frame_size, self.frame_size],
            dtype=np.uint8)
        shp = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k, ) + shp[1:]),
            dtype=env.observation_space.dtype)
        if hasattr(env, 'max_path_length'):
            self._max_episode_steps = env.max_path_length
        else:
            env.max_path_length = 150
            self._max_episode_steps = env.max_path_length
        self.metaworld_task = metaworld_task

    def reset(self):
        state_obs = self.env.reset()
        if self.metaworld_task:
            obs = self.env.render(mode='rgb_array',
                                  width=self.frame_size,
                                  height=self.frame_size)
        else:
            obs = self.env.render(mode='rgb_array',
                                  width=self.frame_size,
                                  height=self.frame_size,
                                  camera_id='front_far')
        obs = obs.reshape(3, obs.shape[0], obs.shape[1])
        for _ in range(self._k):
            self._frames.append(obs)
        pixel_obs = self._get_obs()
        obs = {'pix_obs': pixel_obs, 'state_low_obs': state_obs}
        return obs

    def step(self, action):
        state_obs, reward, done, info = self.env.step(action)
        # obs = self.env.get_image()
        if self.metaworld_task:
            obs = self.env.render(mode='rgb_array',
                                  width=self.frame_size,
                                  height=self.frame_size)
        else:
            obs = self.env.render(mode='rgb_array',
                                  width=self.frame_size,
                                  height=self.frame_size,
                                  camera_id='front_far')
        obs = obs.reshape(3, obs.shape[0], obs.shape[1])
        self._frames.append(obs)
        pixel_obs = self._get_obs()
        obs = {'pix_obs': pixel_obs, 'state_low_obs': state_obs}
        return obs, reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class FrameStackJacoReal(gym.Wrapper):
    def __init__(self, env, k, frame_size=84, dummy_env=None):
        self.frame_size = 84
        env.action_space = dummy_env.action_space
        env.observation_space = Box(
            low=0,
            high=255,
            shape=[3, self.frame_size, self.frame_size],
            dtype=np.uint8)
        shp = env.observation_space.shape
        env.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k, ) + shp[1:]),
            dtype=env.observation_space.dtype)
        env.reward_range = (-float('inf'), float('inf'))
        env.metadata = {'render.modes': []}
        gym.Wrapper.__init__(self, env)

        self._k = k
        self._frames = deque([], maxlen=k)

    def render(self):
        pixel_obs = self.env.render(height=640, width=480)
        return pixel_obs

    def reset(self):
        _ = self.env.reset()
        obs = self.env.render(height=self.frame_size, width=self.frame_size)
        obs = obs.reshape(3, obs.shape[0], obs.shape[1])
        for _ in range(self._k):
            self._frames.append(obs)
        pixel_obs = self._get_obs()
        obs = {'pix_obs': pixel_obs, 'state_low_obs': None}
        return obs

    def step(self, action):
        _ = self.env.step(action)
        obs = self.env.render(height=self.frame_size, width=self.frame_size)
        obs = obs.reshape(3, obs.shape[0], obs.shape[1])
        self._frames.append(obs)
        pixel_obs = self._get_obs()
        obs = {'pix_obs': pixel_obs, 'state_low_obs': None}
        return obs

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def on_press(window, key, scancode, action, mods):
    global action_to_take
    pos = np.zeros(6)
    _pos_step = 0.2

    # controls for moving position
    if key == glfw.KEY_A:
        pos[1] -= _pos_step  # dec x
    elif key == glfw.KEY_D:
        pos[1] += _pos_step  # inc x
    elif key == glfw.KEY_W:
        pos[0] -= _pos_step  # dec y
    elif key == glfw.KEY_S:
        pos[0] += _pos_step  # inc y
    elif key == glfw.KEY_DOWN:
        pos[2] -= _pos_step  # dec z
    elif key == glfw.KEY_UP:
        pos[2] += _pos_step  # inc z

    elif key == glfw.KEY_LEFT:
        pos[3:] = 0
    elif key == glfw.KEY_RIGHT:
        pos[3:] = 1
    elif key == glfw.KEY_ESCAPE:
        exit()

    action_to_take = pos


class EnvSampler():
    """Env Class used for sampling environments with different
        simulation parameters.
    """
    def __init__(self, cfg, work_dir):
        self.cfg = cfg

        self.env_name = self.cfg.env

        self.metaworld_env = False
        if self.env_name == 'jaco_reach_site_features':
            self.domain_name = 'jaco'
            self.task_name = 'reach_site_features'

        elif self.env_name == 'jaco_reach_duplo_features':
            self.domain_name = 'jaco'
            self.task_name = 'reach_duplo_features'

        elif self.env_name == 'ball_in_cup_catch':
            self.domain_name = 'ball_in_cup'
            self.task_name = 'catch'
        elif self.env_name == 'point_mass_easy':
            self.domain_name = 'point_mass'
            self.task_name = 'easy'
        else:
            self.metaworld_env = True

        self.num_envs = cfg.num_envs
        self.apply_mod = cfg.apply_mod
        self.remote_render = cfg.remote_render

        if self.env_name == 'button-press-v1' or self.env_name == 'drawer-close-v1':
            self.view_setting = [180, -20, 1.66]
        elif self.env_name == 'door-open-v1':
            self.view_setting = [150, -20, 1.66]
        elif self.env_name == 'window-open-v1' or self.env_name == 'window-close-v1':
            self.view_setting = [150, -20, 1.66]
        else:
            self.view_setting = [-70, -10, 1.66]

        self.work_dir = work_dir

        self.table_tags = OrderedDict([('table_dark_wood', 0),
                                       ('table_marble', 1), ('table_blue', 2),
                                       ('table_tennis', 3), ('table_wood', 4),
                                       ('table_light_wood_v3', 5),
                                       ('table_light_wood_v2', 6),
                                       ('default', 7)])

        self.goal_tags = OrderedDict([('goal_red', 0), ('goal_yellow', 1),
                                      ('goal_blue', 2), ('goal_pink', 3),
                                      ('default', 4)])
        self.sky_tags = OrderedDict([('default', 0), ('red_star', 1),
                                     ('orange_star', 2), ('yellow_star', 3),
                                     ('pink_star', 4), ('amber_star', 5),
                                     ('black_star', 6), ('default', 7)])

        self.eval_env_tags = OrderedDict([('table_granite_goal_purple', 0)])

    def make_env(self, args=None, kwargs=None, dm_task_name=None):
        """Create dm_control/metaworld environment"""

        if self.metaworld_env:
            env = mtw_envs_rand[self.env_name](*args, **kwargs)

            if debug_mode:
                env._max_episode_steps = 10000

                env.reset()
                env.render()
                global action_to_take
                glfw.set_key_callback(env.unwrapped.viewer.window, on_press)

                while True:
                    env.render()

                    if not np.array_equal(action_to_take, np.zeros(6)):
                        _, _, d, _ = env.step(action_to_take)
                        if d:
                            env.seed(args.seed)
                            env.reset()
                            env.render()

                        # Commenting this out makes the mocap faster but
                        # introduces some instabilities.
                        # action_to_take = np.zeros(6)
        else:

            camera_id = 2 if self.domain_name == 'quadruped' else 0
            if dm_task_name is not None:
                task_name = dm_task_name
            else:
                task_name = self.task_name
            env = dmc2gym.make(domain_name=self.domain_name,
                               task_name=task_name,
                               seed=self.cfg.seed,
                               visualize_reward=False,
                               from_pixels=False,
                               height=self.cfg.image_size,
                               width=self.cfg.image_size,
                               frame_skip=self.cfg.action_repeat,
                               camera_id=camera_id)

            if debug_mode:
                from dm_control import viewer
                viewer.launch(env)

        env = FrameStack(env, k=self.cfg.frame_stack)
        env.seed(self.cfg.seed)

        return env

    def sample_all_train_envs(self, experiment_id, num_envs=12):

        train_envs = OrderedDict()
        if self.metaworld_env:
            for _ in range(num_envs):
                # augmentation_toggle = str(random.choice([0, 1]))
                augmentation_toggle = str(1)
                table_color = str(random.choice(list(
                    self.table_tags.values())))
                goal_color = str(random.choice(list(self.goal_tags.values())))
                sky_color = str(random.choice(list(self.sky_tags.values())))
                # Sample a new env
                args = copy.copy(mtw_args_rand[self.env_name]["args"])
                kwargs = copy.copy(mtw_args_rand[self.env_name]["kwargs"])

                if self.cfg.goal_mode == 'single_goal':
                    # Single goal location
                    kwargs["random_init"] = False
                elif self.cfg.goal_mode == 'multi_goal':
                    # Random goal locations
                    kwargs["random_init"] = True
                else:
                    raise ValueError('Invalid goal mode.')
                # This is only for low-level state obs
                # just_goal : Gives only the goal location
                # with_goal : Gives low level obs + goal
                if self.cfg.lowobs_append:
                    kwargs["obs_type"] = "with_goal"
                # elif self.cfg.goal_append:
                #     kwargs["obs_type"] = "just_goal"
                mod_id = table_color + goal_color + sky_color
                kwargs["intervention_id"] = mod_id
                kwargs["apply_mod"] = self.apply_mod
                kwargs["remote_render"] = self.remote_render
                kwargs["view_setting"] = self.view_setting
                kwargs["experiment_id"] = experiment_id
                env_id = augmentation_toggle + mod_id + str(
                    random.randint(10, 1000))
                env = self.make_env(args, kwargs)
                train_envs.update({env_id: env})
        else:
            for i in range(num_envs):
                env = self.make_env()
                env_tag = env.get_mod_tag
                env_name = str(i) + env_tag
                train_envs.update({env_name: env})

        return train_envs

    def sample_eval_envs(self, experiment_id=None, num_envs=3):
        env_tags = [0, 1, 2]
        eval_envs = OrderedDict()

        if self.metaworld_env:
            for env_id in env_tags:
                args = copy.copy(mtw_args_rand[self.env_name]["args"])
                kwargs = copy.copy(mtw_args_rand[self.env_name]["kwargs"])
                if self.cfg.goal_mode == 'single_goal':
                    # Single goal location
                    kwargs["random_init"] = False
                elif self.cfg.goal_mode == 'multi_goal':
                    # Random goal locations
                    kwargs["random_init"] = True
                else:
                    raise ValueError('Invalid goal mode.')
                if self.cfg.lowobs_append:
                    kwargs["obs_type"] = "with_goal"
                # elif self.cfg.goal_append:
                #     kwargs["obs_type"] = "just_goal"
                kwargs["phase"] = "eval"
                # Always randomize rendering for eval
                kwargs["apply_mod"] = True
                kwargs["intervention_id"] = str(env_id)
                kwargs["remote_render"] = self.remote_render
                kwargs["view_setting"] = self.view_setting
                kwargs["experiment_id"] = experiment_id
                env = self.make_env(args, kwargs)
                eval_envs.update({env_id: env})
        else:
            for i in range(num_envs):
                env = self.make_env(dm_task_name='reach_site_features_eval')
                env_tag = env.get_mod_tag
                env_name = str(i) + env_tag
                eval_envs.update({env_name: env})

        return eval_envs


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu