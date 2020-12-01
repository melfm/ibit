import numpy as np
import random

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import os, sys
from typing import List, Optional, Union


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class MultiEnvReplayBuffer(object):
    """Buffer to store environment transitions for multiple environments"""
    def __init__(self,
                 obs_shape,
                 action_shape,
                 capacity,
                 image_pad,
                 device,
                 image_augmentation,
                 num_envs: int,
                 state_append=False,
                 state_lstate_shape=3):

        self.env_id_to_replay_buffer_map = [
            ReplayBuffer(
                obs_shape=obs_shape,
                action_shape=action_shape,
                capacity=int(capacity / num_envs),
                device=device,
                image_pad=image_pad,
                image_augmentation=image_augmentation,
                state_append=state_append,
                state_lstate_shape=state_lstate_shape,
            ) for _ in range(num_envs)
        ]
        self.capacity = capacity
        self.num_envs = num_envs
        self.image_augmentation = image_augmentation

    def add(self, env_id, obs, action, reward, next_obs, done, done_no_max):
        self.env_id_to_replay_buffer_map[env_id].add(obs, action, reward,
                                                     next_obs, done,
                                                     done_no_max)

    def sample(self, batch_size, env_tag, env_id: Optional[int] = None):
        # env_tag is the tag name of env
        # env_id is numerical index for lookups
        if env_id is None:
            env_id = random.randint(0, self.num_envs - 1)
        # TODO: Clean way to disabling this completely, currently it can be randomized
        image_augmentation = self.image_augmentation  #bool(env_tag[0])
        return self.env_id_to_replay_buffer_map[env_id].sample(
            batch_size, image_augmentation)

    def save(self, save_dir):
        for replay_buffer in self.env_id_to_replay_buffer_map:
            replay_buffer.save(save_dir)

    def load(self, save_dir):
        for replay_buffer in self.env_id_to_replay_buffer_map:
            replay_buffer.load(save_dir)

    def __len__(self):
        return self.env_id_to_replay_buffer_map[
            0].idx if not self.env_id_to_replay_buffer_map[
                0].full else self.capacity


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self,
                 obs_shape,
                 action_shape,
                 capacity,
                 image_pad,
                 device,
                 image_augmentation,
                 state_lstate_shape=3,
                 state_append=False):
        self.capacity = capacity
        self.device = device

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.image_augmentation = image_augmentation
        self.state_append = state_append
        if self.state_append:
            state_lstate_shape_tuple=(1, state_lstate_shape)
            self.state_low_obs = np.empty((capacity, *state_lstate_shape_tuple),
                                        dtype=np.float32)
            self.next_state_low_obs = np.empty((capacity, *state_lstate_shape_tuple),
                                             dtype=np.float32)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs['pix_obs'])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs['pix_obs'])
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        if self.state_append:
            np.copyto(self.state_low_obs[self.idx], obs['state_low_obs'])
            np.copyto(self.next_state_low_obs[self.idx], next_obs['state_low_obs'])

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, image_augmentation):

        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        state_low_obs = None
        next_obses = self.next_obses[idxs]
        next_state_low_obs = None
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        if self.state_append:
            state_low_obs = self.state_low_obs[idxs]
            state_low_obs = torch.as_tensor(state_low_obs,
                                          device=self.device).float()
            next_state_low_obs = self.next_state_low_obs[idxs]
            next_state_low_obs = torch.as_tensor(next_state_low_obs,
                                               device=self.device).float()

        obses = torch.as_tensor(obses, device=self.device).float()

        cat_obses = {"pix_obs": obses, "state_low_obs": state_low_obs}
        next_obses = torch.as_tensor(next_obses, device=self.device).float()

        next_cat_obses = {
            "pix_obs": next_obses,
            "state_low_obs": next_state_low_obs
        }
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug,
                                         device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        if image_augmentation:
            with HiddenPrints():
                obses = self.aug_trans(obses)
                next_obses = self.aug_trans(next_obses)

                obses_aug = self.aug_trans(obses_aug)
                next_obses_aug = self.aug_trans(next_obses_aug)
        else:
            obses = obses
            next_obses = next_obses

            obses_aug = obses_aug
            next_obses_aug = next_obses_aug

        return cat_obses, actions, rewards, next_cat_obses, not_dones_no_max, obses_aug, next_obses_aug
