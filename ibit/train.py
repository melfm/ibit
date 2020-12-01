import copy
import math
import os
import pickle as pkl
import sys
import time
import random
import shutil

import numpy as np

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
from replay_buffer import MultiEnvReplayBuffer
from video import VideoRecorder
from omegaconf import DictConfig

torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        config_file = self.work_dir.split('runs')[0] + 'configs/' \
             + cfg.env.replace('-', '_') + '.yaml'
        shutil.copy(config_file, self.work_dir)

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat,
                             overwrite=True)

        experiment_identifier = self.work_dir.split('runs')[1]
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # Interventions
        interventions = cfg.internvention
        image_augmentation = False

        if 'type_1' in interventions:
            image_augmentation = True
        if 'type_2' in interventions:
            cfg.apply_mod = True

        # Environment Sampler
        self.num_train_envs = cfg.num_envs
        self.env_sampler = utils.EnvSampler(cfg, work_dir=self.work_dir)
        self.eval_envs = self.env_sampler.sample_eval_envs(
            experiment_identifier)
        self.train_envs = self.env_sampler.sample_all_train_envs(
            experiment_identifier)

        self.resample_envs = cfg.resample_env
        self.env_resample_rate = cfg.env_resample_rate

        self.render_train_samples = True

        if self.render_train_samples:
            if cfg.env.startswith('jaco'):
                height = 256
                width = 256
            else:
                height = width = 500
            from PIL import Image
            for env_idx, env in self.train_envs.items():
                name = 'Environment_' + str(env_idx) + '.png'
                env.reset()
                obs = env.render(mode='rgb_array', height=height, width=width)
                im = Image.fromarray(obs)
                im.save(name)
            for env_idx, env in self.eval_envs.items():
                name = 'Eval_Unseen_Environment_' + str(env_idx) + '.png'
                env.reset()
                obs = env.render(mode='rgb_array', height=height, width=width)
                im = Image.fromarray(obs)
                im.save(name)

        env_sample_key = list(self.eval_envs.keys())[0]
        sample_env = self.eval_envs[env_sample_key]
        cfg.agent.params.obs_shape = sample_env.observation_space.shape
        cfg.agent.params.action_shape = sample_env.action_space.shape
        cfg.agent.params.action_range = [
            float(sample_env.action_space.low.min()),
            float(sample_env.action_space.high.max())
        ]
        state_append = cfg.lowobs_append
        if state_append:
            if cfg.env == 'window-open-v1':
                # Double check this
                cfg.agent.params.lstate_shape = 9
            elif cfg.env == 'jaco_reach_site_features':
                cfg.agent.params.lstate_shape = 49
            else:
                cfg.agent.params.lstate_shape = 9
        else:
            cfg.agent.params.lstate_shape = 0

        self.agent = hydra.utils.instantiate(cfg.agent)
        self.replay_buffer = MultiEnvReplayBuffer(
            sample_env.observation_space.shape,
            sample_env.action_space.shape,
            cfg.replay_buffer_capacity,
            self.cfg.image_pad,
            self.device,
            image_augmentation,
            num_envs=self.num_train_envs,
            state_append=state_append,
            state_lstate_shape=cfg.agent.params.lstate_shape)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.train_video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None, phase='train')
        self.model_dir = self.work_dir + '/agent_model'
        self.step = [0] * self.num_train_envs
        self.reload_weights = cfg.reload_weights

        self.train_vid_interval = cfg.train_vid_interval

    def evaluate(self, phase, eval_env):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = eval_env.reset()
            if phase == 'unseen':
                self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            # not done doesnt work for metaworld
            while (episode_step <= eval_env._max_episode_steps - 1):
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = eval_env.step(action)
                if phase == 'unseen':
                    self.video_recorder.record(eval_env)
                episode_reward += reward
                episode_step += 1
                if done: break
            average_episode_reward += episode_reward
            if phase == 'unseen':
                self.video_recorder.save(f'{self.step[0]}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        if phase == 'seen':
            self.logger.log('eval_seen/episode_reward', average_episode_reward,
                            self.step[0])
            self.logger.dump(self.step[0], ty='eval_seen')
        elif phase == 'unseen':
            self.logger.log('eval_unseen/episode_reward',
                            average_episode_reward, self.step[0])
            self.logger.dump(self.step[0], ty='eval_unseen')
        eval_env.reset()

    def run(self):
        init_env = None
        keys_to_sample = random.sample(list(self.train_envs),
                                       self.num_train_envs)
        sampled_train_envs = {
            key: self.train_envs[key]
            for key in keys_to_sample
        }
        # Better way to access first elem of OrderedDict?
        for env_idx, env in sampled_train_envs.items():
            init_env = env
            break
        episode, episode_reward, episode_step, done = [0] * self.num_train_envs, [0] * self.num_train_envs, \
            [0] * self.num_train_envs, [True] * self.num_train_envs
        obs, next_obs = [init_env.reset()] * self.num_train_envs, [
            init_env.reset()
        ] * self.num_train_envs
        start_time = time.time()

        train_recording = False
        env_to_rec = 0

        if self.reload_weights and os.path.exists(self.model_dir):
            # Continue training
            try:
                latest_step = utils.get_latest_file(self.model_dir)
                self.agent.load(self.model_dir, latest_step)
            except:
                print('Could not reload weights!')
        while self.step[0] < self.cfg.num_train_steps:

            if self.resample_envs and self.step[
                    0] > 0 and self.step[0] % self.env_resample_rate == 0:
                keys_to_sample = random.sample(list(self.train_envs),
                                               self.num_train_envs)
                sampled_train_envs = {
                    key: self.train_envs[key]
                    for key in keys_to_sample
                }

            for env_idx, (env_tag,
                          env) in enumerate(sampled_train_envs.items()):
                episode_step[env_idx] = 0
                while (episode_step[env_idx] <= env._max_episode_steps - 1):
                    if not train_recording and env_idx == env_to_rec and self.step[
                            env_idx] % self.train_vid_interval == 0:
                        train_recording = True
                        self.train_video_recorder.init(enabled=True)

                    if done[env_idx] or (episode_step[env_idx] >=
                                         env._max_episode_steps - 1):
                        if self.step[env_idx] > 0:
                            self.logger.log('train/duration',
                                            time.time() - start_time,
                                            self.step[env_idx])
                            start_time = time.time()

                        if self.step[
                                env_idx] > 0 and env_idx == env_to_rec and train_recording:
                            file_name = str(self.step[env_idx]) + '_' + env_tag
                            self.train_video_recorder.save(f'{file_name}.mp4')
                            self.train_video_recorder.frames = []
                            train_recording = False
                            env_to_rec = random.randint(
                                0,
                                len(sampled_train_envs) - 1)

                        # Evaluate agent periodically
                        if env_idx == 0 and episode[
                                env_idx] % self.cfg.eval_frequency == 0:
                            # Evaluate an env from training
                            self.logger.log('eval_seen/episode',
                                            episode[env_idx],
                                            self.step[env_idx])
                            eval_env = self.train_envs[random.sample(
                                list(self.train_envs), 1)[0]]
                            self.evaluate(phase='seen', eval_env=eval_env)
                            # Evaluate an unseen env
                            self.logger.log('eval_unseen/episode',
                                            episode[env_idx],
                                            self.step[env_idx])
                            eval_env = self.eval_envs[random.sample(
                                list(self.eval_envs), 1)[0]]
                            self.evaluate(phase='unseen', eval_env=eval_env)
                        if episode[env_idx] % self.cfg.ckpt_frequency == 0:
                            self.agent.save(self.model_dir, episode[env_idx])
                        self.logger.log('train/episode_reward',
                                        episode_reward[env_idx],
                                        self.step[env_idx])
                        obs[env_idx] = env.reset()
                        done[env_idx] = False
                        episode_reward[env_idx] = 0
                        episode[env_idx] += 1

                        self.logger.log('train/episode', episode[env_idx],
                                        self.step[env_idx])
                        self.logger.log('train/env_idx', env_tag,
                                        self.step[env_idx])

                    # sample action for data collection
                    if self.step[env_idx] < self.cfg.num_seed_steps:
                        action = env.action_space.sample()
                    else:
                        with utils.eval_mode(self.agent):
                            action = self.agent.act(obs[env_idx], sample=True)

                    next_obs[env_idx], reward, done[env_idx], _ = env.step(
                        action)
                    if train_recording and env_idx == env_to_rec:
                        self.train_video_recorder.record(env)

                    # allow infinite bootstrap
                    done[env_idx] = float(done[env_idx])
                    done_no_max = 0 if episode_step[
                        env_idx] + 1 == env._max_episode_steps - 1 else done[
                            env_idx]

                    episode_reward[env_idx] += reward

                    self.replay_buffer.add(env_idx, obs[env_idx], action,
                                           reward, next_obs[env_idx],
                                           done[env_idx], done_no_max)

                    obs[env_idx] = next_obs[env_idx]
                    episode_step[env_idx] += 1
                    self.step[env_idx] += 1

                    # Run training update
                    if self.step[env_idx] >= self.cfg.num_seed_steps:
                        #print('Running train update')
                        for _ in range(self.cfg.num_train_iters):
                            self.agent.update(self.replay_buffer,
                                              self.num_train_envs, self.logger,
                                              self.step[env_idx], env_tag,
                                              env_idx)
                # At the end of each episode, log
                self.logger.dump(
                    self.step[env_idx],
                    save=(self.step[env_idx] > self.cfg.num_seed_steps),
                    ty='train')


@hydra.main(config_name='configs/jaco_reach_site_features.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    from train import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
