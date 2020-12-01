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
from video import VideoRecorder
from omegaconf import DictConfig

torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd().split('runs')[0] + 'runs/'
        self.work_dir = self.work_dir + \
            '2020.10.21/jaco_reach_site_features_drq_agent.cls=agents.drq_agent.DRQAgent,agent.name=drq,batch_size=64,lr=0.005/seed=0/'
        self.model_dir = self.work_dir + '/agent_model'
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.log_eval_dir = self.work_dir + '/eval_standalone'
        # Use a separate eval dir to avoid overwriting training files
        if not os.path.exists(self.log_eval_dir):
            os.makedirs(self.log_eval_dir)
        self.logger = Logger(self.log_eval_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat,
                             overwrite=True)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # Environment Sampler
        self.num_train_envs = cfg.num_envs
        self.env_sampler = utils.EnvSampler(cfg,
                                            False,
                                            False,
                                            work_dir=self.work_dir)
        experiment_identifier = self.work_dir.split('runs')[1]
        self.eval_envs = self.env_sampler.sample_eval_envs(
            experiment_identifier)
        env_sample_key = list(self.eval_envs.keys())[0]
        sample_env = self.eval_envs[env_sample_key]
        cfg.agent.params.obs_shape = sample_env.observation_space.shape
        cfg.agent.params.action_shape = sample_env.action_space.shape
        cfg.agent.params.action_range = [
            float(sample_env.action_space.low.min()),
            float(sample_env.action_space.high.max())
        ]
        if cfg.lowobs_append:
            if cfg.env == 'jaco_reach_site_features':
                cfg.agent.params.lstate_shape = 49
            else:
                cfg.agent.params.lstate_shape = 9
        else:
            cfg.agent.params.lstate_shape = 0

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.render_train_samples = True
        if self.render_train_samples:
            if cfg.env.startswith('jaco'):
                height = 256
                width = 256
            else:
                height = width = 500
            from PIL import Image
            for env_idx, env in self.eval_envs.items():
                name = 'StandAloneEval_Unseen_Environment_' + str(
                    env_idx) + '.png'
                img_path = self.work_dir + name
                env.reset()
                obs = env.render(mode='rgb_array', height=height, width=width)
                im = Image.fromarray(obs)
                im.save(img_path)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None, phase='eval_standalone')

        self.reload_weights = cfg.reload_weights
        self.train_vid_interval = cfg.train_vid_interval

        self.eval_trials = 100
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.eval_trials):
            print('Episode Trial ', episode)
            self.video_recorder.init(enabled=True)
            eval_env = self.eval_envs[random.sample(list(self.eval_envs),
                                                    1)[0]]
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            while (episode_step <= eval_env._max_episode_steps - 1):
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, _, _ = eval_env.step(action)
                self.video_recorder.record(eval_env)
                episode_reward += reward
                episode_step += 1
                self.step += 1
                if done: break
            average_episode_reward += episode_reward
            print('Episode Reward ', episode_reward)
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.eval_trials
        self.logger.log('eval_standalone/episode_reward',
                        average_episode_reward, self.step)
        self.logger.dump(self.step, ty='eval_standalone')

    def run(self):
        if os.path.exists(self.model_dir):
            latest_step = utils.get_latest_file(self.model_dir)
            self.agent.load(self.model_dir, latest_step)
        else:
            raise ValueError('Could not reload weights!')

        self.evaluate()


@hydra.main(config_name='configs/jaco_reach_site_features.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    from evaluate_sim import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
