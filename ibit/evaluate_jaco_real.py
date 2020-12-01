import sys
sys.path.append('/home/melissa/Workspace/simrealrep/')
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

from jaco_real.jaco_physics import JacoPhysics

torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd().split('runs')[0] + 'runs/'
        self.work_dir = self.work_dir + \
            '2020.10.22/jaco_reach_site_features_drq_agent.cls=agents.drq_agent.DRQAgent,agent.name=drq,batch_size=64,lr=0.005/'
        self.model_dir = self.work_dir + '/agent_model'
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat,
                             overwrite=True)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # Environment Sampler
        self.num_train_envs = cfg.num_envs
        self.frame_stack = 5
        self.env_sampler = utils.EnvSampler(cfg,
                                            False,
                                            False,
                                            work_dir=self.work_dir)
        self.eval_env_sim = self.env_sampler.make_env()

        self.jaco_real_env = JacoPhysics('j2s7s300',
                                         robot_server_ip='127.0.0.1',
                                         robot_server_port=9030,
                                         control_type='position')
        self.frame_size = 84
        self.jaco_real_env = utils.FrameStackJacoReal(self.jaco_real_env,
                                                      k=self.frame_stack,
                                                      frame_size=self.frame_size,
                                                      dummy_env=self.eval_env_sim)

        cfg.agent.params.obs_shape = self.eval_env_sim.observation_space.shape
        cfg.agent.params.action_shape = self.eval_env_sim.action_space.shape
        cfg.agent.params.action_range = [
            float(self.eval_env_sim.action_space.low.min()),
            float(self.eval_env_sim.action_space.high.max())
        ]
        if cfg.lowobs_append:
            if cfg.env == 'jaco_reach_site_features':
                cfg.agent.params.lstate_shape = 49
            else:
                cfg.agent.params.lstate_shape = 9
        else:
            cfg.agent.params.lstate_shape = 0

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.sim_video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            dir_name='jaco_sim_video',
            phase='eval')
        self.real_video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            dir_name='jaco_real_video',
            phase='eval',
            height=640,
            width=480)

        self.reload_weights = cfg.reload_weights
        self.train_vid_interval = cfg.train_vid_interval

        self.num_eval_episodes = 1
        self.episode_max_step = 30
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for trial in range(self.num_eval_episodes):
            # This will send jaco to real home
            obs = self.jaco_real_env.reset()
            sim_obs = self.eval_env_sim.reset()
            obs['state_low_obs'] = sim_obs['state_low_obs']
            # Now lets go to sim home
            self.send_robot_to_sim_home()
            print('Done sending him home')
            self.sim_video_recorder.init(enabled=(trial == 0))
            self.real_video_recorder.init(enabled=(trial == 0))
            # What to do with done? Make sim to indicate done?
            # done = False
            episode_reward = 0
            episode_step = 0
            while (episode_step <= self.episode_max_step):
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                translated_act = self.translate_action_sim_to_real(action)

                obs = self.jaco_real_env.step(translated_act)
                obs['state_low_obs'] = sim_obs['state_low_obs']
                print('Translated Act ', translated_act)
                # Take a sim step with the original action
                sim_obs, reward, done, _ = self.eval_env_sim.step(action)

                self.sim_video_recorder.record(self.eval_env_sim)
                self.real_video_recorder.record(self.jaco_real_env, real_jaco=True)
                episode_reward += reward
                episode_step += 1
                # if done: break
            average_episode_reward += episode_reward
            self.sim_video_recorder.save(f'{trial}.mp4')
            self.real_video_recorder.save(f'{trial}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        print('Rewards ', average_episode_reward)

    def run(self):

        if os.path.exists(self.model_dir):
            latest_step = utils.get_latest_file(self.model_dir)
            self.agent.load(self.model_dir, latest_step)

        self.evaluate()

    def translate_action_sim_to_real(self, action):

        self.eval_env_sim.step(action)
        sim_qpos = self.eval_env_sim.physics.data.qpos
        return sim_qpos

    def send_robot_to_sim_home(self):
        self.eval_env_sim.reset()
        home_sim = self.eval_env_sim.physics.data.qpos
        self.jaco_real_env.step(home_sim)


@hydra.main(config_name='configs/jaco_reach_site_features.yaml')
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
