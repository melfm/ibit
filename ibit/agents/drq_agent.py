import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import math

import utils
import hydra
from modules import Actor, Critic


class DRQAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self,
                 obs_shape,
                 action_shape,
                 lstate_shape,
                 action_range,
                 device,
                 encoder_cfg,
                 critic_cfg,
                 actor_cfg,
                 discount,
                 init_temperature,
                 lr,
                 actor_update_frequency,
                 critic_tau,
                 critic_target_update_frequency,
                 batch_size,
                 penalty_type=None,
                 penalty_weight=None,
                 penalty_anneal_iters=None):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.lstate_shape = lstate_shape

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        pix_obs = torch.FloatTensor(obs['pix_obs']).to(self.device)
        pix_obs = pix_obs.unsqueeze(0)
        state_low_obs = None
        if self.lstate_shape != 0:
            state_low_obs = torch.FloatTensor(obs['state_low_obs']).to(self.device)
            state_low_obs = torch.unsqueeze(state_low_obs, 0)
        dist = self.actor(pix_obs, state_low_obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, logger, step):
        pix_obs = obs['pix_obs']
        state_low_obs = None
        next_pix_obs = next_obs['pix_obs']
        next_state_low_obs = None
        if self.lstate_shape != 0:
            state_low_obs = obs['state_low_obs']
            next_state_low_obs = next_obs['state_low_obs']

        with torch.no_grad():
            dist = self.actor(next_pix_obs, next_state_low_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_pix_obs,
                                                      next_action,
                                                      next_state_low_obs)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            dist_aug = self.actor(next_obs_aug, next_state_low_obs)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                  keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                      next_action_aug,
                                                      next_state_low_obs)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(pix_obs, action, state_low_obs)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action, state_low_obs)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        pix_obs = obs['pix_obs']
        state_low_obs = obs['state_low_obs']
        dist = self.actor(pix_obs, state_low_obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(pix_obs,
                                         action,
                                         state_low_obs,
                                         detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, num_envs, logger, step, env_tag, env_id):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
            self.batch_size, env_tag, env_id)

        logger.log('train/batch_reward', reward.mean(), step)
        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def save(self, model_dir, step):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.actor.state_dict(),
                   '%s/actor_%s.pt' % (model_dir, step))
        torch.save(self.critic.state_dict(),
                   '%s/critic_%s.pt' % (model_dir, step))
        torch.save(self.actor_optimizer.state_dict(),
                   '%s/actor_optim_%s.pt' % (model_dir, step))
        torch.save(self.critic_optimizer.state_dict(),
                   '%s/critic_optim_%s.pt' % (model_dir, step))

        torch.save(self.log_alpha, '%s/log_alpha_%s.pt' % (model_dir, step))
        torch.save(self.log_alpha_optimizer.state_dict(),
                   '%s/alpha_optim_%s.pt' % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step)))
        self.actor_optimizer.load_state_dict(
            torch.load('%s/actor_optim_%s.pt' % (model_dir, step)))
        self.critic_optimizer.load_state_dict(
            torch.load('%s/critic_optim_%s.pt' % (model_dir, step)))

        self.log_alpha = torch.load('%s/log_alpha_%s.pt' % (model_dir, step))
        self.log_alpha_optimizer.load_state_dict(
            torch.load('%s/alpha_optim_%s.pt' % (model_dir, step)))