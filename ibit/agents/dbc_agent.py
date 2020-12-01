import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import utils
import hydra
from modules import Actor, Critic, ProbabilisticTransitionModel


class DBCAgent(object):
    """DBC algorithm with transition model."""
    def __init__(self, obs_shape, action_shape, lstate_shape, action_range,
                 device, encoder_cfg, critic_cfg, actor_cfg, discount,
                 init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size, penalty_type,
                 penalty_weight, penalty_anneal_iters):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.lstate_shape = lstate_shape
        self.penalty_type = penalty_type
        self.penalty_weight = penalty_weight
        self.penalty_anneal_iters = penalty_anneal_iters

        self.decoder_latent_lambda = .0

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.transition_model = ProbabilisticTransitionModel(
            encoder_cfg.params.feature_dim + self.lstate_shape,
            action_shape=action_shape,
            layer_width=512).to(self.device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_cfg.params.feature_dim + self.lstate_shape, 512),
            nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, 1)).to(device)

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
        decoder_params = list(self.transition_model.parameters()) + list(
            self.reward_decoder.parameters())
        self.decoder_optimizer = torch.optim.Adam(decoder_params, lr=lr)
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=lr)

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

    def update_transition_reward_encoder(self, obs, action, next_obs, reward,
                                         L, step, num_envs, logger):
        # Compute update for transition and reward models
        pix_obs = obs['pix_obs']
        next_pix_obs = next_obs['pix_obs']
        state_low_obs= obs['state_low_obs']
        next_state_low_obs = next_obs['state_low_obs']

        h = self.critic.encoder(pix_obs)
        if state_low_obs is not None:
            if len(state_low_obs.shape) > len(h.shape):
                state_low_obs = torch.squeeze(state_low_obs, 1)
            h = torch.cat((h, state_low_obs), axis=-1)

        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(
            torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.critic.encoder(next_pix_obs)
        if next_state_low_obs is not None:
            if len(next_state_low_obs.shape) > len(next_h.shape):
                next_state_low_obs = torch.squeeze(next_state_low_obs, 1)
            next_h = torch.cat((next_h, next_state_low_obs), axis=-1)

        diff_latent = (pred_next_latent_mu - next_h.detach())
        tran_loss = torch.mean(0.5 * diff_latent.pow(2))
        L.log('train_ae/transition_loss', tran_loss, step)

        pred_next_latent = self.transition_model.sample_prediction(
            torch.cat([h, action], dim=1))
        pred_next_reward = self.reward_decoder(pred_next_latent)

        ### Computing update for encoder using bisim metric ###
        # Sample random states across episodes at random
        batch_size = pix_obs.size(0)
        perm = np.random.permutation(batch_size)
        h2 = h[perm]
        curr_reward = reward[perm]
        with torch.no_grad():
            pred_next_h = self.transition_model.sample_prediction(
                torch.cat([h2, action], dim=1))

        env_batch_size = int(self.batch_size / num_envs)

        # Calculate penalties
        if self.penalty_type == 'rex':
            env_index = 0
            self.env_reward_losses = []
            for _ in range(num_envs):
                env_pred_next_reward = pred_next_reward[env_index:env_index +
                                                        env_batch_size]
                env_reward = reward[env_index:env_index + env_batch_size]
                env_reward_loss = F.mse_loss(env_pred_next_reward, env_reward)
                self.env_reward_losses.append(env_reward_loss)
                env_index += env_batch_size

            total_loss = torch.stack(self.env_reward_losses).mean() + tran_loss
        else:
            if self.penalty_type == 'irm':
                env_index = 0
                self.irm_penalties = []
                for _ in range(num_envs):
                    env_pred_next_reward = pred_next_reward[
                        env_index:env_index + env_batch_size]
                    env_reward = reward[env_index:env_index + env_batch_size]

                    self.irm_penalties.append(
                        utils.irm_penalty(env_pred_next_reward, env_reward))

                    env_index += env_batch_size

            reward_loss = F.mse_loss(pred_next_reward, reward)
            total_loss = tran_loss + reward_loss

        z_dist = F.smooth_l1_loss(h, h2, reduction='none')
        r_dist = F.smooth_l1_loss(reward, curr_reward, reduction='none')

        transition_dist = F.smooth_l1_loss(next_h,
                                           pred_next_h,
                                           reduction='none')

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        encoder_loss = (z_dist - r_dist - self.discount * transition_dist).pow(
            2).mean() + self.decoder_latent_lambda * latent_loss

        L.log('train_ae/env_encoder_loss', encoder_loss, step)

        total_loss = total_loss + encoder_loss

        # Add the penalties
        if self.penalty_type == 'rex':
            train_penalty = torch.var(torch.stack(self.env_reward_losses))
        elif self.penalty_type == 'irm':
            train_penalty = torch.stack(self.irm_penalties).mean()
        else:
            train_penalty = 0

        penalty_weight = (self.penalty_weight
                          if step >= self.penalty_anneal_iters else 1.0)
        logger.log('train_encoder/penalty', train_penalty, step)
        logger.log('train_encoder/penalty_weight', penalty_weight, step)
        logger.log('train_encoder/penalty_anneal_iters',
                   self.penalty_anneal_iters, step)
        logger.log('train_encoder/decoder_loss_bf_pen', total_loss, step)
        total_loss += penalty_weight * train_penalty
        if penalty_weight > 1.0 and self.penalty_type != 'erm':
            # Rescale the entire loss to keep gradients in a reasonable range
            total_loss /= penalty_weight

        return total_loss

    def update(self,
               replay_buffer,
               num_envs,
               logger,
               step,
               env_tag,
               env_id=None):
        all_env_obs = []
        all_env_state_low_obs = []
        all_env_obs_aug = []
        all_env_actions = []
        all_env_next_obs = []
        all_env_next_state_low_obs = []
        all_env_next_obs_aug = []
        all_env_reward = []
        all_not_dones = []

        decoder_loss = []
        batch_size = int(self.batch_size / num_envs)

        for env_id in range(num_envs):
            obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
                batch_size, env_tag, env_id)

            logger.log('train/batch_reward', reward.mean(), step)

            all_env_obs.append(obs['pix_obs'])
            all_env_obs_aug.append(obs_aug)
            all_env_actions.append(action)
            all_env_next_obs.append(next_obs['pix_obs'])
            if self.lstate_shape != 0:
                all_env_state_low_obs.append(obs['state_low_obs'])
                all_env_next_state_low_obs.append(next_obs['state_low_obs'])
            all_env_next_obs_aug.append(next_obs_aug)
            all_env_reward.append(reward)
            all_not_dones.append(not_done)

        all_env_obs = torch.cat(all_env_obs)
        all_env_obs_aug = torch.cat(all_env_obs_aug)
        all_env_actions = torch.cat(all_env_actions)
        all_env_next_obs = torch.cat(all_env_next_obs)
        all_env_next_obs_aug = torch.cat(all_env_next_obs_aug)
        all_env_reward = torch.cat(all_env_reward)
        all_not_dones = torch.cat(all_not_dones)

        if self.lstate_shape != 0:
            all_env_state_low_obs = torch.cat(all_env_state_low_obs)
            all_env_next_state_low_obs = torch.cat(all_env_next_state_low_obs)
        else:
            all_env_state_low_obs = None
            all_env_next_state_low_obs = None

        all_env_obs = {
            'pix_obs': all_env_obs,
            'state_low_obs': all_env_state_low_obs
        }
        all_env_next_obs = {
            'pix_obs': all_env_next_obs,
            'state_low_obs': all_env_next_state_low_obs
        }

        self.update_critic(all_env_obs, all_env_obs_aug, all_env_actions,
                           all_env_reward, all_env_next_obs,
                           all_env_next_obs_aug, all_not_dones, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(all_env_obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

        decoder_loss = self.update_transition_reward_encoder(
            all_env_obs, all_env_actions, all_env_next_obs, all_env_reward,
            logger, step, num_envs, logger)

        logger.log('train_encoder/decoder_loss_af_pen', decoder_loss, step)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        decoder_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

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

        torch.save(self.transition_model.state_dict(),
                   '%s/transition_model_%s.pt' % (model_dir, step))
        torch.save(self.reward_decoder.state_dict(),
                   '%s/reward_decoder_model_%s.pt' % (model_dir, step))
        torch.save(self.decoder_optimizer.state_dict(),
                   '%s/decoder_optim_%s.pt' % (model_dir, step))
        torch.save(self.encoder_optimizer.state_dict(),
                   '%s/encoder_optim_%s.pt' % (model_dir, step))

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

        self.transition_model.load_state_dict(
            torch.load('%s/transition_model_%s.pt' % (model_dir, step)))
        self.reward_decoder.load_state_dict(
            torch.load('%s/reward_decoder_model_%s.pt' % (model_dir, step)))
        self.decoder_optimizer.load_state_dict(
            torch.load('%s/decoder_optim_%s.pt' % (model_dir, step)))
        self.encoder_optimizer.load_state_dict(
            torch.load('%s/encoder_optim_%s.pt' % (model_dir, step)))

        self.log_alpha = torch.load('%s/log_alpha_%s.pt' % (model_dir, step))
        self.log_alpha_optimizer.load_state_dict(
            torch.load('%s/alpha_optim_%s.pt' % (model_dir, step)))