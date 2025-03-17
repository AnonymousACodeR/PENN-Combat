from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from agent.models.layer.PESymetry import PESymetryMean

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## RPPO Network ##################################
class PEEmbedding(nn.Module):
    def __init__(self, state_dim, h_dim):
        super(PEEmbedding, self).__init__()
        self.pe_embedding = PESymetryMean(state_dim, h_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pe_embedding(x)
        return h


class PEActor(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim):
        super(PEActor, self).__init__()
        actor_pe_layers = [
            nn.ELU(),
            PESymetryMean(h_dim, h_dim),
            nn.ELU(),
            PESymetryMean(h_dim, action_dim),
        ]
        self.actor = nn.Sequential(*actor_pe_layers)

    def forward(self, x: torch.Tensor, act_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.actor(x)

        if act_mask is not None:
            x = x * act_mask + (1 - 1 / act_mask)
        x = nn.Softmax(dim=-1)(x)
        return x


class PECritic(nn.Module):
    def __init__(self, state_dim, h_dim):
        super(PECritic, self).__init__()
        self.critic = nn.Sequential(
            nn.ELU(),
            PESymetryMean(h_dim, h_dim),
            nn.ELU(),
            PESymetryMean(h_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.critic(x)
        return x


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.act_masks = []
        self.states = []
        self.hidden_states = []
        self.next_states = []
        self.logprobs = []
        self.next_hidden_states = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.act_masks[:]
        del self.states[:]
        del self.hidden_states[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.next_hidden_states[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PEActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim, has_continuous_action_space, action_std_init):
        super(PEActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        # Embedding
        self.pe_embedding = PEEmbedding(state_dim, h_dim)
        # Actor
        self.actor = PEActor(state_dim, action_dim, h_dim)
        # Critic
        self.critic = PECritic(state_dim, h_dim)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state, act_mask=None):
        h_state = self.pe_embedding(state)

        if self.has_continuous_action_space:
            raise NotImplementedError
        else:
            if act_mask is not None:
                action_probs = self.actor(
                    h_state, act_mask)
            else:
                action_probs = self.actor(h_state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(h_state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action, act_mask=None):

        if self.has_continuous_action_space:
            raise NotImplementedError
        else:
            h_state = self.pe_embedding(state)
            action_probs = self.actor(h_state, act_mask)
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            state_values = self.critic(h_state)

        return action_logprobs, state_values, dist_entropy


class PEPPO:
    def __init__(self, state_dim, action_dim, h_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = (PEActorCritic(state_dim, action_dim, h_dim, has_continuous_action_space, action_std_init)
                       .to(device))
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = (PEActorCritic(state_dim, action_dim, h_dim, has_continuous_action_space, action_std_init)
                           .to(device))
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, act_mask=None):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                if act_mask is not None:
                    act_mask = torch.tensor(act_mask, dtype=torch.float32).to(device)
                    action, action_logprob, state_val = self.policy_old.act(state, act_mask=act_mask)
                else:
                    action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.act_masks.append(act_mask)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal, next_state in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals),
                                                   reversed(self.buffer.next_states)):
            next_h_state = self.policy.pe_embedding(next_state.to(device))
            next_state_value = self.policy.critic(next_h_state).detach()
            if is_terminal:
                discounted_reward = torch.zeros(next_state_value.shape).to(device)
                next_state_value = torch.zeros(next_state_value.shape).to(device)
            discounted_reward = (torch.FloatTensor(reward).to(device).reshape(next_state_value.shape) + next_state_value
                                 + (self.gamma * discounted_reward))
            # discounted_reward = reward + next_state_value + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.stack(rewards, dim=0).squeeze().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)

        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        if self.buffer.act_masks[0] is None:
            old_act_masks = None
        else:
            old_act_masks = torch.squeeze(torch.stack(self.buffer.act_masks, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_act_masks)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # clear buffer
        self.buffer.clear()

        return loss.mean().item()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class PEPPO_Inf:
    def __init__(self, state_dim, action_dim, h_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.policy_old = PEActorCritic(state_dim, action_dim, h_dim, has_continuous_action_space, action_std_init).to(
            device)

    def select_action(self, state, act_mask=None):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state, act_mask=act_mask)

            return action

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def load_from_model(self, model):
        self.policy_old.load_state_dict(model.state_dict())
