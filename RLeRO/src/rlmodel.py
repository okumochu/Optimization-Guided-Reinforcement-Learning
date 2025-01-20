from utils.jspenv import JSPEnv


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from datahandler import DataHandler

dh_ = DataHandler()
mi = dh_.generate_RL_model_input(
    last_sols=None, normal_input=None, inference=False)
_ = JSPEnv(mi)
s = _.reset()


N_ACTIONS = dh_.product_cnt
N_STATE_SPACE = _.state.to_numpy().shape[0]
LR_ACTOR = 0.000005
LR_CRITIC = 0.001
EPSILON = 0.9  # 0.6
TARGET_REPLACE_ITER = 100
GAMMA = 0.9
DENSE_SIZE = 150


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)


class Actor(nn.Module):
    def __init__(self) -> None:
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STATE_SPACE, DENSE_SIZE),  # all 50 before
            nn.ReLU(),
            nn.Linear(DENSE_SIZE, DENSE_SIZE),
            nn.ReLU(),
            nn.Linear(DENSE_SIZE, N_ACTIONS)
        )

    def forward(self, s):
        output = self.net(s)
        # print(output)
        output = F.softmax(output, dim=-1)  # TODO: check dim or log_softmax
        nan_cnt = sum(output.isnan().flatten()).numpy()
        if nan_cnt != 0:
            raise ValueError(f"{output} nan output")
        return output


class Critic(nn.Module):
    def __init__(self) -> None:
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STATE_SPACE, DENSE_SIZE),
            nn.ReLU(),
            nn.Linear(DENSE_SIZE, DENSE_SIZE),
            nn.ReLU(),
            nn.Linear(DENSE_SIZE, 1)
        )

    def forward(self, s):
        output = self.net(s)
        # print(output)
        return output


class A2C:
    def __init__(self):
        print(f"N_ACTIONS = {N_ACTIONS}")
        print(f"N_STATE_SPACE = {N_STATE_SPACE}")
        self.actor_net = Actor().apply(init_weights)
        self.critic_net = Critic().apply(init_weights)
        self.target_net = Critic().apply(init_weights)

        self.learn_step_count = 0
        self.optimizer_actor = optim.Adam(
            self.actor_net.parameters(), lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(
            self.critic_net.parameters(), lr=LR_CRITIC)
        self.criterion_critic = nn.MSELoss()

    def choose_action(self, s, inference=False):
        s = torch.unsqueeze(torch.FloatTensor(s), dim=0)
        # s = torch.nan_to_num(s)
        is_random = False
        entr_ = None
        if np.random.uniform() < EPSILON or inference:
            action_value = self.actor_net(s)
            entr_ = entropy(torch.detach(action_value).flatten().numpy())
            action = torch.max(action_value, dim=1)[1].item()
        else:
            is_random = True
            action = np.random.randint(0, N_ACTIONS)

        return [1 if i == action else 0 for i in range(N_ACTIONS)], is_random, entr_

    def learn(self, s, a, r, s_):
        if self.learn_step_count % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.critic_net.state_dict())
            # print(self.critic_net.state_dict())
        self.learn_step_count += 1

        s = torch.FloatTensor(s)
        # s = torch.nan_to_num(s)
        s_ = torch.FloatTensor(s_)
        # s_ = torch.nan_to_num(s_)

        q_actor = self.actor_net(s)
        q_critic = self.critic_net(s)
        q_next = self.target_net(s_).detach()
        q_target = r + GAMMA * q_next
        td_error = (q_critic - q_target).detach()

        loss_critic = self.criterion_critic(q_critic, q_target)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # + 0.1  # to avoid nan?
        log_q_actor = torch.log(q_actor)  # torch.add(torch.log(q_actor), 0.1)
        actor_loss = log_q_actor[a] * td_error
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # print(f"actor_loss: {actor_loss}")


# if __name__ == '__main__':
#     mi = dh.generate_RL_model_input()
#     env = JSPEnv(mi, )
#     a2c = A2C()

#     for episode in range(10000):
#         s = env.reset()
#         ep_r  = 0

#         while True:
#             a = a2c.choose_action(s)
#             s_, r, done, info = env.step(a)

#             ep_r += r

#             a2c.learn(s, a, r, s_)

#             if done:
#                 break
#             s = s_

#         print(f"Episode: {episode} | return: {ep_r}")
