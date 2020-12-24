import os
import numpy as np
import torch
from torch.optim import Adam

from .base import BaseAgent
from sacd.model import TwinnedQNetwork, CateoricalPolicy
from sacd.utils import disable_gradients


class SacdAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
<<<<<<< Updated upstream
                 cuda=True, seed=0):
=======
                 cuda=True, seed=0, render=False):

        #Инциируем sacd rl базу
>>>>>>> Stashed changes
        super().__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps,
            log_interval, eval_interval, cuda, seed)

        #Создаем актёра, которому передаём observation space и action space
        # Define networks.
        self.policy = CateoricalPolicy(
            self.env.observation_space.shape[0], self.env.action_space.n
            ).to(self.device)

        #Создаём двух критиков, в онлайн и таргет режиме
        self.online_critic = TwinnedQNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n,
            dueling_net=dueling_net).to(device=self.device)
        self.target_critic = TwinnedQNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n,
            dueling_net=dueling_net).to(device=self.device).eval()

        #Таргет критику даём веса онлайн критика
        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        #Говорим таргет критику что не надо считать градиенты
        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        #Создаём оптимизаторы для политики, а также для онлайн критиков (Q!)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

        #Идеальная энтропия когда все вероятности равны, или когда каждая вероятность имеет значение 1/кол-во возможных действий
        #  -log(1/a)
        #Умножаем на ratio, чтобы не перепрыгнуть случайно???
        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = \
            -np.log(1.0 / self.env.action_space.n) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.

        #Оптимизировать мы будем на сам альфа, а log-alpha, так как с loged значениями легче возиться
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        #Альфа -> exp(log_alpha)
        self.alpha = self.log_alpha.exp()

        #создаём оптимизатор лог альфы
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

    #Выбираем случайное из возможных учитывая вес каждого из действий
    def explore(self, state):
        # Act with randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()

    #Выбираем самое наиболее вероятное действие
    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()

    #Передаём таргету весы онлайн сети
    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    #Получаем q1 и q2 для всех возможных действий от критика
    #С помощью gather получаем q нужных action
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    #Считаем q-value для таргета
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        #Не считая градиенты
        with torch.no_grad():
            #Узнаем наше действие из сети политики для следующего состояния
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            #Узнаём q-value для следующего состояния
            next_q1, next_q2 = self.target_critic(next_states)
            #Узнаем следующее q как шанс того что политика выберет это действие помножное на qvalue этого действия + альфа * энтропия
            #Те действия, которые наименее вероятно исполняться, будут иметь больший log_action_prob и соответсвенно q-value, а те, которые наиболее вероятно исполняться, будут иметь 
            #меньший q-value. Таким образом мы даём шанс действиям, у которых была небольшая вероятность выполнения тоже шанс на выполнение.
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        #Вознаграждение + 1 - d * gamma * next_q
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    #Считаем потерю критика
    def calc_critic_loss(self, batch, weights):
        #Считаем q_value текущего состояния
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        #Считаем q_value последующего состояния
        target_q = self.calc_target_q(*batch)

        #Ошибки для TD
        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        #Средние q
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        #Считаем mse q
        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    #Считаем потерю для политики
    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # Берём actions политики
        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        #Без нахождения градиентов для онлайн критика (актёр получит градиенты) получаем q value
        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        #Считаем энтропию как минус сумму вероятности действия на log вероятности
        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        #Умножаем q-value на его вероятность и суммируем, таким образом можем посчитать ценность состояния
        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        
        
        #Наша задача максимизировать q + энтропия или другими словами, минимизировать  - q - self.alpha * entropy
        #Те элементы, у которых была самая большая ошибка, сильнее всего влияют на loss
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    #Считаем ошибку у энтропии
    def calc_entropy_loss(self, entropies, weights):
        #Энтропия не должно иметь градиент
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.

        # минимализация минус логарифма к увеличению альфы, если энтропия меньше целевой энтропии
        # min(-log_alpha) -> max(log_alpha) -> max(alpha) 
        # и к уменьшению, если энтропия больше целевой энтропии
        # min(log_alpha) -> min(log_alpha) -> min(alpha)

        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    #При сохранении модели сохраняем полиси, а также и критиков
    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))
