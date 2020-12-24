from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sacd.memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
from sacd.utils import update_params, RunningMeanStats


class BaseAgent(ABC):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, num_eval_steps=125000, max_episode_steps=27000,
                 log_interval=10, eval_interval=1000, cuda=True, seed=0):
        super().__init__()

        #ENV для обучения
        self.env = env
        #энв для тестов
        self.test_env = test_env

        # Set seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        #Для тестовой другой сид
        self.test_env.seed(2**31-1-seed)

        #Бенчмарк нужен детерминирование не нужно
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        #Ставим куду если она есть и сказали её использовать
        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")


        #Если сказали использовать приоритетную память -> используем приоритетную (чем больше разница между q модели и q правильным, тем чаще смотрим туда)
        # LazyMemory efficiently stores FrameStacked states.
        if use_per:
            #(Всего шагов - начальные шаги)/количество интервалов для обновлений -> сколько шагов в одном интервале
            beta_steps = (num_steps - start_steps) / update_interval
            #Создаём память, определённого размера
            # с определённым шейпом состояния на определённым девайсе
            # с определённой гаммой (скидка будущего шага)
            # multi step (????)
            # шагов в одном интервале перед обновлением
            self.memory = LazyPrioritizedMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.observation_space.shape,
                device=self.device, gamma=gamma, multi_step=multi_step,
                beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.observation_space.shape,
                device=self.device, gamma=gamma, multi_step=multi_step)

        #Папка куда логи кидаем
        self.log_dir = log_dir
        #Папка куда модель киннем
        self.model_dir = os.path.join(log_dir, 'model')
        #Папка куда кинем как модель на тренилась
        self.summary_dir = os.path.join(log_dir, 'summary')
        #Если папок нет, то создаём
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        #Создаём tensorboard логгер
        self.writer = SummaryWriter(log_dir=self.summary_dir)

        #Создаём штуковину, которая следит за средним значением
        self.train_return = RunningMeanStats(log_interval)

        #Кол-во шагов
        self.steps = 0

        #Кол-во шагов обучения
        self.learning_steps = 0

        #Кол-во эпизодов
        self.episodes = 0

        #Лучший результат
        self.best_eval_score = -np.inf

        #Всего шагов
        self.num_steps = num_steps

        #Размер батча
        self.batch_size = batch_size

        #Гамму последнего шага
        self.gamma_n = gamma ** multi_step

        #Кол-во начальных шагов
        self.start_steps = start_steps

        #Кол-во интервалов для обновления
        self.update_interval = update_interval

        #Кол-во интервалов для апдейта таргет сети
        self.target_update_interval = target_update_interval

        #Используем ли приоритетную сетка
        self.use_per = use_per

        #Кол-во шагов для оценки
        self.num_eval_steps = num_eval_steps

        #Максимальное кол-во шагов в эпизоде
        self.max_episode_steps = max_episode_steps

        #Раз во сколько шагов
        self.log_interval = log_interval
        self.eval_interval = eval_interval

    def run(self):
        #Тренируем эпизоды пока кол-во шагов не будет больше кол-ва шагов обучения
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    #Если в данный шаг мы должны обновиться и данный шаг не является начальным шагом, то нам надо обновиться
    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_critic_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies, weights):
        pass

    #Тренировка эпизода
    def train_episode(self):
        #Увеличиваем итератор эпизодов на 1
        self.episodes += 1

        #Заработок от этого эпизода
        episode_return = 0.

        #Кол-во шагов в эпизоде
        episode_steps = 0

        #Закончился ли эпизод
        done = False

        #Получаем изначальное состояние и ресетим энв
        state = self.env.reset()

        #Если шаг не конечный и кол-во пройденных шагов меньше чем максимально допустимое кол-во шагов
        while (not done) and episode_steps <= self.max_episode_steps:

            #Если мы ещё в стартовых шагах то рандомим действие
            if self.start_steps > self.steps:
                action = self.env.action_space.sample()
            #Если не в стартовых то спрашиваем у модели через исследование
            else:
                action = self.explore(state)

            #Получаем следующее состояние у модели, а также вознаграждение и информацию закончился 
            next_state, reward, done, _ = self.env.step(action)

            # Clip reward to [-1.0, 1.0].
            #Обрезаем вознаграждение, чтобы не выходило за рамки
            clipped_reward = max(min(reward, 1.0), -1.0)

            #Добавляем в память состояние действие, вознаграждение, следующее состояние и является ли этот шаг последним
            # To calculate efficiently, set priority=max_priority here.
            self.memory.append(state, action, clipped_reward, next_state, done)

            #Увеличиваем общее кол-во шагов и шагов в этом эпизоде, прибавляем вознаграждение этого шага к вознаграждениям за эпизод, обновляем состояние
            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            #Учим актёров и критиков (????)
            if self.is_update():
                self.learn()

            #Обновляем критиков
            if self.steps % self.target_update_interval == 0:
                self.update_target()

            #Оцениваем модель и сохраняем
            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models(os.path.join(self.model_dir, 'final'))

        # We log running mean of training rewards.
        self.train_return.append(episode_return)

        #Если данный эпизод нужно логить, записываем его
        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_return.get(), self.steps)

        #Печатаем номер эпизода, кол-во шагов в эпизоде и вознаграждение з
        print(f'Episode: {self.episodes:<4}  '
              f'Episode steps: {episode_steps:<4}  '
              f'Return: {episode_return:<5.1f}')

    #Обычение моделли
    #Критиков обучаем, чтобы лучше предсказывали q-value
    #Актёра обучаем, чтобы он сильнее всего радовал критиков
    #Также критикам нравится, когда актёр делает действия с большей энтропией
    def learn(self):
        #У нас должен быть q1 q2 policy и альфа оптимизатор
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
            hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        #Увеличиваем кол-во шагов обучения
        self.learning_steps += 1

        #Если используем приоритетный сэмплинг то получаем батч и веса
        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        #Если не приоритезированные, то просто сэмплим батчи с весом 1
        else:
            batch = self.memory.sample(self.batch_size)
            # Set priority weights to 1 when we don't use PER.
            weights = 1.

        #Считаем ошибку критиков, на сколько критики ошиблись и средий q-value
        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(batch, weights)

        #Считаем ошибку политики, также получаем энтропию, которая зависит от размаха Гауссианы
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        #Считаем ошибку энтропии
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        #Обновляем с данными для каждого оптимизатора из 4
        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        #Получаем текущую альфу (энтропию в экспоненте)
        self.alpha = self.log_alpha.exp()

        #Обновляем приоритеты с помощью новых ошибок
        if self.use_per:
            self.memory.update_priority(errors)

        #Логгируем
        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    #Оцениваем работы моделей
    def evaluate(self):
        #Кол-во эпизодов
        num_episodes = 0
        #Кол-во шагов
        num_steps = 0
        #Общий
        total_return = 0.0

        while True:
            #Получаем состояние
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                #Получаем действие уже через эксплоит -> среднее 
                action = self.exploit(state)
                next_state, reward, done, _ = self.test_env.step(action)
<<<<<<< Updated upstream
=======

                if self.render:
                    self.test_env.render()

>>>>>>> Stashed changes
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        #Считаем среднее вознаграждение за эпизод
        mean_return = total_return / num_episodes

        #Если среднее вознаграждение больше наибольшего вознаграждения, сохраняем модель
        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        #Записываем среднее вознаграждение
        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    #При удалении объекта закрываем env и tensorboard writer
    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()
