from collections import deque
import numpy as np
import torch

#Буффер для сохранения нескольких шагов
class MultiStepBuff:
    #Принимаем максимальное кол-во шагов
    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        #Обнулляем буффер
        self.reset()

    #Добавляем состояние действия и вознаграждение
    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    #Получаем первое на очереди состояние, действие и вознаграждение учитывая скидку на само вознаграждение
    def get(self, gamma=0.99):
        assert len(self.rewards) > 0
        state = self.states.popleft()
        action = self.actions.popleft()
        reward = self._nstep_return(gamma)
        return state, action, reward

    #Умножаем вознаграждение на гамму в степени дальности шага от текущего значения, таким образом делаем скидку по времени
    def _nstep_return(self, gamma):
        r = np.sum([r * (gamma ** i) for i, r in enumerate(self.rewards)])
        self.rewards.popleft()
        return r

    #При обнулении создаём 3 очереди для состояния, действия и вознаграждения
    def reset(self):
        # Buffer to store n-step transitions.
        self.states = deque(maxlen=self.maxlen)
        self.actions = deque(maxlen=self.maxlen)
        self.rewards = deque(maxlen=self.maxlen)

    #Проверяем пустой ли буффер
    def is_empty(self):
        return len(self.rewards) == 0

    #Проверяем полный ли буффер
    def is_full(self):
        return len(self.rewards) == self.maxlen

    def __len__(self):
        return len(self.rewards)

#Ленивая память
class LazyMemory(dict):
    #Принимает общий размер памяти
    #Форма состояния
    # И то, где это все будет храниться 
    def __init__(self, capacity, state_shape, device):
        super(LazyMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.device = device
        self.reset()

    #Пустоый список состояний
    #Следующих состояний
    #Массив действия, которые являются integer
    #А также вознаграждений создаём пустыми
    # n   
    def reset(self):
        self['state'] = []
        self['next_state'] = []

        self['action'] = np.empty((self.capacity, 1), dtype=np.int64)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)

        #Индекс для состояния
        self._n = 0
        #Индекс для ard
        self._p = 0

    #При добавллении вызываем скрытую функцию
    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        #В которой добавляем состояние следующее состояние в список, а также в ячейку p записываем действие вознаграждение и был ли закончен эпизод 
        self['state'].append(state)
        self['next_state'].append(next_state)
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        #При изменении состояния списка последний элемент всегда будет последний, так как первый элемент мы не перезаписываем а удаляем
        self._n = min(self._n + 1, self.capacity)
        #Тут мы первый элемент перезаписываем, поэтому считаем p как циклическую очередь
        self._p = (self._p + 1) % self.capacity

        #Убираем первое состояние
        self.truncate()

    def truncate(self):
        while len(self['state']) > self.capacity:
            del self['state'][0]
            del self['next_state'][0]

    #Получаем случайную ситуацию из памяти
    def sample(self, batch_size):
        #Выбираем индексы ситуации, которые могут быть от 0, до кол-ва состояний, размером batch size
        indices = np.random.randint(low=0, high=len(self), size=batch_size)
        #Получаем sample по индексу
        return self._sample(indices, batch_size)

    def _sample(self, indices, batch_size):
        #Если n равен capacicty, то p будет равно 0, поэтому делаем сдвиг до 0
        bias = -self._p if self._n == self.capacity else 0

        #Создаём массивы, куда будем записывать состояние
        states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)
        next_states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)

        #Для каждого индекса, добавляем состояние в массив состояний
        for i, index in enumerate(indices):
            _index = np.mod(index+bias, self.capacity)
            states[i, ...] = self['state'][_index]
            next_states[i, ...] = self['next_state'][_index]

        #Создаём финальные тензоры        
        states = torch.FloatTensor(states).to(self.device)/ 255.
        next_states = torch.FloatTensor(next_states).to(self.device)/ 255.
        actions = torch.LongTensor(self['action'][indices]).to(self.device)
        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n

#Создаём ленивую мультишаговую память, она добавляет скидку и сколько шагов надо сохранять
class LazyMultiStepMemory(LazyMemory):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3):
        super(LazyMultiStepMemory, self).__init__(
            capacity, state_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    #При добавления, если кол-во шагов обычное, то просто добавляем, если кол-во шагов больше одного, то добавляем шаг в буффер, и когда буффер заполнился до добавляем в обучную память уже со скидками
    def append(self, state, action, reward, next_state, done):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if self.buff.is_full():
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if done:
                while not self.buff.is_empty():
                    state, action, reward = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done)
        else:
            self._append(state, action, reward, next_state, done)
