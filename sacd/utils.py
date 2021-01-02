from collections import deque
import numpy as np

#Функция обнуляет градиенты, считает новые и изменяет веса согласно новым градиентам
def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()

#Для каждого параметра считаем что градиенты равны 0
def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


#Простое скользящее среднее
class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)
