import numpy as np
from collections import deque, namedtuple
from torch.autograd import Variable
import torch


class PrioritizedBuffer(object):
    def __init__(self, background_data, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        # self.capacity = capacity
        self.memory = background_data
        self.pos = 0
        self.priorities = np.ones((len(background_data),), dtype=np.float32)
        self.indices = np.ones((len(background_data)), dtype=np.float32)
        self.transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))

        self.weights = None
        self.softmax_fn = torch.nn.Softmax(dim=-1)

    # def push(self, *args):
    #     max_prio = self.priorities.max() if self.memory else 1.0
    #
    #     if len(self.memory) < self.capacity:
    #         self.memory.append(self.transition(*args))
    #     else:
    #         self.memory[self.pos] = self.transition(*args)
    #
    #     self.priorities[self.pos] = max_prio
    #     self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        # if len(self.memory) == self.capacity:
        prios = self.priorities
        # else:
        #     prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        # probs = probs.max() - probs + probs.min()
        # probs /= probs.sum()
        probs = self.softmax_fn(torch.tensor(probs, dtype=torch.float32)).numpy()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        self.weights = (total * probs[indices]) ** (-beta)
        # self.weights /= self.weights.max()
        self.weights = np.array(self.weights, dtype=np.float32)
        return samples, self.weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def finalize_priorities(self):
        self.priorities /= self.indices

    def initialize_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.indices[idx] += 1
            self.priorities[idx] += prio

    def __len__(self):
        return len(self.memory)
