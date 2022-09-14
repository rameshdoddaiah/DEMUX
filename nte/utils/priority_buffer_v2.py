import numpy as np
from collections import deque, namedtuple
from torch.autograd import Variable
import torch


class PrioritizedBufferV2(object):
    def __init__(self, background_data, args, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        # self.capacity = capacity
        self.memory = background_data
        self.pos = 0
        self.priorities = np.ones((len(background_data),), dtype=np.float32)
        self.transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))
        self.args = args
        self.weights = None

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
        # print("Pri: ", self.priorities.min(), self.priorities.max(), self.priorities.var())
        prios = self.priorities
        # else:
        #     prios = self.priorities[:self.pos]

        prios = prios **self.prob_alpha
        # probs =
        # print(sorted(prios))
        # probs = probs.max() - probs + probs.min()
        # probs_1 = prios/ prios.sum()
        probs = torch.nn.Softmax(dim=-1)(torch.tensor(prios, dtype=torch.float32)).numpy()
        # print("Prob: ", probs.min(), probs.max(), probs.var())
        # print("Prob 1: ", probs_1.min(), probs_1.max(), probs_1.var())
        # print(self.priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        self.weights = (total * prios[indices]) ** (-beta)
        # print(self.weights)
        # self.weights /= self.weights.max()
        self.weights = np.array(self.weights, dtype=np.float32)
        return samples, prios[indices] if prios[indices]>0 else [0], indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] =  np.clip(prio, 0, None)



    def __len__(self):
        return len(self.memory)