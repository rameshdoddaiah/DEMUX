# -*- coding: utf-8 -*-
"""
| **@created on:** 12/19/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** 
"""

import numpy as np
from collections import deque, namedtuple
from torch.autograd import Variable
import torch


class PrioritizedBufferV1(object):
    def __init__(self, background_data, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.memory = background_data
        self.priorities = np.ones((len(background_data)), dtype=np.float32)
        self.softmax_fn = torch.nn.Softmax(dim=-1)

    def sample(self):
        prios = self.priorities
        # prios = prios ** self.prob_alpha
        probs = self.softmax_fn(torch.tensor(prios, dtype=torch.float32))
        index = np.random.choice(len(self.memory), 1, p=probs.numpy())[0]
        sample = self.memory[index]
        return sample, index

    def update_priorities(self, index, priority):
        self.priorities[index] = priority
