from nte.masking.mask import Mask
import numpy as np
from nte.data.dataset import Dataset


class RandomMasks(Mask):
    def __init__(self, timesteps: int, number_of_masks: int):
        Mask.__init__(self, timesteps=timesteps)
        self.number_of_masks = number_of_masks
        self.masks = None

    def generate_masks(self):
        self.masks = np.random.choice([0, 1], size=[self.number_of_masks, self.timesteps])
        return self.masks

    def generate_uniform_masks(self, new_to_shuffle_ratio=0.1):
        self.new_to_shuffle_ratio = new_to_shuffle_ratio
        self.new_counter = int(self.number_of_masks * self.new_to_shuffle_ratio)
        masks = []
        for i in range(self.new_counter):
            cur_array = np.random.choice([0, 1], size=[self.timesteps],
                                         p=[i / self.new_counter, 1 - i / self.new_counter])
            for j in range(int(self.number_of_masks / self.new_counter)):
                np.random.shuffle(cur_array)
                masks.append(cur_array)
        self.masks = np.array(masks)
        return self.masks
