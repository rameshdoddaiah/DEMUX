from nte.masking.mask import Mask
import numpy as np


class SlidingMasks(Mask):
    def __init__(self, timesteps: int, window: list = [1, 2, 5, 10, 25], stride: list = [1, 2, 5, 10, 25]):
        super().__init__(timesteps=timesteps)
        self.window = window
        self.stride = stride
        self.masks = None

    def generate_masks(self):
        masks = [np.zeros(self.timesteps)]
        for w in self.window:
            for s in self.stride:
                masks[-1][:w] = 1
                for i in range(int(self.timesteps / (s)) - w):
                    masks.append(np.roll(masks[-1], s))
        self.masks = np.array(masks)
        return self.masks
