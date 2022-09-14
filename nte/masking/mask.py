from abc import ABCMeta, abstractmethod


class Mask(metaclass=ABCMeta):
    def __init__(self, timesteps: int, *args, **kwargs):
        self.timesteps = timesteps
        self.masks = None

    @abstractmethod
    def generate_masks(self):
        pass

    def batch(self, batch_size=32):
        """
        Function to batch the data
        :param batch_size: batches
        :return: batches of masks
        """
        l = len(self.masks)
        for ndx in range(0, l, batch_size):
            yield self.masks[ndx:min(ndx + batch_size, l)]
