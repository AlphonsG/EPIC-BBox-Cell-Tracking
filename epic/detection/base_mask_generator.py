from abc import ABC, abstractmethod


class BaseMaskGenerator(ABC):

    @abstractmethod
    def gen_mask(self, frame, coords):
        raise NotImplementedError()
