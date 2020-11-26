import abc

class AbstractManager(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, ep_ret: float):
        pass
