from abc import ABCMeta, abstractmethod

import torch
from mgds.MGDS import MGDS, TrainDataLoader

from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin


class BaseDataLoader(
    DataLoaderMgdsMixin,
    metaclass=ABCMeta,
):

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
    ):
        super(BaseDataLoader, self).__init__()

        self.train_device = train_device
        self.temp_device = temp_device

    @abstractmethod
    def get_data_set(self) -> MGDS:
        pass

    @abstractmethod
    def get_data_loader(self) -> TrainDataLoader:
        pass
