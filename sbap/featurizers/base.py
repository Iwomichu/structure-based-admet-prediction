import logging

import numpy.typing as npt

from abc import ABC, abstractmethod

from sbap.sdf import ChemblSdfRecord


class BaseFeaturizer(ABC):
    def __init__(self, logging_level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)

    @abstractmethod
    def fit(self, records: list[ChemblSdfRecord]) -> None:
        pass

    @abstractmethod
    def transform(self, records: list[ChemblSdfRecord]) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        pass
